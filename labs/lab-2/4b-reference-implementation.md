# Lab Guide: Fine-Tuning and Serving Gemma 3 4B on TPU

This guide walks through the full pipeline for your team: fine-tune Gemma 3 4B with LoRA on a hermes function-calling dataset, merge the adapter into the base weights, and serve the result with vLLM on a TPU v5e slice — then confirm it's working with a live inference request. Every file you need is created from scratch by the commands below; you don't need any pre-existing lab files.

## Environment

| Item | Value |
|---|---|
| GCP project | `soe-hpccenter` |
| GKE cluster | `class-tpu-cluster-west4` |
| Region | `us-west4` |
| kubectl context | `gke_soe-hpccenter_us-west4_class-tpu-cluster-west4` |
| TPU accelerator | `tpu-v5-lite-podslice`, topology `2x4` (8 chips) |
| GCS bucket | `gs://me344-tpu-labs-west4` |
| Base model path | `/gcs/models/gemma-3-4b-it` (already staged — shared by all teams) |
| Dataset path | `/gcs/data/hermes-fc` (already staged — shared by all teams) |
| Artifact Registry | `us-central1-docker.pkg.dev/soe-hpccenter/tpu-images/` |

All teams share the same cluster and the same Kubernetes namespace (`default`) — there is no per-team namespace isolation. Your resources are kept separate purely by **name**, via the `TEAM` value you set below. **Your team name must be unique across the whole class** — if two teams pick the same value, their jobs, checkpoints, and deployments will collide and overwrite each other.

## Step 0 — Set up your session

Verify you're pointed at the right cluster:
```bash
kubectl config current-context
```
It should print `gke_soe-hpccenter_us-west4_class-tpu-cluster-west4`. If it doesn't (this can happen if you log back in after a break), switch to it:
```bash
kubectl config use-context gke_soe-hpccenter_us-west4_class-tpu-cluster-west4
```

Choose your team identifier (e.g. `team-<your-sunet-id>` or a name your instructor assigns) and export the full set of variables used throughout this lab. **Re-run this after every fresh login** — these are session-scoped and won't persist:

```bash
export TEAM=<your-team-name>
export REGION=us-central1
export WORKSHOP_BUCKET=me344-tpu-labs-west4
export IMAGE_URI_4B=us-central1-docker.pkg.dev/soe-hpccenter/tpu-images/gemma3-4b-finetune-${TEAM}:latest
```

---

## Step 1 — Build your fine-tuning image

Create the build directory and its two files:

```bash
mkdir -p tunix-build-4b

cat > tunix-build-4b/Dockerfile <<'EOF'
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.11.26 /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
# Pins below are required for Gemma 3 + Tunix main — copy as-is.
RUN printf 'transformers==5.13.1\n' > /tmp/override.txt && \
    uv pip install --system --no-cache --override /tmp/override.txt \
    'jax[tpu]==0.10.2' \
    'google-tunix[prod] @ git+https://github.com/google/tunix@c00f424b47964758361869233e72ca3b1a73119f' \
    'qwix @ git+https://github.com/google/qwix@03b4c1d5040cfa4165f14930b13351a5585717f4' \
    datasets==5.0.0 'kagglesdk<0.1.24' 'wandb==0.28.0' \
    -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
COPY tunix_sft_main.py /app/tunix_sft_main.py
WORKDIR /app
EOF

cat > tunix-build-4b/tunix_sft_main.py <<'EOF'
"""LoRA-fine-tune Gemma 3 4B on hermes function-calling, on a single-host TPU slice.

Reads model + data from the FUSE-mounted bucket (/gcs), trains with Tunix PeftTrainer, optionally
streams metrics to Weights & Biases, and exports the LoRA adapter as safetensors for Lab 3.
"""

import dataclasses
import json
import logging
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import qwix
from datasets import load_from_disk
from flax import nnx
from orbax import checkpoint as ocp
from safetensors.numpy import save_file
from transformers import AutoConfig, AutoTokenizer

from tunix.models.gemma3 import model as gemma3_lib
from tunix.models.gemma3 import params as gemma3_params
from tunix.models.gemma3 import params_safetensors as params_safetensors_lib
from tunix.models.safetensors_saver import join_path
from tunix.rl import reshard
from tunix.sft import peft_trainer, utils as sft_utils
from tunix.sft.metrics_logger import MetricsLoggerOptions

TEAM         = os.environ["TEAM"]
GCS          = Path("/gcs")
MODEL_PATH   = GCS / "models" / "gemma-3-4b-it"
DATA_PATH    = GCS / "data" / "hermes-fc"
CKPT_PATH    = GCS / "teams" / TEAM / "lora-ckpt-4b"
ADAPTER_PATH = GCS / "teams" / TEAM / "hermes-lora-adapter-4b"

BATCH_SIZE = 8
MAX_LEN    = 512
MAX_STEPS  = 100
LORA_RANK  = 32
LORA_ALPHA = 64.0
LR         = 1e-3

WANDB_PROJECT = "getting-started-with-tpu-on-gcp"
WANDB_RUN     = "gemma3-4b-hermes-lora"

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logging.getLogger("orbax").setLevel(logging.WARNING)
log = logging.getLogger("tunix-sft").info

log("TPU devices: %d | %s", jax.device_count(), [d.coords for d in jax.devices()])
MESH = [(jax.device_count(), 1), ("fsdp", "tp")]
mesh = jax.make_mesh(*MESH, axis_types=(jax.sharding.AxisType.Auto,) * len(MESH[0]))
log("Mesh: %s — loading %s (FSDP-%d, ~2-3 min from FUSE)…",
    dict(zip(MESH[1], MESH[0])), MODEL_PATH, MESH[0][0])

config = gemma3_lib.ModelConfig.gemma3_4b_it()
hf_config = AutoConfig.from_pretrained(str(MODEL_PATH))
hf_vocab_size = getattr(getattr(hf_config, "text_config", hf_config), "vocab_size")
if hf_vocab_size != config.num_embed:
    log("Overriding Tunix num_embed: %d -> checkpoint vocab_size %d", config.num_embed, hf_vocab_size)
    config = dataclasses.replace(config, num_embed=hf_vocab_size)
list(MODEL_PATH.iterdir())  # prime gcsfuse implicit-dirs cache before Tunix glob
with jax.set_mesh(mesh):
    base_model = params_safetensors_lib.create_model_from_safe_tensors(str(MODEL_PATH), config, mesh)
log("Base model loaded onto mesh.")

lora_provider = qwix.LoraProvider(
    module_path=".*q_einsum|.*kv_einsum|.*attn_vec_einsum|.*gate_proj|.*down_proj|.*up_proj",
    rank=LORA_RANK,
    alpha=LORA_ALPHA,
)
# Trace inputs matching Gemma3.__call__ — note the first arg is `last_tokens`, not `tokens`.
B, T = jax.device_count(), 8
trace_input = dict(
    last_tokens=jnp.ones((B, T), dtype=jnp.int32),
    positions=jnp.tile(jnp.arange(T)[None, :], (B, 1)),
    cache=None,
    attention_mask=jnp.tril(jnp.ones((B, T, T), dtype=jnp.bool_)),
)
log("Wrapping with LoRA (rank=%d, alpha=%.1f) — tracing forward pass…", LORA_RANK, LORA_ALPHA)
with jax.set_mesh(mesh):
    lora_model = qwix.apply_lora_to_model(base_model, lora_provider, rngs=nnx.Rngs(0), **trace_input)
del base_model
lora_model = reshard.reshard_model_to_mesh(lora_model, mesh)

n_lora = sum(p.size for p in jax.tree.leaves(nnx.state(lora_model, nnx.LoRAParam)))
assert n_lora > 0, "qwix matched zero modules — check module_path regex"
log("LoRA adapters injected and sharded — %.1fM trainable params.", n_lora / 1e6)

remat_cfg = dataclasses.replace(config, remat_config=gemma3_lib.RematConfig.DECODER)
lora_model.config = remat_cfg
for layer in lora_model.layers:
    layer.config = remat_cfg
log("Enabled RematConfig.DECODER on the LoRA-wrapped model.")

log("Loading tokenizer from %s and dataset from %s…", MODEL_PATH, DATA_PATH)
hf_tok = AutoTokenizer.from_pretrained(str(MODEL_PATH))
raw = load_from_disk(str(DATA_PATH))["train"].train_test_split(test_size=0.05, seed=0)
log("Tokenizing %d train / %d val examples (max_len=%d)…", len(raw["train"]), len(raw["test"]), MAX_LEN)

ROLE = {"system": "user", "human": "user", "gpt": "model", "tool": "user"}

def tokenize(split):
    texts = [
        "".join(f"<start_of_turn>{ROLE[m['from']]}\n{m['value']}<end_of_turn>\n"
                for m in ex["conversations"])
        for ex in split
    ]
    enc = hf_tok(texts, max_length=MAX_LEN, padding="max_length", truncation=True, return_tensors="np")
    return jnp.asarray(enc.input_ids), jnp.asarray(enc.attention_mask, dtype=jnp.bool_)

def to_batches(split):
    tokens, mask = tokenize(split)
    for i in range(0, len(tokens) - BATCH_SIZE + 1, BATCH_SIZE):
        tok, m = tokens[i:i + BATCH_SIZE], mask[i:i + BATCH_SIZE]
        yield {
            "input_tokens": tok,
            "input_mask": m,
            "positions": sft_utils.build_positions_from_mask(m),
            "attention_mask": sft_utils.make_causal_attn_mask(m),
        }

train_ds, val_ds = list(to_batches(raw["train"])), list(to_batches(raw["test"]))
log("Batches ready — train=%d val=%d shape=%s", len(train_ds), len(val_ds), train_ds[0]["input_tokens"].shape)

cfg = dict(
    eval_every_n_steps=20,
    max_steps=MAX_STEPS,
    checkpoint_root_directory=str(CKPT_PATH),
    checkpointing_options=ocp.CheckpointManagerOptions(save_interval_steps=20, max_to_keep=3),
)
if os.environ.get("WANDB_API_KEY"):
    cfg["metrics_logging_options"] = MetricsLoggerOptions(
        log_dir=str(CKPT_PATH / "logs"),
        project_name=WANDB_PROJECT,
        run_name=f"{WANDB_RUN}-{TEAM}",
        backend_kwargs={"wandb": {"tags": ["tunix", "gemma3-4b", "lora", f"v5e-{jax.device_count()}"]}},
    )
training_config = peft_trainer.TrainingConfig(**cfg)
trainer = peft_trainer.PeftTrainer(lora_model, optax.adamw(LR), training_config)
log("PeftTrainer ready. Checkpoints: %s", CKPT_PATH)
log("Starting train: %d steps, eval every 20. The FIRST step is an XLA compile (silence is normal)…", MAX_STEPS)
with jax.set_mesh(mesh):
    trainer.train(train_ds, val_ds)
log("Training complete — Orbax training checkpoint at %s", CKPT_PATH)

lora_layers: dict[str, list] = {}
for path, value in nnx.iter_graph(lora_model):
    if isinstance(value, nnx.LoRAParam):
        path_str = join_path(path[:-1])
        if path_str in lora_layers:
            assert "lora_b" in str(path[-1]), f"expected lora_b second, got {path[-1]}"
            lora_layers[path_str].append(np.asarray(value.value))
        else:
            assert "lora_a" in str(path[-1]), f"expected lora_a first, got {path[-1]}"
            lora_layers[path_str] = [np.asarray(value.value)]

def to_peft(name, t):
    arr = np.asarray(t)
    return arr.reshape(-1, LORA_RANK).T if name == "lora_A" else arr.reshape(LORA_RANK, -1).T

adapter = {}
for key, (lora_a, lora_b) in gemma3_params._extract_gemma3_lora_layers(lora_layers).items():
    hf = gemma3_params._gemma3_state_key_to_safetensors_key(key).removesuffix(".weight")
    adapter[f"{hf}.lora_A.weight"] = np.ascontiguousarray(to_peft("lora_A", lora_a))
    adapter[f"{hf}.lora_B.weight"] = np.ascontiguousarray(to_peft("lora_B", lora_b))

ADAPTER_PATH.mkdir(parents=True, exist_ok=True)
save_file(adapter, str(ADAPTER_PATH / "adapter_model.safetensors"))
(ADAPTER_PATH / "adapter_config.json").write_text(json.dumps({
    "peft_type": "LORA", "r": LORA_RANK, "lora_alpha": LORA_ALPHA,
    "base_model_name_or_path": "google/gemma-3-4b-it",
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}, indent=2))
log("LoRA adapter (%d tensors) written to %s — Lab 3 merges it at serve time", len(adapter), ADAPTER_PATH)
EOF

gcloud builds submit --tag=$IMAGE_URI_4B tunix-build-4b/
```

Takes about 2 minutes.

## Step 2 — Fine-tune (LoRA)

```bash
cat > training-4b.yaml <<'EOF'
apiVersion: batch/v1
kind: Job
metadata:
  name: finetune-4b-${TEAM}
  labels:
    kueue.x-k8s.io/queue-name: student-queue
spec:
  backoffLimit: 2
  template:
    metadata:
      annotations:
        gke-gcsfuse/volumes: "true"
    spec:
      serviceAccountName: jax-sa
      restartPolicy: Never
      nodeSelector:
        cloud.google.com/gke-tpu-accelerator: tpu-v5-lite-podslice
        cloud.google.com/gke-tpu-topology: 2x4
        topology.kubernetes.io/zone: us-west4-a
      containers:
        - name: training
          image: ${IMAGE_URI_4B}
          command: ["python3", "tunix_sft_main.py"]
          env:
            - { name: TEAM, value: "${TEAM}" }
            - name: WANDB_API_KEY
              valueFrom:
                secretKeyRef: { name: wandb-secret, key: api_key, optional: true }
          ports:
            - containerPort: 8431
          resources:
            limits: { google.com/tpu: "8", memory: 300Gi }
          volumeMounts:
            - { name: gcs, mountPath: /gcs }
            - { name: dshm, mountPath: /dev/shm }
      volumes:
        - name: gcs
          csi:
            driver: gcsfuse.csi.storage.gke.io
            volumeAttributes:
              bucketName: ${WORKSHOP_BUCKET}
              mountOptions: "implicit-dirs"
        - name: dshm
          emptyDir: { medium: Memory, sizeLimit: 16Gi }
EOF

envsubst < training-4b.yaml | kubectl apply -f -

kubectl get pods -l job-name=finetune-4b-${TEAM} -w
```

Ctrl-C once the pod shows `2/2 Running`. Watch progress with:
```bash
kubectl logs finetune-4b-${TEAM}-<pod-suffix> -c training --tail=20 | grep -v "Missing keys"
```
(The `grep -v` filters out a long, harmless line about unused vision-tower weights that otherwise floods the output.)

Expect this stage to take about **30–35 minutes** end to end (checkpoint load, 100 training steps, adapter export). When the job's `STATUS` shows `Complete`, confirm the adapter was written:
```bash
kubectl get job finetune-4b-${TEAM}
gsutil ls -la gs://me344-tpu-labs-west4/teams/${TEAM}/hermes-lora-adapter-4b/
```
You should see `adapter_model.safetensors` and `adapter_config.json`.

## Step 3 — Merge the LoRA adapter into the base model

```bash
cat > tunix_merge_main_4b.py <<'EOF'
import json, os, shutil
import numpy as np
import jax, jax.numpy as jnp
import qwix
from flax import nnx
from orbax import checkpoint as ocp
from safetensors.flax import load_file, save_file
import safetensors.numpy as _safe_np_mod
from transformers import AutoConfig

from tunix.models.gemma3 import model as gemma3_lib
from tunix.models.gemma3 import params as gemma3_params
from tunix.models.gemma3 import params_safetensors as params_safetensors_lib
from tunix.models import safetensors_saver
from tunix.rl import reshard

_orig_np_save_file = _safe_np_mod.save_file
def _np_safe_save_file(tensor_dict, filename, metadata=None):
    tensor_dict = {k: np.asarray(v) for k, v in tensor_dict.items()}
    return _orig_np_save_file(tensor_dict, filename, metadata=metadata)
_safe_np_mod.save_file = _np_safe_save_file

TEAM       = os.environ["TEAM"]
MODEL_PATH = "/gcs/models/gemma-3-4b-it"
CKPT_PATH  = f"/gcs/teams/{TEAM}/lora-ckpt-4b"
OUT_PATH   = f"/gcs/teams/{TEAM}/my-checkpoint-4b"
LORA_RANK, LORA_ALPHA = 32, 64.0

mesh = jax.make_mesh((jax.device_count(), 1), ("fsdp", "tp"),
                     axis_types=(jax.sharding.AxisType.Auto,) * 2)
config = gemma3_lib.ModelConfig.gemma3_4b_it()
hf_config = AutoConfig.from_pretrained(MODEL_PATH)
hf_vocab_size = getattr(getattr(hf_config, "text_config", hf_config), "vocab_size")
if hf_vocab_size != config.num_embed:
    print(f"Overriding Tunix num_embed: {config.num_embed} -> checkpoint vocab_size {hf_vocab_size}", flush=True)
    import dataclasses
    config = dataclasses.replace(config, num_embed=hf_vocab_size)
with jax.set_mesh(mesh):
    base = params_safetensors_lib.create_model_from_safe_tensors(MODEL_PATH, config, mesh)
prov = qwix.LoraProvider(
    module_path=".*q_einsum|.*kv_einsum|.*attn_vec_einsum|.*gate_proj|.*down_proj|.*up_proj",
    rank=LORA_RANK, alpha=LORA_ALPHA)
B, T = jax.device_count(), 8
trace = dict(last_tokens=jnp.ones((B, T), jnp.int32),
             positions=jnp.tile(jnp.arange(T)[None, :], (B, 1)), cache=None,
             attention_mask=jnp.tril(jnp.ones((B, T, T), jnp.bool_)))
with jax.set_mesh(mesh):
    lora_model = qwix.apply_lora_to_model(base, prov, rngs=nnx.Rngs(0), **trace)
del base
lora_model = reshard.reshard_model_to_mesh(lora_model, mesh)

abstract = nnx.state(lora_model, nnx.LoRAParam)
mngr = ocp.CheckpointManager(CKPT_PATH, item_names=("model_params",))
restored = mngr.restore(mngr.latest_step(),
    args=ocp.args.Composite(model_params=ocp.args.PyTreeRestore(item=abstract)))
nnx.update(lora_model, restored.model_params)
nnx.update(lora_model, jax.device_get(nnx.state(lora_model, nnx.LoRAParam)))

cpu = jax.devices("cpu")[0]
work = "/ramdisk/_merge_base"; os.makedirs(work, exist_ok=True)
with jax.default_device(cpu):
    for fn in os.listdir(MODEL_PATH):
        src = f"{MODEL_PATH}/{fn}"
        if os.path.isfile(src) and not fn.endswith(".safetensors") and not fn.endswith(".index.json"):
            shutil.copy(src, f"{work}/{fn}")
    state = {}
    idx = json.load(open(f"{MODEL_PATH}/model.safetensors.index.json"))
    for shard in sorted(set(idx["weight_map"].values())):
        state.update(load_file(f"{MODEL_PATH}/{shard}"))
    save_file(state, f"{work}/model.safetensors"); del state
    safetensors_saver.save_lora_merged_model_as_safetensors(
        local_model_path=work, output_dir=OUT_PATH, lora_model=lora_model,
        rank=LORA_RANK, alpha=LORA_ALPHA,
        state_key_transform_fn=lambda k: 'language_model.' + gemma3_params._gemma3_state_key_to_safetensors_key(k),
        custom_layer_extractor_fn=gemma3_params._extract_gemma3_lora_layers,
        transpose_rules=gemma3_params._GEMMA3_HUGGINGFACE_TRANSPOSE_RULES)
shutil.rmtree(work, ignore_errors=True)
print("Merge complete — merged checkpoint at", OUT_PATH, flush=True)
EOF

cat > merge-4b.yaml <<'EOF'
apiVersion: batch/v1
kind: Job
metadata:
  name: merge-4b-${TEAM}
  labels:
    kueue.x-k8s.io/queue-name: student-queue
spec:
  backoffLimit: 2
  template:
    metadata:
      annotations: { gke-gcsfuse/volumes: "true" }
    spec:
      serviceAccountName: jax-sa
      restartPolicy: Never
      nodeSelector:
        cloud.google.com/gke-tpu-accelerator: tpu-v5-lite-podslice
        cloud.google.com/gke-tpu-topology: 2x4
      containers:
        - name: merge
          image: ${IMAGE_URI_4B}
          command: ["bash", "-c"]
          args:
            - |
              uv pip install --system --quiet \
                'google-tunix @ git+https://github.com/google/tunix@c00f424b47964758361869233e72ca3b1a73119f' \
                'kagglesdk<0.1.24'
              python3 /scripts/tunix_merge_main.py
          env:
            - { name: TEAM, value: "${TEAM}" }
          resources:
            limits: { google.com/tpu: "8", memory: 100Gi }
          volumeMounts:
            - { name: gcs, mountPath: /gcs }
            - { name: scripts, mountPath: /scripts }
            - { name: ramdisk, mountPath: /ramdisk }
            - { name: dshm, mountPath: /dev/shm }
      volumes:
        - name: gcs
          csi:
            driver: gcsfuse.csi.storage.gke.io
            volumeAttributes: { bucketName: "${WORKSHOP_BUCKET}", mountOptions: "implicit-dirs", fileCacheCapacity: 60Gi }
        - { name: scripts, configMap: { name: merge-script-4b-${TEAM} } }
        - { name: ramdisk, emptyDir: { medium: Memory, sizeLimit: 20Gi } }
        - { name: dshm, emptyDir: { medium: Memory, sizeLimit: 4Gi } }
EOF

kubectl create configmap merge-script-4b-${TEAM} --from-file=tunix_merge_main.py=tunix_merge_main_4b.py

envsubst < merge-4b.yaml | kubectl apply -f -

kubectl get pods -l job-name=merge-4b-${TEAM} -w
```

Ctrl-C once the pod shows `2/2 Running`. This stage is quick — about **3 minutes**. Check progress and completion the same way as Step 2:
```bash
kubectl logs merge-4b-${TEAM}-<pod-suffix> -c merge --tail=20 | grep -v "Missing keys"
kubectl get job merge-4b-${TEAM}
```

Confirm the merged checkpoint landed in the bucket:
```bash
gsutil ls -la gs://me344-tpu-labs-west4/teams/${TEAM}/my-checkpoint-4b/
```
You should see a complete standalone HF checkpoint: `model.safetensors` (~8.6 GB) plus tokenizer/config files (~8 GB total, 14 objects).

## Step 4 — Serve your merged model

```bash
cat > serving-4b.yaml <<'EOF'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-4b-${TEAM}
spec:
  replicas: 1
  strategy:
    type: Recreate
  selector:
    matchLabels: { app: vllm-4b-${TEAM} }
  template:
    metadata:
      labels:
        app: vllm-4b-${TEAM}
        kueue.x-k8s.io/queue-name: student-queue
      annotations: { gke-gcsfuse/volumes: "true" }
    spec:
      serviceAccountName: jax-sa
      nodeSelector:
        cloud.google.com/gke-tpu-accelerator: tpu-v5-lite-podslice
        cloud.google.com/gke-tpu-topology: 2x4
      containers:
        - name: vllm
          image: vllm/vllm-tpu:latest
          command: ["vllm", "serve", "/mnt/models/teams/${TEAM}/my-checkpoint-4b"]
          args:
            - --host=0.0.0.0
            - --port=8000
            - --tensor-parallel-size=8
            - --gpu-memory-utilization=0.5
            - --max-model-len=2048
            - --max-num-seqs=128
            - --max-num-batched-tokens=4096
            - --disable_chunked_mm_input
          env:
            - { name: VLLM_XLA_CACHE_PATH, value: /mnt/models/teams/${TEAM}/xla-cache-4b }
            - { name: VLLM_TPU_MOST_MODEL_LEN, value: "2048" }
            - { name: VLLM_TPU_BUCKET_PADDING_GAP, value: "128" }
            - { name: HF_HUB_ENABLE_HF_TRANSFER, value: "1" }
            - { name: HF_HOME, value: /dev/shm/hf }
          ports:
            - { containerPort: 8000, name: api }
          resources:
            requests: { google.com/tpu: "8", memory: 100Gi }
            limits:   { google.com/tpu: "8", memory: 300Gi }
          startupProbe:
            httpGet: { path: /health, port: api }
            periodSeconds: 10
            failureThreshold: 180
          readinessProbe:
            httpGet: { path: /v1/models, port: api }
            periodSeconds: 10
          volumeMounts:
            - { name: gcs-mount, mountPath: /mnt/models }
            - { name: dshm, mountPath: /dev/shm }
      volumes:
        - name: gcs-mount
          csi:
            driver: gcsfuse.csi.storage.gke.io
            volumeAttributes: { bucketName: "${WORKSHOP_BUCKET}", mountOptions: "implicit-dirs", fileCacheCapacity: 100Gi }
        - { name: dshm, emptyDir: { medium: Memory, sizeLimit: 60Gi } }
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-svc-4b-${TEAM}
spec:
  type: LoadBalancer
  selector: { app: vllm-4b-${TEAM} }
  ports:
    - { port: 80, targetPort: 8000, protocol: TCP, name: http }
EOF

envsubst < serving-4b.yaml | kubectl apply -f -

kubectl get pods -l app=vllm-4b-${TEAM} -w
```

Ctrl-C once the pod shows `2/2 Running`. **The first boot takes about 20 minutes** — vLLM has to compile JAX/XLA graphs for every padded sequence-length bucket it might see (16 tokens up through 4096, twice — once for text-only input, once including multimodal embeddings). This is a one-time cost per fresh deployment; subsequent restarts of the same pod reuse the compiled cache and come up much faster.

Watch progress with:
```bash
kubectl logs vllm-4b-${TEAM}-<pod-suffix> -c vllm --tail=20
```
You'll see a long series of `Precompile ... finished in N.NN [secs]` lines — this is expected and not an error. It's done when you see the server start up and `/health` / `/v1/models` return `200 OK`.

## Step 5 — Test your deployment

```bash
kubectl port-forward pod/vllm-4b-${TEAM}-<pod-suffix> 8000:8000 &
sleep 3

curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"/mnt/models/teams/${TEAM}/my-checkpoint-4b\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Say hello in one short sentence.\"}],
    \"max_tokens\": 50
  }"
```

A successful response is a JSON object with `"object":"chat.completion"` and a non-empty `message.content`. Don't be surprised if the reply reads a little oddly (e.g. asking a clarifying question rather than just saying hello) — this fine-tune only ran 100 steps on function-calling data, so it isn't tuned for open-ended chat quality. That's expected; a valid, well-formed completion is what confirms the pipeline is working end to end.

When you're done testing, stop the port-forward:
```bash
kill %1
```

---

## Expected timing

| Stage | Wall time |
|---|---|
| Image build | ~2 min |
| Fine-tune | ~30–35 min |
| Merge | ~3 min |
| Serve (first boot) | ~20 min |
| **Total** | **well under an hour** |

## Troubleshooting

- **`kubectl logs` output is dominated by one huge repeated line** — pipe through `grep -v "Missing keys"` (training/merge) as shown above; it's a harmless message about unused vision-tower weights.
- **A pod is `CrashLoopBackOff` or shows `Error`** — check why before retrying:
  ```bash
  kubectl describe pod <pod-name> | grep -A10 "Last State"
  kubectl logs <pod-name> -c <container-name> --previous --tail=40
  ```
  `Exit Code: 137` means the process was killed (often out-of-memory) — the previous logs will usually show what it was doing right before the kill.
- **A resource looks like it belongs to someone else, or you get a "not found" mounting error** — double-check `echo $TEAM` matches what you intended, and that you re-exported all four variables from Step 0 after any new login.
- **`kubectl apply` fails with an empty-name or missing-image error** — one of your environment variables isn't actually exported in the current shell. Re-run the `export` block from Step 0.
