Change TEAM=team-[TEAM-NAME] to name of your team (e.g. TEAM=team-smjones as my SUNetID is smjones)

Setup
```
gcloud config set project soe-hpccenter
gcloud container clusters get-credentials class-tpu-cluster --region=us-central1 --project=soe-hpccenter

export REGION=us-central1
export WORKSHOP_BUCKET=me344-tpu-labs
export TEAM=team-[TEAM-NAME]
export IMAGE_URI=${REGION}-docker.pkg.dev/soe-hpccenter/tpu-images/gemma3-finetune-${TEAM}:latest

kubectl get serviceaccount jax-sa
gcloud storage ls gs://${WORKSHOP_BUCKET}/models/gemma-3-27b-it/ | head
```
Lab 2 - fine-tuning

1. Write the build directory
```
mkdir -p tunix-build
cat > tunix-build/Dockerfile <<'EOF'
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
```

Create the training script:
```
cat > tunix-build/tunix_sft_main.py <<'PYEOF'
"""LoRA-fine-tune Gemma 3 27B on hermes function-calling, on a single-host TPU slice.

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
from transformers import AutoTokenizer

from tunix.models.gemma3 import model as gemma3_lib
from tunix.models.gemma3 import params as gemma3_params
from tunix.models.gemma3 import params_safetensors as params_safetensors_lib
from tunix.models.safetensors_saver import join_path
from tunix.rl import reshard
from tunix.sft import peft_trainer, utils as sft_utils
from tunix.sft.metrics_logger import MetricsLoggerOptions

TEAM         = os.environ["TEAM"]
GCS          = Path("/gcs")
MODEL_PATH   = GCS / "models" / "gemma-3-27b-it"
DATA_PATH    = GCS / "data" / "hermes-fc"
CKPT_PATH    = GCS / "teams" / TEAM / "lora-ckpt"
ADAPTER_PATH = GCS / "teams" / TEAM / "hermes-lora-adapter"

BATCH_SIZE = 8
MAX_LEN    = 512
MAX_STEPS  = 100
LORA_RANK  = 32
LORA_ALPHA = 64.0
LR         = 1e-3

WANDB_PROJECT = "getting-started-with-tpu-on-gcp"
WANDB_RUN     = "gemma3-27b-hermes-lora"

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logging.getLogger("orbax").setLevel(logging.WARNING)
log = logging.getLogger("tunix-sft").info

log("TPU devices: %d | %s", jax.device_count(), [d.coords for d in jax.devices()])
MESH = [(jax.device_count(), 1), ("fsdp", "tp")]
mesh = jax.make_mesh(*MESH, axis_types=(jax.sharding.AxisType.Auto,) * len(MESH[0]))
log("Mesh: %s — loading %s (FSDP-%d, ~2-3 min from FUSE)…",
    dict(zip(MESH[1], MESH[0])), MODEL_PATH, MESH[0][0])

config = gemma3_lib.ModelConfig.gemma3_27b_it()
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
        backend_kwargs={"wandb": {"tags": ["tunix", "gemma3-27b", "lora", f"v5e-{jax.device_count()}"]}},
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
    "base_model_name_or_path": "google/gemma-3-27b-it",
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}, indent=2))
log("LoRA adapter (%d tensors) written to %s — Lab 3 merges it at serve time", len(adapter), ADAPTER_PATH)
PYEOF
```

Verify it wrote correctly:
```
cat tunix-build/tunix_sft_main.py | head -20
wc -l tunix-build/tunix_sft_main.py
```
The last line of output should be <b>172 tunix-build/tunix_sft_main.py</b>

2. Build and push the image (~5 mins):
```
gcloud builds submit tunix-build/ --tag "$IMAGE_URI"
gcloud artifacts docker images list ${REGION}-docker.pkg.dev/soe-hpccenter/tpu-images | grep "gemma3-finetune-${TEAM}"
```

3. Save and submit the training Job:
```
envsubst < training.yaml | kubectl apply -f -
kubectl get workloads
kubectl logs -f job/finetune-${TEAM} -c training | grep -v vision_tower
```
Watch specifically for LoRA adapters injected and sharded — N.NM trainable params early on — that confirms the last_tokens fix worked. If it errors there instead, that's the trace-input mismatch resurfacing and needs a look before continuing.

4. Confirm outputs, clean up:
```
gcloud storage ls gs://${WORKSHOP_BUCKET}/teams/${TEAM}/hermes-lora-adapter/
kubectl delete job finetune-${TEAM} --ignore-not-found
```
