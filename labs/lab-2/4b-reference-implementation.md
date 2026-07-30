# Lab Guide: Fine-Tuning and Serving Gemma 3 4B on TPU

This guide walks through the full pipeline for your team: fine-tune Gemma 3 4B with LoRA on a hermes function-calling dataset, merge the adapter into the base weights, and serve the result with vLLM on a TPU v5e slice — then confirm it's working with a live inference request.

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
It should print `gke_soe-hpccenter_us-west4_class-tpu-cluster-west4`. If it doesn't, add and switch to it:
```bash
gcloud container clusters get-credentials class-tpu-cluster-west4 \
  --region us-west4 --project soe-hpccenter
```

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

```bash
gcloud builds submit --tag=$IMAGE_URI_4B tunix-build-4b/
```

This packages `tunix-build-4b/tunix_sft_main.py` into a container image tagged with your team name. Takes about 2 minutes.

## Step 2 — Fine-tune (LoRA)

```bash
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

Create the ConfigMap holding the merge script (the key must be named exactly `tunix_merge_main.py` — that's what the job's container command expects, regardless of the local filename) and submit the merge job:

```bash
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
