# Render Your Galaxy: PyTorch N-Body Simulation on GH200

## Introduction
**Duration: 15:00**

A short, self-contained demo: simulate 16,384 stars gravitationally orbiting a central mass on the GH200's GPU, render the simulation as an animated GIF, and pull it back to your own machine to watch spiral arms form in real orbital mechanics you computed yourself.

This runs on `stanford-pilot` — control plane `hpcc-gke.stanford.edu` + GPU worker `hpcc-pilot` — from the same class Linux server (`hpcc-cluster-N`) you already use for the TPU labs, via a separate kubectl context.

**Heads up**: there is exactly one physical GPU on this cluster. If anyone else's workload is still running, this Job will queue in `Pending` until the GPU frees up. Either wait, or clean up the other workload first.

### What you'll do

- Connect to the cluster (or confirm you're already connected)
- Submit the simulation as a Job — it computes on the GPU, renders to a GIF, and smuggles the file out through the logs (base64-encoded), since your RBAC persona can't `exec` into pods or reach any shared storage
- Decode the GIF and pull it to your own machine
- Optionally, render a second galaxy with a different seed
- Watch your galaxy — and clean up everything you created before you're done

### What you'll need

- Your class Linux server (`hpcc-cluster-N`), with a working `stanford-pilot` kubectl context already in `~/.kube/config`
- Your assigned namespace (e.g. `ns-student08`)

No Hugging Face token, no dataset, no pip installs from your shell — the container image used here (`nvcr.io/nvidia/pytorch:25.01-py3`) already has everything needed baked in.

---

## Before you begin — switch context
**Duration: 03:00**

Skip this if you've already switched context and have `$NS` set from earlier in your session. The following uses <b>student01</b> as an example, use your student number (same as your cluster number):

```bash
kubectl config get-contexts
```


```bash
kubectl config use-context student[N]-context
```

`e.g. student01-context`

```bash
kubectl auth whoami
```

`confirm: system:serviceaccount:ns-student01:student01`

```bash
kubectl get nodes -o wide
```


```bash
export NS=ns-student[N]
```

`your assigned namespace, e.g. ns-student01`

```bash
kubectl get pods
```

`should return No resources found in ns-student[N] namespace.`

---

## Run the simulation
**Duration: 08:00**

This is a known-good image, already verified working on this exact GH200 (arm64 + Hopper). **Do not swap the image tag** — the runbook this is adapted from originally used `nvcr.io/nvidia/k8s/cuda-sample:nbody`, which fails with `exec format error` on this hardware because that tag has no arm64 build. `nvcr.io/nvidia/pytorch:25.01-py3` is the confirmed-working substitute.

```bash
cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: galaxy-gif
  namespace: ${NS}
spec:
  ttlSecondsAfterFinished: 300
  activeDeadlineSeconds: 600
  backoffLimit: 0
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: galaxy
        image: nvcr.io/nvidia/pytorch:25.01-py3
        command: ["python", "-u", "-c"]
        args:
          - |
            import torch, time, math, io, base64
            import numpy as np
            from PIL import Image
            dev = 'cuda' if torch.cuda.is_available() else 'cpu'
            N, FRAMES, SUB = 16384, 80, 3
            GRID, SCALE = 200, 2
            G, M, dt, soft = 1.0, 60.0, 0.004, 0.05
            torch.manual_seed(0)
            r  = 0.15 + 1.7 * torch.rand(N, device=dev).sqrt()
            th = 2 * math.pi * torch.rand(N, device=dev)
            pos = torch.stack([r * th.cos(), r * th.sin()], 1)
            pos = pos + 0.04 * torch.randn(N, 2, device=dev)
            v   = (G * M / r).sqrt()
            vel = torch.stack([-v * th.sin(), v * th.cos()], 1)
            vel = vel * (0.92 + 0.16 * torch.rand(N, 1, device=dev))
            m   = torch.full((N, 1), 0.02, device=dev)
            def acc(p):
                d  = p.unsqueeze(0) - p.unsqueeze(1)
                r2 = (d * d).sum(-1) + soft * soft
                a  = (d * (m.unsqueeze(0) / (r2 * r2.sqrt()).unsqueeze(-1))).sum(1) * G
                rc = (p * p).sum(1, keepdim=True) + soft * soft
                return a - G * M * p / rc.pow(1.5)
            stops = [(0,(0,0,8)),(70,(20,24,110)),(130,(120,40,160)),(190,(255,140,40)),(255,(255,255,235))]
            cmap = np.zeros((256,3), dtype=np.uint8)
            for (a0,c0),(a1,c1) in zip(stops[:-1], stops[1:]):
                for i in range(a0, a1+1):
                    t = (i-a0)/max(1,(a1-a0))
                    cmap[i] = [round(c0[k]+(c1[k]-c0[k])*t) for k in range(3)]
            frames, glow = [], torch.zeros(GRID, GRID, device=dev)
            a = acc(pos); t0 = time.time()
            for f in range(FRAMES):
                for s in range(SUB):
                    vel = vel + 0.5*dt*a; pos = pos + dt*vel
                    a = acc(pos);         vel = vel + 0.5*dt*a
                ix = ((pos[:,0]+2.3)/4.6*GRID).long()
                iy = ((pos[:,1]+2.3)/4.6*GRID).long()
                ok = (ix>=0)&(ix<GRID)&(iy>=0)&(iy<GRID)
                counts = torch.bincount((iy[ok]*GRID+ix[ok]), minlength=GRID*GRID).reshape(GRID,GRID).float()
                glow = glow*0.55 + counts
                li = torch.log1p(glow)
                li = (li / li.max().clamp(min=1e-6)).pow(0.65) * 255
                li[li < 26] = 0
                img = li.clamp(0, 255).byte().cpu().numpy()
                big = Image.fromarray(img, 'L').resize((GRID*SCALE, GRID*SCALE), Image.BILINEAR)
                pim = Image.fromarray(np.asarray(big), 'P')
                pim.putpalette(cmap.flatten())
                frames.append(pim)
                if (f+1) % 20 == 0:
                    print("rendered frame %d/%d" % (f+1, FRAMES), flush=True)
            dtt = time.time()-t0
            name = torch.cuda.get_device_name(0) if dev=='cuda' else 'cpu'
            print("simulated on: %s" % name, flush=True)
            print("%d bodies x %d steps = %.1f billion interactions/sec" % (N, FRAMES*SUB, N*N*FRAMES*SUB/dtt/1e9), flush=True)
            buf = io.BytesIO()
            frames[0].save(buf, format='GIF', save_all=True, append_images=frames[1:], duration=60, loop=0)
            data = buf.getvalue()
            if len(data) > 5500000:
                frames = frames[::2]
                buf = io.BytesIO()
                frames[0].save(buf, format='GIF', save_all=True, append_images=frames[1:], duration=120, loop=0)
                data = buf.getvalue()
            print("GIF size: %.2f MB" % (len(data)/1e6), flush=True)
            b64 = base64.b64encode(data).decode()
            print("-----BEGIN GIF-----")
            for i in range(0, len(b64), 76):
                print(b64[i:i+76])
            print("-----END GIF-----", flush=True)
        resources:
          limits:
            nvidia.com/gpu: 1
EOF
kubectl get pods -n $NS -l job-name=galaxy-gif -w
```

Expect `Pending → ContainerCreating → Running → Completed` within a couple of minutes (longer the very first time, while the ~10 GB image pulls, if it wasn't pre-pulled).

`CTRL+C keys end the wait and returns you to the shell`

---

## Decode and view your galaxy
**Duration: 04:00**

Pull the GIF out of the logs — the only output channel available to your persona, since it can't `exec` into pods or reach shared storage:

```bash
kubectl logs job/galaxy-gif -n $NS | sed -n '/-----BEGIN GIF-----/,/-----END GIF-----/p' | sed '1d;$d' | base64 -d > galaxy.gif
ls -lh galaxy.gif
```

Expect a file roughly 2–4 MB. The logs also show which GPU it ran on and a throughput score (billions of interactions/sec) — worth a look:

```bash
kubectl logs job/galaxy-gif -n $NS | grep -E "simulated on|interactions/sec"
```

Do the decode step within the Job's 5-minute TTL window, or just re-run the Job — it starts in seconds once the image is cached.

**Pull it to your own laptop** to actually view it, e.g.:

```bash
scp admin@hpcc-cluster-N.stanford.edu:galaxy.gif .
```

**Viewing on a Mac**: use Quick Look (select the file in Finder, press spacebar) or drag it into a browser — macOS Preview does **not** animate GIFs, it shows them as a click-through list of frames, which looks broken but isn't. Windows and Linux default viewers animate GIFs normally.

Expect: a spinning galaxy — thousands of stars orbiting a glowing core, spiral arms winding up as inner stars lap outer ones. Real orbital mechanics, visualized.

---

## Optional — render a second galaxy with a different seed
**Duration: 08:00**

Change `torch.manual_seed(0)` to any other integer and resubmit under a new Job name — every seed produces a visibly different galaxy. Below uses seed `1`:

```bash
cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: galaxy-gif-1
  namespace: ${NS}
spec:
  ttlSecondsAfterFinished: 300
  activeDeadlineSeconds: 600
  backoffLimit: 0
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: galaxy
        image: nvcr.io/nvidia/pytorch:25.01-py3
        command: ["python", "-u", "-c"]
        args:
          - |
            import torch, time, math, io, base64
            import numpy as np
            from PIL import Image
            dev = 'cuda' if torch.cuda.is_available() else 'cpu'
            N, FRAMES, SUB = 16384, 80, 3
            GRID, SCALE = 200, 2
            G, M, dt, soft = 1.0, 60.0, 0.004, 0.05
            torch.manual_seed(1)
            r  = 0.15 + 1.7 * torch.rand(N, device=dev).sqrt()
            th = 2 * math.pi * torch.rand(N, device=dev)
            pos = torch.stack([r * th.cos(), r * th.sin()], 1)
            pos = pos + 0.04 * torch.randn(N, 2, device=dev)
            v   = (G * M / r).sqrt()
            vel = torch.stack([-v * th.sin(), v * th.cos()], 1)
            vel = vel * (0.92 + 0.16 * torch.rand(N, 1, device=dev))
            m   = torch.full((N, 1), 0.02, device=dev)
            def acc(p):
                d  = p.unsqueeze(0) - p.unsqueeze(1)
                r2 = (d * d).sum(-1) + soft * soft
                a  = (d * (m.unsqueeze(0) / (r2 * r2.sqrt()).unsqueeze(-1))).sum(1) * G
                rc = (p * p).sum(1, keepdim=True) + soft * soft
                return a - G * M * p / rc.pow(1.5)
            stops = [(0,(0,0,8)),(70,(20,24,110)),(130,(120,40,160)),(190,(255,140,40)),(255,(255,255,235))]
            cmap = np.zeros((256,3), dtype=np.uint8)
            for (a0,c0),(a1,c1) in zip(stops[:-1], stops[1:]):
                for i in range(a0, a1+1):
                    t = (i-a0)/max(1,(a1-a0))
                    cmap[i] = [round(c0[k]+(c1[k]-c0[k])*t) for k in range(3)]
            frames, glow = [], torch.zeros(GRID, GRID, device=dev)
            a = acc(pos); t0 = time.time()
            for f in range(FRAMES):
                for s in range(SUB):
                    vel = vel + 0.5*dt*a; pos = pos + dt*vel
                    a = acc(pos);         vel = vel + 0.5*dt*a
                ix = ((pos[:,0]+2.3)/4.6*GRID).long()
                iy = ((pos[:,1]+2.3)/4.6*GRID).long()
                ok = (ix>=0)&(ix<GRID)&(iy>=0)&(iy<GRID)
                counts = torch.bincount((iy[ok]*GRID+ix[ok]), minlength=GRID*GRID).reshape(GRID,GRID).float()
                glow = glow*0.55 + counts
                li = torch.log1p(glow)
                li = (li / li.max().clamp(min=1e-6)).pow(0.65) * 255
                li[li < 26] = 0
                img = li.clamp(0, 255).byte().cpu().numpy()
                big = Image.fromarray(img, 'L').resize((GRID*SCALE, GRID*SCALE), Image.BILINEAR)
                pim = Image.fromarray(np.asarray(big), 'P')
                pim.putpalette(cmap.flatten())
                frames.append(pim)
                if (f+1) % 20 == 0:
                    print("rendered frame %d/%d" % (f+1, FRAMES), flush=True)
            dtt = time.time()-t0
            name = torch.cuda.get_device_name(0) if dev=='cuda' else 'cpu'
            print("simulated on: %s" % name, flush=True)
            print("%d bodies x %d steps = %.1f billion interactions/sec" % (N, FRAMES*SUB, N*N*FRAMES*SUB/dtt/1e9), flush=True)
            buf = io.BytesIO()
            frames[0].save(buf, format='GIF', save_all=True, append_images=frames[1:], duration=60, loop=0)
            data = buf.getvalue()
            if len(data) > 5500000:
                frames = frames[::2]
                buf = io.BytesIO()
                frames[0].save(buf, format='GIF', save_all=True, append_images=frames[1:], duration=120, loop=0)
                data = buf.getvalue()
            print("GIF size: %.2f MB" % (len(data)/1e6), flush=True)
            b64 = base64.b64encode(data).decode()
            print("-----BEGIN GIF-----")
            for i in range(0, len(b64), 76):
                print(b64[i:i+76])
            print("-----END GIF-----", flush=True)
        resources:
          limits:
            nvidia.com/gpu: 1
EOF
kubectl get pods -n $NS -l job-name=galaxy-gif-1 -w
```

Decode the same way, into a differently-named file:

```bash
kubectl logs job/galaxy-gif-1 -n $NS | sed -n '/-----BEGIN GIF-----/,/-----END GIF-----/p' | sed '1d;$d' | base64 -d > galaxy-1.gif
ls -lh galaxy-1.gif
scp admin@hpcc-cluster-N.stanford.edu:galaxy-1.gif .
```

**If you want to try more seeds**, give each attempt its own Job name (`galaxy-gif-2`, `galaxy-gif-3`, …) rather than reusing one — reusing a name without deleting the old Job first will fail (see Common Failures below).

---

## Common failures

- **`exec format error`**: wrong CPU architecture for the image — GH200's CPU is Grace (ARM64), not x86. Don't swap in an amd64-only tag (this is exactly what broke the original `cuda-sample:nbody` image).
- **Job sits `Pending`**: only one GPU on the cluster. Check the pod's events for the actual reason:
  ```bash
  kubectl describe pod -n $NS -l job-name=galaxy-gif | sed -n '/Events/,$p'
  ```
  `0/2 nodes are available: 1 Insufficient nvidia.com/gpu, ...` means someone else (yours or a classmate's workload) already holds the GPU — this is normal queueing, not a bug. Your persona can't list pods outside your own namespace, so you can't see who; just wait, or ask around.
- **`kubectl apply` fails with `... is forbidden: ... cannot patch resource "jobs"`**: this happens if a Job with the same name already exists (even a finished/terminated one that hasn't been garbage-collected yet) — `kubectl apply` tries to patch it, and this RBAC persona can only `create`/`delete` Jobs, not `patch`/`update`. Fix: always delete before re-applying a Job with the same name:
  ```bash
  kubectl delete job galaxy-gif -n $NS --ignore-not-found
  ```
- **First run is slow**: the ~10 GB image pull, if it wasn't pre-pulled by the instructor. Subsequent runs start in seconds once cached on the node.
- **`kubectl logs` output looks truncated or the decode fails**: the GIF-extraction `sed` pattern needs the full, unbroken `-----BEGIN GIF-----` … `-----END GIF-----` block — if the Job was still running when you grabbed logs, wait for `Completed` and try again.

---

## Clean up
**Duration: 02:00**

**Do this before you finish** — you're sharing one GPU with the whole class, and nothing here scales down or expires quickly enough to rely on. `ttlSecondsAfterFinished: 300` will eventually clean up finished Jobs on its own, but don't wait on it — remove everything you created now, so the GPU is free the moment you're done:

```bash
kubectl delete jobs,cronjobs,deployments --all -n $NS
```

That catches every Job you ran in this lab (`galaxy-gif`, `galaxy-gif-1`, and any extra seeds you tried), plus anything else this RBAC persona is able to create, even if you experimented beyond what this lab covers.

Confirm nothing is left running:

```bash
kubectl get pods,jobs,cronjobs,deployments -n $NS
```

Expect `No resources found in ns-<you> namespace.` for all of them. If anything still shows up, delete it by name and re-check.

---

## What you can now explain

1. Why does this Job print a base64-encoded block to the logs instead of just writing `galaxy.gif` to a file somewhere you could copy it from directly?
2. What would happen if you ran this Job while another GPU workload was still deployed in your namespace?
3. What's the actual physics being simulated, and why do spiral arms emerge from what starts as a fairly random distribution of stars?
4. Why does re-running a Job with the same name fail differently than running one with a brand-new name?
