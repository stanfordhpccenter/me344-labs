Log onto your assigned cluster (Located on ME344 Canvas >> Pages >> Course Cluster Assignment

1. Install gcloud CLI
```
sudo tee /etc/yum.repos.d/google-cloud-sdk.repo <<'EOF'
[google-cloud-cli]
name=Google Cloud CLI
baseurl=https://packages.cloud.google.com/yum/repos/cloud-sdk-el9-x86_64
enabled=1
gpgcheck=1
repo_gpgcheck=0
gpgkey=https://packages.cloud.google.com/yum/doc/rpm-package-key.gpg
EOF
sudo dnf install -y google-cloud-cli
```

2. Install kubectl
```
sudo tee /etc/yum.repos.d/kubernetes.repo <<'EOF'
[kubernetes]
name=Kubernetes
baseurl=https://pkgs.k8s.io/core:/stable:/v1.30/rpm/
enabled=1
gpgcheck=1
gpgkey=https://pkgs.k8s.io/core:/stable:/v1.30/rpm/repodata/repomd.xml.key
EOF
sudo dnf install -y kubectl
```

3. Install the GKE auth plugin
```
sudo dnf install -y google-cloud-cli-gke-gcloud-auth-plugin
echo 'export USE_GKE_GCLOUD_AUTH_PLUGIN=True' >> ~/.bashrc
source ~/.bashrc
```

4. Authenticate with your SUNetID Google Account
```
gcloud auth login --no-launch-browser
```
Open the printed URL in a web browser on your computer, log in with your @stanford.edu account, paste the code back into the shell prompt.

5. Set the project and get cluster credentials
```
gcloud config set project soe-hpccenter
gcloud container clusters get-credentials class-tpu-cluster --region=us-west4 --project=soe-hpccenter
```

6. Create and submit the smoke test
Save as test-job.yaml — use a unique job name (e.g. your SUNetID or Group Name) in "tpu-smoke-test-[SUNetID or GroupName]" so it doesn't collide with classmates:
```
apiVersion: batch/v1
kind: Job
metadata:
  name: tpu-smoke-test-[SUNetID or GroupName]
  labels:
    kueue.x-k8s.io/queue-name: student-queue
spec:
  template:
    spec:
      containers:
      - name: test
        image: python:3.11
        command: ["python3", "-c", "print('TPU pod scheduled successfully')"]
        resources:
          requests:
            google.com/tpu: "4"
          limits:
            google.com/tpu: "4"
      nodeSelector:
        cloud.google.com/gke-tpu-accelerator: tpu-v5-lite-podslice
        cloud.google.com/gke-tpu-topology: 2x2
      restartPolicy: Never
```

7. Apply, watch it get admitted and scheduled
```
kubectl apply -f test-job.yaml
kubectl get workloads
kubectl get pods --watch
```
If it's stuck Pending with quota already ADMITTED: True, that just means GKE is booting a fresh TPU node — this commonly takes ~5 minutes, so let it sit before assuming something's wrong.

8. Get the logs — stream live to avoid missing them
TPU nodes scale down automatically once idle, which garbage-collects the pod (and its logs) shortly after the job finishes. Don't wait to fetch logs after the fact — stream them live right after applying:
```
kubectl logs -f job/tpu-smoke-test-[NAME]
```
You should see TPU pod scheduled successfully.

9. If you missed the window — recover logs from Cloud Logging
Even after the pod is gone, its stdout/stderr is retained in Cloud Logging (Autopilot ships it there automatically). Replace tpu-smoke-test-[NAME] with your job name:
```
gcloud logging read 'resource.type="k8s_container" AND resource.labels.namespace_name="default" AND labels."k8s-pod/job-name"="tpu-smoke-test-[NAME]"' --project=soe-hpccenter --limit=20 --format="value(textPayload)"
```
That completes the loop — submit, monitor, and retrieve output, with a fallback for logs even after the pod itself no longer exists.

10. Clean up the smoke test

The Job has no ttlSecondsAfterFinished set, so it won't remove itself even after it completes — and this is a shared default namespace, so a stale Job clutters things for every other student and instructor running kubectl get jobs. Delete it once you've confirmed the smoke test worked:

```bash
kubectl delete job tpu-smoke-test-[SUNetID or GroupName]
```
Confirm nothing's left:

```bash
kubectl get pods
kubectl get workloads
```
Both should come back with no trace of your job — kubectl get pods empty (or showing only other students'/teams' work), and your job no longer listed under kubectl get workloads. The TPU node itself scales back down automatically once nothing is using it (same as noted in step 8), but that's node-level cleanup — it doesn't remove the Job object itself, which is why the explicit kubectl delete above is still needed.

Your cluster setup is now fully clean and ready for the labs.
