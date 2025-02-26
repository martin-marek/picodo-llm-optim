
# step 1: set up port forwarding
- forward tpu port 22 to local port 2222:
```bash
gcloud compute tpus tpu-vm ssh martin@tpu-v4-64 --zone=us-central2-b --worker=6 -- -N -L 2222:localhost:22
```


# step 2: add vscode ssh config
- VS Code -> Command Palette → "Remote-SSH: Open SSH Configuration File":
```
Host tpu-v4-64-w6
  HostName localhost
  User martin
  Port 2222
```


# step 3: connect to tpu from VS Code

