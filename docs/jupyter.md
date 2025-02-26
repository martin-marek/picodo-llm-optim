
# step 1: set up port forwarding
- forward tpu port 8888 to local port 8889
- worker is `0...7`
```bash
gcloud compute tpus tpu-vm ssh martin@tpu-v4-64 --zone=us-central2-b --worker=6 -- -L 8889:localhost:8888
```


# (step 2): start jupyter lab server
- (only if not already running!)
```bash
pkill python && tmux kill-server
tmux new-session -d "jupyter lab --NotebookApp.token=''"
```


# step 3: open jupyter lab in local browser 
- [http://localhost:8889/lab](http://localhost:8889/lab)
