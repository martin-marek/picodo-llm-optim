

# step 1: install the gcloud CLI
- [https://cloud.google.com/sdk/docs/install](https://cloud.google.com/sdk/docs/install)


# step 2: set up port forwarding
- forward tpu port 8888 to local port 8889:
```bash
gcloud compute tpus tpu-vm ssh martin@tpu-v4-64 --zone=us-central2-b --worker=6 -- -L 8889:localhost:8888
```

# (step 3): pull changes from github
```bash
cd ~
rm -rf ~/picodo-llm-optim
git clone --depth=1 https://github.com/martin-marek/picodo-llm-optim.git
```


# (step 4): start jupyter lab server
- (only if not already running!)
```bash
tmux new-session -d "jupyter lab --NotebookApp.token=''"
```


# step 5: open jupyter lab in local browser 
- [http://localhost:8889/lab](http://localhost:8889/lab)
