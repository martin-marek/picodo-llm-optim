

# step 1: install the gcloud CLI
- [https://cloud.google.com/sdk/docs/install](https://cloud.google.com/sdk/docs/install)


# step 2: ssh to tpu worker
- worker is `0...7`
```bash
gcloud compute tpus tpu-vm ssh tpu-v4-64 --zone=us-central2-b --worker=7
```


# (step 3): pull changes from github
```bash
cd ~
rm -rf ~/picodo-llm-optim
git clone --depth=1 https://github.com/martin-marek/picodo-llm-optim.git
```


# (step 4): kill tmux python processes
```bash
pkill python && tmux kill-server
wait 60
```


# training run (single run with 4 chips)
- create a single training run that uses all 4 chips (takes ~1h:40min)
```bash
tmux new-session -d "
    # use all chips (0,1,2,3) on this worker
    export TPU_CHIPS_PER_PROCESS_BOUNDS='2,2,1'
    export TPU_PROCESS_BOUNDS='1,1,1'
    export TPU_VISIBLE_DEVICES='0,1,2,3'

    # run training job
    cd ~/picodo-llm-optim
    python main.py -cn tpuvm \
        wandb_project='picodo-aditya' \
        run_name='my first training run' \
        opt.train_batch_size=32 \
        opt.peak_learning_rate=0.002 \
        opt.weight_decay=0.05
"
```


# training run (4 runs, each 1 chip)
- create 4 training runs, each using 1 chip (takes ~9 hours)
- for loop over `i` craetes 4 tmux processes, each process uses chip `i`
```bash
w=0.01
lrs=(0.0001 0.0002 0.0003 0.0004)
for i in {0..3}; do
    lr="${lrs[$i]}"
    tmux new-session -d -s $i "
        # use only chip 'i' for this run
        export TPU_CHIPS_PER_PROCESS_BOUNDS="1,1,1"
        export TPU_PROCESS_BOUNDS="1,1,1"
        export TPU_VISIBLE_DEVICES=$i

        # run training job
        cd ~/picodo-llm-optim
        python main.py -cn tpuvm \
            wandb_project='picodo-aditya' \
            run_name='bs:1 lr:$lr w:$w' \
            opt.train_batch_size=64 \
            opt.single_step_training=True \
            opt.eval_batch_size=1 \
            opt.peak_learning_rate=$lr \
            opt.weight_decay=$w
    "
done
```
