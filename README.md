
# step 1: install the gcloud CLI
- [https://cloud.google.com/sdk/docs/install](https://cloud.google.com/sdk/docs/install)


# step 2: connect to tpu
- follow specific instructions for [tmux](docs/tmux.md), [jupyter lab](docs/jupyter.md), or [vs code](docs/vscode.md)


# (step 3): pull changes from github
```bash
cd ~
rm -rf ~/picodo-llm-optim
git clone --depth=1 https://github.com/martin-marek/picodo-llm-optim.git
```


# training run (4 chips)
- create a single training run that uses all 4 chips (takes ~1h:40min)
```bash
# use all chips (0,1,2,3) on this worker
export TPU_CHIPS_PER_PROCESS_BOUNDS='2,2,1'
export TPU_PROCESS_BOUNDS='1,1,1'
export TPU_VISIBLE_DEVICES='0,1,2,3'

# run training job
cd ~/picodo-llm-optim
python main.py -cn tpuvm \
    wandb_project='picodo-test' \
    run_name='bs:32 test' \
    opt.train_microbatch_size=32 \
    opt.peak_learning_rate=0.002
```


# training run (1 chip)
- use only a single chip for the training run
- useful for BS=1 training
```bash
# use only a single chip
export TPU_CHIPS_PER_PROCESS_BOUNDS='1,1,1'
export TPU_PROCESS_BOUNDS='1,1,1'
export TPU_VISIBLE_DEVICES=0 # <-- select chip {0,1,2,3}

# run training job
cd ~/picodo-llm-optim
python main.py -cn tpuvm \
    wandb_project='picodo-test' \
    run_name='bs:1 test' \
    opt.train_microbatch_size=64 \
    opt.single_step_training=True \
    opt.eval_batch_size=16 \
    opt.peak_learning_rate=0.0002 \
    opt.adam_t2_t1_ratio=24 \
    opt.weight_decay=0.005
```
