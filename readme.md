## Instructions for Setting Up the Environment


To create a new conda environment, open your terminal and run the following command:

```bash
conda create --name dpt python=3.9.15
```

Install PyTorch by following the [official instructions here](https://pytorch.org/get-started/locally/) appropriately for your system. The recommended versions for the related packages are as follows with CUDA 12.1:

```bash
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

The remaining requirements are fairly standard and are listed in the `requirements.txt`. These can be installed by running

```bash
pip install -r requirements.txt
```

Otherwise, you can either make singularity environment using the [following instruction](https://sites.google.com/nyu.edu/nyu-hpc/).

## Running Experiments
All the commands are for slurm script. If you want to run them in your local server, just change .sbatch to .sh.
### (Linear) Bandit Experiments
Bandit experiment on DPT:
```bash
train_offline.sh configs/bandit_offline.yaml
```

Linear Bandit experiment on DPT:
```bash
train_offline.sh configs/linear_bandit_offline.yaml
```

Bandit experiment on online DPT:
```bash
train_online_rollout.sh configs/bandit_offline.yaml
```

Linear Bandit experiment on online DPT:
```bash
train_online_rollout.sh configs/linear_bandit_offline.yaml
```

### Darkroom Experiments
Darkroom experiment on DPT:
```bash
train_offline.sh configs/darkroom_offline.yaml
```

Darkroom experiment on online DPT:
```bash
train_online_rollout.sh configs/darkroom_online.yaml
