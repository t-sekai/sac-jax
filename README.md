# SAC-JAX

Soft Actor-Critic implemented in JAX, adapted from [Maniskill](https://github.com/haosulab/ManiSkill) and [RFCL](https://github.com/StoneT2000/rfcl.git)'s SAC. Currently, it works for Maniskill tasks, and the gpu vectorization isn't really working. However, it solves PushCube-v1 in 10 minutes, which is really fast for SAC.

## Installation

To get started run `git clone https://github.com/StoneT2000/rfcl.git sac_jax --branch ms3-gpu` which contains the code for RLPD written in jax (a partial fork of the original RLPD and JaxRL repos that has been optimized to run faster and support vectorized environments).

We recommend using conda/mamba and you can install the dependencies as so:

```bash
conda create -n "sac_jax" "python==3.9"
conda activate sac_jax
pip install --upgrade "jax[cuda12_pip]==0.4.28" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -e sac_jax
```

Then you can install ManiSkill and its dependencies

```bash
pip install mani_skill torch==2.3.1
```
Note that since jax and torch are used, we recommend installing the specific versions detailed in the commands above as those are tested to work together.

## Train

To train with environment vectorization run

```bash
env_id=PushCube-v1
seed=42
XLA_PYTHON_CLIENT_PREALLOCATE=false python train_sac.py configs/base_sac_ms3.yml \
  logger.exp_name="sac-${env_id}-state-${seed}-walltime_efficient" \
  seed=${seed} train.steps=200_000 env.env_id=${env_id} env.num_envs=32 \
  logger.wandb=True logger.project_name="sac_jax" logger.wandb_cfg.group=${env_id}
```

<!-- This should solve the PickCube-v1 task in a few minutes, but won't get good sample efficiency.

For sample-efficient settings you can use the sample-efficient configurations stored in configs/base_rlpd_ms3_sample_efficient.yml (no env parallelization, more critics, higher update-to-data ratio). This will take less environment samples (around 50K to solve) but runs slower.

```bash
env_id=PickCube-v1
demos=1000 # number of demos to train on.
seed=42
XLA_PYTHON_CLIENT_PREALLOCATE=false python train_ms3.py configs/base_rlpd_ms3_sample_efficient.yml \
  logger.exp_name="rlpd-${env_id}-state-${demos}_rl_demos-${seed}-sample_efficient" logger.wandb=True \
  seed=${seed} train.num_demos=${demos} train.steps=100_000 \
  env.env_id=${env_id} \
  train.dataset_path="~/.maniskill/demos/${env_id}/rl/trajectory.state.pd_joint_delta_pos.h5"
``` -->

evaluation videos are saved to `exps/<exp_name>/videos`.

<!-- ## Generating Demonstrations / Evaluating policies

To generate 1000 demonstrations you can run

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python rlpd_jax/scripts/collect_demos.py exps/path/to/model.jx \
  num_envs=8 num_episodes=1000
```
This saves the demos which uses CPU vectorization to generate demonstrations in parallel. Note that while the demos are generated on the CPU, you can always convert them to demonstrations on the GPU via the [replay trajectory tool](https://maniskill.readthedocs.io/en/latest/user_guide/datasets/replay.html) as so

```bash
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path exps/<exp_name>/eval_videos/trajectory.h5 \
  -b gpu --use-first-env-state
```

The replay_trajectory tool can also be used to generate videos

See the rlpd_jax/scripts/collect_demos.py code for details on how to load the saved policies and modify it to your needs. -->