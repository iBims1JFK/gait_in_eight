## Installation
1. Install RL-X

Default installation for a Linux system with a NVIDIA GPU.
For other configurations, see the RL-X [documentation](https://nico-bohlinger.github.io/RL-X/#detailed-installation-guide).
```bash 
conda create -n gait_in_eight python=3.11.4
conda activate gait_in_eight
git clone git@github.com:iBims1JFK/RL-X.git
cd RL-X
pip install -e .[all] --config-settings editable_mode=compat
pip uninstall $(pip freeze | grep -i '\-cu12' | cut -d '=' -f 1) -y
pip install "torch>=2.2.1" --index-url https://download.pytorch.org/whl/cu118 --upgrade
pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# downgrade to jax==0.4.28 jaxlib==0.4.28 
# i.e.: pip install jaxlib==0.4.28+cuda12.cudnn89 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# for more information see the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html#installing-older-jaxlib-wheels)
# for nvidia gpu ensure that jax-cuda12-pjrt==0.4.28 jax-cuda12-plugin==0.4.28
```

2. Install the project
```bash
git clone git@github.com:iBims1JFK/gait_in_eight.git
cd gait_in_eight
pip install -e .
```


## Experiments
1. Setup conda environment and installation as described in the installation section
2. Run the following commands to start an experiment
```bash
cd gait_in_eight/experiments
# use on of the following commands
bash jtp_forward.sh
bash jtp_max.sh
bash jtp_omnidirectional.sh
bash cpg_forward.sh
bash cpg_max.sh
bash cpg_omnidirectional.sh
```


## Testing a trained model
1. Create test.sh file in the experiments folder (all .sh files besides slurm_experiment.sh are ignored by git)
```bash
cd gait_in_eight/experiments
touch test.sh
```
2. Add the following content to the test.sh file
```bash
python experiment.py \
    --algorithm.name=crossq.default \
    --environment.name="honey_badger_jtp" \
    --environment.mode=test \
    --environment.render=False \
    --environment.add_goal_arrow=True \
    --runner.mode=test \
    --runner.load_model=model_best_jax
```
#### Controlling the robot
Either create commands.txt file
```bash
cd gait_in_eight/experiments
touch commands.txt
```
And add the following content to the commands.txt file. Where the values are target x, y and yaw velocities
```bash
1.0
0.0
0.0
```

## On robot experiments
1. Setup conda environment and installation as described in the installation section
2. Ensure that the robot is connected to the network
3. Run the ROS frontend provided in ros-frontend
4. Source the ROS workspace
5. Start experiment
```bash
cd gait_in_eight/experiments
bash jtp_on_robot_experiment.sh
# or
bash cpg_on_robot_experiment.sh
```

# License
This project is licensed under the GPLv3 License - see the [LICENSE](LICENSE) file for details.
