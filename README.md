# Informed Reinforcement Learning for Situation-Aware Traffic Rule Exceptions

The project uses DreamerV3 to train self-driving cars for motion planning in anomaly traffic rule exception scenarios based on the Carla environment.

## Table of Contents

- [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [Results](#results)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

This repository has been tested in following environments:

- Ubuntu 20.04
- Python 3.8.10
- Carla 0.9.13

### Installation

Follow these steps to get a development environment running :

1. Clone the repository:

    ```
    git clone git@github.com:daniel-bogdoll/rl_rule_deviations.git
    ```

2. Navigate to the cloned directory and install the required dependencies:

    ```
    cd rl_rule_deviations/dreamerv3-torch
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
3. Run RL training with Carla:
    - change your carla host in `dreamer.py` function `make_env`, in `suite == "carla"` condition.
    - set `ClearML` in `dreamer.py` function `main(config)` if needed.
    - run main execution file.
    ```
    python3 dreamer.py --configs carla_vision
    ```
you could use python virtual environment: `python -m venv env`

## Usage

- You can adjust parameters in `config.yaml` to control the training process, especially the parameters in the `carla_vision` subgroup.
- This repository currently relies on ClearML. The base model and scenarios dataset are stored on ClearML, which can be accessed through the variables `PATH_SCENARIO` and `PATH_MODEL` in `dreamer.py`. A predefined CARLA scenario set is provided in utilities (scenario_set_2.json).
- To train with or without the rulebook, modify the variable `self.rule_book_on` in the `TrafficRules` class found in `envs/traffic_rulebook.py`.

## Code Structure

Here is the basic code structure of this repository.
```
rl_rule_deviation
│
├── dreamer3-torch/
│ ├── exploration.py
│ ├── models.py
│ ├── networks.py
│ └── tools.py
│
├── envs/
│ ├── carla_wrapper.py
│ └── wrapper.py 
│
├── utilities/
│ ├── controller.py 
│ ├── cubic_spline_planner.py 
│ ├── frenet_optimal_trajectory.py 
│ └── traffic_rulebook.py 
│
├── configs.yaml
├── dreamer.py
├── requirements.txt
└── README.md
```
Explanation:
- `dreamer3-torch/`: Directory containing the main codebase.
- `envs/`: Directory containing environment files and supporting code.
- `utilities/`: Directory containing all instrumental code, such as math, controllers, and trajectory generator. 
- `configs.yaml`: A YAML file containing various configurations for the experiments.
- `dreamer.py`: The main script for executing the program.
- `requirements.txt`: Contains a list of necessary packages, installable via `pip install -r requirements.txt`.


## Citation

If you found our work useful, please cite our paper:
```
@inproceedings{bogdoll2024informed,
      title={Informed Reinforcement Learning for Situation-Aware Traffic Rule Exceptions}, 
      author={Daniel Bogdoll and Jing Qin and Moritz Nekolla and Ahmed Abouelazm and Tim Joseph and J. Marius Zöllner},
      year={2024},
      booktitle={IEEE International Conference on Robotics and Automation (ICRA)}
}
```
