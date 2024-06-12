# Taken or adapted from Curt-Park/rainbow-is-all-you-need (MIT license)
# https://github.com/Curt-Park/rainbow-is-all-you-need
# https://colab.research.google.com/github/Curt-Park/rainbow-is-all-you-need/blob/master/08.rainbow.ipynb#scrollTo=xYbhMIvf2Xrd

# Further inspiration:
# https://github.com/Lizhi-sjtu/Rainbow-DQN-pytorch
# https://github.com/Kaixhin/Rainbow
# https://github.com/davide97l/Rainbow
# https://github.com/kochlisGit/Autonomous-Vehicles-Adaptive-Cruise-Control


from typing import Deque, Dict, List, Tuple

import random
import os
import io

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
# from IPython.display import clear_output
from torch.nn.utils import clip_grad_norm_

from model_pixel import Network
from data import ReplayBuffer, PrioritizedReplayBuffer

from torch.utils.tensorboard import SummaryWriter

from clearml import Task, Dataset, InputModel
from munch import DefaultMunch
import json
os.environ["MUJOCO_GL"] = "egl"

from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import PIL.Image

from env_carla import Environment
from env_carla import IM_HEIGHT, IM_WIDTH, N_ACTIONS

from moviepy import editor as mpy

VIDEO_RECORDING = True
REMOTE_EXECUTION = True
FIXED_DELTA_SECONDS = 0.1

PATH_SCENARIO = ""
PATH_SCENARIO_DIR = ""
PATH_MODEL = ""

Task.add_requirements(
    package_name="setuptools",
    package_version="59.5.0",
)
Task.add_requirements(
    package_name="torch",
    package_version="",
)
Task.add_requirements(
    package_name="torchvision",
    package_version="",
)
Task.add_requirements(
    package_name="numpy",
    package_version="",
)
Task.add_requirements(
    package_name="moviepy",
    package_version="1.0.3",
)

Task.add_requirements(package_name="gym",
                        package_version="",)
Task.add_requirements(package_name="scikit-image",
                        package_version="",)
Task.add_requirements(package_name="moviepy",
                        package_version="",)
Task.add_requirements(package_name="mkl",
                        package_version="",)

task = Task.init(
    project_name="bogdoll/rl_traffic_rule_Jing",
    task_name="Rainbow Baseline",
    reuse_last_task_id=True,
    tags="Rainbow_test_2",
)

logger = task.get_logger()

if REMOTE_EXECUTION:
    task.set_base_docker(
        docker_image="nvcr.io/nvidia/pytorch:22.12-py3",  # nvcr.io/nvidia/pytorch:22.12-py3 from https://catalog.ngc.nvidia.com/containers?filters=&orderBy=dateModifiedDESC&query=
        docker_arguments="-e NVIDIA_DRIVER_CAPABILITIES=all --network=host",
        docker_setup_bash_script=["apt-get install -y libgl1"],
    )

    task.execute_remotely("docker", clone=False, exit_process=True)

writer = SummaryWriter()

def load_settings(path):
    with open(path+'/scenario_set_2.json') as json_file:
        settings = json.load(json_file)
        # convert to a dictionary that supports attribute-style access, a la JavaScript
        settings = DefaultMunch.fromDict(settings)
    folder_name = path.split("/")[-1]
    if folder_name == "": folder_name = path.split("/")[-2]
    print(f"~~~~\n# Scenario set: {folder_name} \n# Contains {settings.size} scenarios among world: {settings.world} \n~~~~")
    return settings

class DQNAgent:
    """DQN Agent interacting with environment.

    Attribute:
        env (gym.Env): openAI Gym environment
        memory (PrioritizedReplayBuffer): replay memory to store transitions
        batch_size (int): batch size for sampling
        target_update (int): period for target model's hard update
        gamma (float): discount factor
        dqn (Network): model to train and select actions
        dqn_target (Network): target model to update
        optimizer (torch.optim): optimizer for training dqn
        transition (list): transition information including
                           state, action, reward, next_state, done
        v_min (float): min value of support
        v_max (float): max value of support
        atom_size (int): the unit number of support
        support (torch.Tensor): support for categorical dqn
        use_n_step (bool): whether to use n_step memory
        n_step (int): step number to calculate n-step td error
        memory_n (ReplayBuffer): n-step replay buffer
    """

    def __init__(
        self,
        env: Environment,
        memory_size: int,
        batch_size: int,
        target_update: int,
        gamma: float = 0.99,
        # PER parameters
        alpha: float = 0.2,
        beta: float = 0.6,
        prior_eps: float = 1e-6,
        # Categorical DQN parameters
        v_min: float = 0.0,
        v_max: float = 200.0,
        atom_size: int = 51,
        # N-step Learning
        n_step: int = 3,
    ):
        """Initialization.

        Args:
            env (gym.Env): openAI Gym environment
            memory_size (int): length of memory
            batch_size (int): batch size for sampling
            target_update (int): period for target model's hard update
            lr (float): learning rate
            gamma (float): discount factor
            alpha (float): determines how much prioritization is used
            beta (float): determines how much importance sampling is used
            prior_eps (float): guarantees every transition can be sampled
            v_min (float): min value of support
            v_max (float): max value of support
            atom_size (int): the unit number of support
            n_step (int): step number to calculate n-step td error
        """
        obs_dim = [3, IM_HEIGHT, IM_WIDTH]
        action_dim = N_ACTIONS

        self.env = env
        self.batch_size = batch_size
        self.target_update = target_update
        self.gamma = gamma
        # NoisyNet: All attributes related to epsilon are removed

        # device: cpu / gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        # PER
        # memory for 1-step Learning
        self.beta = beta
        self.prior_eps = prior_eps
        self.memory = PrioritizedReplayBuffer(obs_dim, memory_size, batch_size, alpha=alpha)

        # memory for N-step Learning
        self.use_n_step = True if n_step > 1 else False
        if self.use_n_step:
            self.n_step = n_step
            self.memory_n = ReplayBuffer(obs_dim, memory_size, batch_size, n_step=n_step, gamma=gamma)

        # Categorical DQN parameters
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.support = torch.linspace(self.v_min, self.v_max, self.atom_size).to(self.device)

        # networks: dqn, dqn_target
        self.dqn = Network(action_dim, self.atom_size, self.support).to(self.device)
        self.dqn_target = Network(action_dim, self.atom_size, self.support).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        # optimizer
        self.optimizer = optim.Adam(self.dqn.parameters())

        # transition to store in memory
        self.transition = list()

        # mode: train / test
        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        # NoisyNet: no epsilon greedy action selection
        debug = torch.FloatTensor(state).to(self.device)
        selected_action = self.dqn(debug).argmax()
        selected_action = selected_action.detach().cpu().numpy()

        if not self.is_test:
            self.transition = [state, selected_action]

        return selected_action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, done, info = self.env.step(action)

        if not self.is_test:
            self.transition += [reward, next_state, done]

            # N-step transition
            if self.use_n_step:
                one_step_transition = self.memory_n.store(*self.transition)
            # 1-step transition
            else:
                one_step_transition = self.transition

            # add a single step transition
            if one_step_transition:
                self.memory.store(*one_step_transition)

        return next_state, reward, done, info

    def update_model(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        # PER needs beta to calculate weights
        samples = self.memory.sample_batch(self.beta)
        weights = torch.FloatTensor(samples["weights"].reshape(-1, 1)).to(self.device)
        indices = samples["indices"]

        # 1-step Learning loss
        elementwise_loss = self._compute_dqn_loss(samples, self.gamma)

        # PER: importance sampling before average
        loss = torch.mean(elementwise_loss * weights)

        # N-step Learning loss
        # we are gonna combine 1-step loss and n-step loss so as to
        # prevent high-variance. The original rainbow employs n-step loss only.
        if self.use_n_step:
            gamma = self.gamma**self.n_step
            samples = self.memory_n.sample_batch_from_idxs(indices)
            elementwise_loss_n_loss = self._compute_dqn_loss(samples, gamma)
            elementwise_loss += elementwise_loss_n_loss

            # PER: importance sampling before average
            loss = torch.mean(elementwise_loss * weights)

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.dqn.parameters(), 10.0)
        self.optimizer.step()

        # PER: update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)

        # NoisyNet: reset noise
        self.dqn.reset_noise()
        self.dqn_target.reset_noise()

        return loss.item()

    def train(self, num_frames: int, plotting_interval: int = 500):
        """Train the agent."""
        self.is_test = False

        state, _ = self.env.reset()
        update_cnt = 0
        losses = []
        scores = []
        score = 0

        max_score = float("-inf")

        chw_list = []
        episode_length = 0

        for frame_idx in range(1, num_frames + 1):
            # Save episode video
            if VIDEO_RECORDING:
                chw = state.squeeze(0)  # Remove batch information from BCHW
                chw_list.append(chw)

            action = self.select_action(state)
            writer.add_scalar("Selected Action", action, frame_idx)
            next_state, reward, done, info = self.step(action)

            state = next_state
            score += reward

            # NoisyNet: removed decrease of epsilon

            # PER: increase beta
            fraction = min(frame_idx / num_frames, 1.0)
            self.beta = self.beta + fraction * (1.0 - self.beta)

            writer.add_scalar("Fraction", fraction, frame_idx)
            writer.add_scalar("Beta", self.beta, frame_idx)

            # if episode ends
            if done:
                arrived_s = info["arrived_s"]
                rule_score = info["rule_score"]
                finished_score = info["finished_score"]
                state, _ = self.env.reset()
                scores.append(score)
                writer.add_scalar("train_return", score, frame_idx)
                writer.add_scalar("train_length", episode_length, frame_idx)
                writer.add_scalar("train_rule_score", rule_score, frame_idx)
                writer.add_scalar("train_finished_score", finished_score, frame_idx)
                writer.add_scalar("train_arrived_s", arrived_s, frame_idx)
                print("[", frame_idx, "]", "This episode step:", episode_length, "return:", score, "arrived length:", arrived_s)
                episode_length = 0
                score = 0

            # if training is ready
            if len(self.memory) >= self.batch_size:
                loss = self.update_model()
                losses.append(loss)
                update_cnt += 1

                # if hard update is needed
                if update_cnt % self.target_update == 0:
                    self._target_hard_update()

                writer.add_scalar("Loss", loss, frame_idx)

            episode_length += 1

            # plotting
            if frame_idx % plotting_interval == 0 and VIDEO_RECORDING:
                tchw_list = torch.stack(chw_list)  # Adds "list" like entry --> TCHW
                writer.add_video(
                    tag="DQN Agent",
                    vid_tensor=tchw_list.unsqueeze(0),
                    global_step=frame_idx,
                    fps=int(1 / FIXED_DELTA_SECONDS),
                )  # Unsqueeze adds batch --> BTCHW
                chw_list = []

        self.env.close()

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray], gamma: float) -> torch.Tensor:
        """Return categorical dqn loss."""
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"]).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        # Categorical DQN algorithm
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with torch.no_grad():
            # Double DQN
            next_action = self.dqn(next_state).argmax(1)
            next_dist = self.dqn_target.dist(next_state)
            next_dist = next_dist[range(self.batch_size), next_action]

            t_z = reward + (1 - done) * gamma * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(0, (self.batch_size - 1) * self.atom_size, self.batch_size)
                .long()
                .unsqueeze(1)
                .expand(self.batch_size, self.atom_size)
                .to(self.device)
            )

            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
            proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))

        dist = self.dqn.dist(state)
        log_p = torch.log(dist[range(self.batch_size), action])
        elementwise_loss = -(proj_dist * log_p).sum(1)

        return elementwise_loss

    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.dqn_target.load_state_dict(self.dqn.state_dict())

    # def _plot(
    #     self,
    #     frame_idx: int,
    #     scores: List[float],
    #     losses: List[float],
    # ):
    #     """Plot the training progresses."""
    #     clear_output(True)
    #     plt.figure(figsize=(20, 5))
    #     plt.subplot(131)
    #     plt.title("frame %s. score: %s" % (frame_idx, np.mean(scores[-10:])))
    #     plt.plot(scores)
    #     plt.subplot(132)
    #     plt.title("loss")
    #     plt.plot(losses)
    #     plt.show()


# Use Moritz scenario settings
path = Dataset.get(dataset_id=PATH_SCENARIO).get_local_copy()
# path = PATH_SCENARIO_DIR
settings = settings = load_settings(path)

# Get base model
base_model = InputModel(model_id=PATH_MODEL).get_local_copy(raise_on_error=True)
print("base model: ", base_model)

# environment
env = Environment(settings.world, 
                  settings, 
                  host="tks-harper.fzi.de", 
                  port=2000,
                  size=(256, 256),
                  grayscale=True)  # This would be better as a command line argument
env.init_ego()

seed = 777
def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


np.random.seed(seed)
random.seed(seed)
seed_torch(seed)

# parameters
num_frames = 200_000
memory_size = 5_000
batch_size = 32
target_update = 100

# train
agent = DQNAgent(env, memory_size, batch_size, target_update)
agent.train(num_frames)

# Evaluation (is_test)

writer.flush()
