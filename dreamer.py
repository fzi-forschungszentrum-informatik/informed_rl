import argparse
import collections
import functools
import os
import io
import pathlib
import sys
import warnings

from clearml import Task, Dataset, InputModel
from munch import DefaultMunch
import json
os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import ruamel.yaml as yaml

sys.path.append(str(pathlib.Path(__file__).parent/"dreamerv3-torch-submod"))

import exploration as expl
import models
import tools
import envs.wrappers as wrappers

import torch
from torch import nn
from torch import distributions as torchd
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt
import PIL.Image

to_np = lambda x: x.detach().cpu().numpy()

# PATH_SCENARIO = "b87ac090f583466999b3efab4a4fe799" # Anomaly scenarios 1
PATH_SCENARIO = "c07cb0b5bfc44c27954fe2c0b0f1ca98" # Anomaly scenarios 2
# PATH_SCENARIO = "019ab9ad127546709f1bccbb172f08fa"
PATH_SCENARIO_DIR = "/fzi/ids/jk639/no_backup/data" # change to the direction where you save scenario set
# PATH_MODEL = "2438e8e1fccd43c7a8a76f97ed698e6e" # 1 stop
PATH_MODEL = "7a432d74fa41410aa57ea2579b4b789e" # Base dreamer

class Dreamer(nn.Module):
    def __init__(self, obs_space, act_space, config, logger, dataset):
        super(Dreamer, self).__init__()
        self._config = config
        self._logger = logger
        self._should_log = tools.Every(config.log_every)
        batch_steps = config.batch_size * config.batch_length
        self._should_train = tools.Every(batch_steps / config.train_ratio)
        self._should_pretrain = tools.Once()
        self._should_reset = tools.Every(config.reset_every)
        self._should_expl = tools.Until(int(config.expl_until / config.action_repeat))
        self._metrics = {}
        self._step = count_steps(config.traindir)
        self._update_count = 0
        # Schedules.
        config.actor_entropy = lambda x=config.actor_entropy: tools.schedule(
            x, self._step
        )
        config.actor_state_entropy = (
            lambda x=config.actor_state_entropy: tools.schedule(x, self._step)
        )
        config.imag_gradient_mix = lambda x=config.imag_gradient_mix: tools.schedule(
            x, self._step
        )
        self._dataset = dataset
        self._wm = models.WorldModel(obs_space, act_space, self._step, config)
        self._task_behavior = models.ImagBehavior(
            config, self._wm, config.behavior_stop_grad
        )
        # if config.compile:
        #     self._wm = torch.compile(self._wm)
        #     self._task_behavior = torch.compile(self._task_behavior)
        reward = lambda f, s, a: self._wm.heads["reward"](f).mean
        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior,
            random=lambda: expl.Random(config),
            plan2explore=lambda: expl.Plan2Explore(config, self._wm, reward),
        )[config.expl_behavior]().to(self._config.device)

    def __call__(self, obs, reset, state=None, reward=None, training=True):
        step = self._step
        if self._should_reset(step):
            state = None
        if state is not None and reset.any():
            mask = 1 - reset
            for key in state[0].keys():
                for i in range(state[0][key].shape[0]):
                    state[0][key][i] *= mask[i]
            for i in range(len(state[1])):
                state[1][i] *= mask[i]
        if training:
            steps = (
                self._config.pretrain
                if self._should_pretrain()
                else self._should_train(step)
            )
            for _ in range(steps):
                self._train(next(self._dataset))
                self._update_count += 1
                self._metrics["update_count"] = self._update_count
            if self._should_log(step):
                # for name, val`ues in self._metrics.items():
                #     self._logger.scalar(name, float(np.mean(values)))
                #     self._metrics[name] = []
                # if self._config.video_pred_log:
                #     openl = self._wm.video_pred(next(self._dataset))
                #     self._logger.video("train_openl", to_np(openl))
                self._logger.write(fps=True)

        policy_output, state = self._policy(obs, state, training)

        if training:
            self._step += len(reset)
            self._logger.step = self._config.action_repeat * self._step
        return policy_output, state

    def _policy(self, obs, state, training):
        if state is None:
            batch_size = len(obs["image"])
            latent = self._wm.dynamics.initial(len(obs["image"]))
            action = torch.zeros((batch_size, self._config.num_actions)).to(
                self._config.device
            )
        else:
            latent, action = state
        obs = self._wm.preprocess(obs)
        embed = self._wm.encoder(obs)
        latent, _ = self._wm.dynamics.obs_step(
            latent, action, embed, obs["is_first"], self._config.collect_dyn_sample
        )
        if self._config.eval_state_mean:
            latent["stoch"] = latent["mean"]
        feat = self._wm.dynamics.get_feat(latent)
        if not training:
            actor = self._task_behavior.actor(feat)
            action = actor.mode()
        elif self._should_expl(self._step):
            actor = self._expl_behavior.actor(feat)
            action = actor.sample()
        else:
            actor = self._task_behavior.actor(feat)
            action = actor.sample()
        logprob = actor.log_prob(action)
        latent = {k: v.detach() for k, v in latent.items()}
        action = action.detach()
        if self._config.actor_dist == "onehot_gumble":
            action = torch.one_hot(
                torch.argmax(action, dim=-1), self._config.num_actions
            )
        action = self._exploration(action, training)
        policy_output = {"action": action, "logprob": logprob}
        state = (latent, action)
        return policy_output, state

    def _exploration(self, action, training):
        amount = self._config.expl_amount if training else self._config.eval_noise
        if amount == 0:
            return action
        if "onehot" in self._config.actor_dist:
            probs = amount / self._config.num_actions + (1 - amount) * action
            return tools.OneHotDist(probs=probs).sample()
        else:
            return torch.clip(torchd.normal.Normal(action, amount).sample(), -1, 1)
        raise NotImplementedError(self._config.action_noise)

    def _train(self, data):
        metrics = {}
        post, context, mets = self._wm._train(data)
        metrics.update(mets)
        start = post
        # start['deter'] (16, 64, 512)
        reward = lambda f, s, a: self._wm.heads["reward"](
            self._wm.dynamics.get_feat(s)
        ).mode()
        metrics.update(self._task_behavior._train(start, reward)[-1])
        if self._config.expl_behavior != "greedy":
            mets = self._expl_behavior.train(start, context, data)[-1]
            metrics.update({"expl_" + key: value for key, value in mets.items()})
        for name, value in metrics.items():
            if not name in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)


def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))


def make_dataset(episodes, config):
    generator = tools.sample_episodes(episodes, config.batch_length)
    dataset = tools.from_generator(generator, config.batch_size)
    return dataset


def make_env(config, logger, mode, train_eps, eval_eps, settings):
    suite, task = config.task.split("_", 1)
    if suite == "carla":
        import envs.carla_wrapper as carla
        env = carla.Carla(
            settings.world,
            settings, 
            host='tks-hazard.fzi.de',
            port=2000,
            action_repeat=config.action_repeat,
            size=config.size,
            grayscale=config.grayscale,
            done=False
        )
        # env = wrappers.NormalizeActions(env)
        env = wrappers.OneHotAction(env)
    else:
        raise NotImplementedError(suite)
    env = wrappers.TimeLimit(env, config.time_limit)
    env = wrappers.SelectAction(env, key="action")
    if (mode == "train") or (mode == "eval"):
        callbacks = [
            functools.partial(
                ProcessEpisodeWrap.process_episode,
                config,
                logger,
                mode,
                train_eps,
                eval_eps,
            )
        ]
        env = wrappers.CollectDataset(env, mode, train_eps, callbacks=callbacks)
    env = wrappers.RewardObs(env)
    return env


class ProcessEpisodeWrap:
    eval_scores = []
    eval_lengths = []
    eval_arrive_inds = []
    eval_rule_scores = []
    last_step_at_eval = -1
    eval_episode_num = 1
    eval_done = False

    @classmethod
    def process_episode(cls, config, logger, mode, train_eps, eval_eps, episode, camera_ls, rule_paras, anomaly):
        directory = dict(train=config.traindir, eval=config.evaldir)[mode]
        cache = dict(train=train_eps, eval=eval_eps)[mode]
        # this saved episodes is given as train_eps or eval_eps from next call
        filename = tools.save_episodes(directory, [episode])[0]
        length = len(episode["reward"]) - 1
        score = float(episode["reward"].astype(np.float64).sum())
        arrived_ind = episode["arrived_s"][-1]
        rule_score = episode["rule_score"][-1]
        finished_score = episode["finished_score"][-1]
        # video = episode["image"]
        video = camera_ls
        cache[str(filename)] = episode
        if mode == "train":
            total = 0
            for key, ep in reversed(sorted(cache.items(), key=lambda x: x[0])):
                if not config.dataset_size or total <= config.dataset_size - length:
                    total += len(ep["reward"]) - 1
                else:
                    del cache[key]
            # logger.scalar("dataset_size", total, len(cache))
            # logger.image(f"{mode}_rule_graph", rule_graph)
            logger.scalar(f"{mode}_return", score)
            logger.scalar(f"{mode}_length", length)
            logger.scalar(f"{mode}_arrived_s", arrived_ind)
            logger.scalar(f"{mode}_rule_score", rule_score)
            logger.scalar(f"{mode}_finished_score", finished_score)
            # logger.video(f"{mode}_policy", video[None])
            logger.scalar(
                f"{mode}_episodes", len(cache)
            )
            # use dataset_size as log step for a condition of envs > 1
            log_step = total * config.action_repeat
            log_episode = len(cache)
        elif mode == "eval":
            # keep only last item for saving memory
            while len(cache) > 1:
                # FIFO
                cache.popitem()
            # start counting scores for evaluation
            if cls.last_step_at_eval != logger.step:
                cls.eval_scores = []
                cls.eval_lengths = []
                cls.eval_arrive_inds = []
                cls.eval_rule_scores = []
                cls.eval_finished_score = []
                cls.eval_done = False
                cls.last_step_at_eval = logger.step
            cls.eval_scores.append(score)
            cls.eval_lengths.append(length)
            cls.eval_arrive_inds.append(arrived_ind)
            cls.eval_rule_scores.append(rule_score)
            cls.eval_finished_score.append(finished_score)
            # ignore if number of eval episodes exceeds eval_episode_num
            if len(cls.eval_scores) < config.eval_episode_num or cls.eval_done:
                return
            score_2 = {'score': score,
                       'avg_score': sum(cls.eval_scores) / len(cls.eval_scores)}
            length_2 = {'length': length,
                        'avg_length': sum(cls.eval_lengths) / len(cls.eval_lengths)}
            arrived_ind_2 = {'arrived_s': arrived_ind,
                             'avg_arrived_ind:': sum(cls.eval_arrive_inds) / len(cls.eval_arrive_inds)}
            rule_score_2 = {'rule_score': rule_score,
                            'avg_rule_score': sum(cls.eval_rule_scores) / len(cls.eval_rule_scores)}
            
            score = sum(cls.eval_scores) / len(cls.eval_scores)
            length = sum(cls.eval_lengths) / len(cls.eval_lengths)
            arrived_ind = sum(cls.eval_arrive_inds) / len(cls.eval_arrive_inds)
            rule_score = sum(cls.eval_rule_scores) / len(cls.eval_rule_scores)
            finished_score = sum(cls.eval_finished_score) / len(cls.eval_finished_score)
            episode_num = len(cls.eval_scores) * cls.eval_episode_num

            instance = cls()
            rule_graph = instance.rule_graph(rule_paras, anomaly, arrived_ind)

            log_step = logger.step
            logger.video(f"{mode}_policy", video[None])
            logger.image(f"{mode}_rule_graph", rule_graph)
            logger.scalar(f"{mode}_return", score)
            logger.scalar(f"{mode}_length", length)
            logger.scalar(f"{mode}_arrived_s", arrived_ind)
            logger.scalar(f"{mode}_rule_score", rule_score)
            logger.scalar(
                f"{mode}_episodes", episode_num
            )
            logger.scalar(f"{mode}_finished_score", finished_score)
            cls.eval_episode_num += 1
            cls.eval_done = True
        print(f"{mode.title()} episode has {length} steps and return {score:.1f}.")
        
        logger.write(step=log_step)
    
    def rule_graph(self, rule_paras, anomaly, arrived_ind):
        r_ls = [pos[0] for pos in rule_paras]
        s_ls = [pos[1] for pos in rule_paras]
        d_ls = [pos[2] for pos in rule_paras]
        length = int(round(s_ls[-1]))
        max_s = int(round(anomaly[2]))
        t1 = np.arange(0, max_s)
        t2 = np.arange(0, length)
        l1 = np.full(max_s, -1.75)
        l2 = np.full(max_s, 1.75)
        l3 = np.full(max_s, 5.25)
        lane_keep = np.zeros(length)
        no_out = np.zeros(length)
        no_collision = np.zeros(length)
        r_ls = [pos[0] for pos in rule_paras]
        s_ls = [pos[1] for pos in rule_paras]
        d_ls = [pos[2] for pos in rule_paras]
        fig, (ax2, ax3, ax4, ax1) = plt.subplots(4, 1, sharex=True, dpi=300)
        ax1.plot(s_ls, d_ls, 'b-')
        ax1.plot(t1, l1, 'g')
        ax1.plot(t1, l2, 'g--')
        ax1.plot(t1, l3, 'g')
        ax1.plot(anomaly[0], anomaly[1], 'r*')
        ax1.plot(anomaly[0]-10, anomaly[1], 'r^')
        ax1.plot(anomaly[0]+10, anomaly[1], 'r^')

        for pos in rule_paras:
            index = int(round(pos[1]))
            if index < length - 3:
                lane_keep[index:index+2] = 1
                no_out[index:index+2] = 1
                no_collision[index:index+2] = 1
                if pos[2] > 1.75:
                    lane_keep[index:index+2] = 1 - (np.fabs(pos[2]) - 1.75) / 3.5
                if pos[2] < -1.75:
                    no_out[index:index+2] = 1 - (np.fabs(pos[2]) - 1.75) / 3.5
                if pos[2] > 5.25:
                    no_out[index:index+2] = 1 - (np.fabs(pos[2]) - 5.25) / 3.5
        ax2.plot(t2, lane_keep, 'c', label='Lane keep')
        ax2.legend(loc="upper right")
        ax3.plot(t2, no_out, 'g', label='No out road')
        ax3.legend(loc="upper right")
        ax4.plot(t2, no_collision, 'b', label='No collision')
        ax4.legend(loc="upper right")
        
        ax2.set_title("Rules Graph")
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        image = PIL.Image.open(buf)
        image = ToTensor()(image).unsqueeze(0)
        image = image[0]
        return image

def load_settings(path):
    with open(path+'/scenario_set_2.json') as json_file:
        settings = json.load(json_file)
        # convert to a dictionary that supports attribute-style access, a la JavaScript
        settings = DefaultMunch.fromDict(settings)
    folder_name = path.split("/")[-1]
    if folder_name == "": folder_name = path.split("/")[-2]
    print(f"~~~~\n# Scenario set: {folder_name} \n# Contains {settings.size} scenarios among world: {settings.world} \n~~~~")
    return settings

def main(config):
    Task.add_requirements(package_name="gym",
                          package_version="",)
    Task.add_requirements(package_name="scikit-image",
                          package_version="",)
    Task.add_requirements(package_name="moviepy",
                          package_version="",)
    Task.add_requirements(package_name="mkl",
                          package_version="",)


    task = Task.init(project_name="bogdoll/rl_traffic_rule_Jing",
                     task_name="google_dreamer_v3_carla",
                     reuse_last_task_id=False,
                     tags="Dreamer_test_5",
                     output_uri="s3://tks-zx.fzi.de:9000/clearml")

    task.set_base_docker(
                "nvcr.io/nvidia/pytorch:22.12-py3",
                docker_setup_bash_script="apt-get update && apt-get install -y python3-opencv",
                docker_arguments="-e NVIDIA_DRIVER_CAPABILITIES=all --network=host"  # --ipc=host",
                )

    # task.execute_remotely('docker', clone=False, exit_process=True)

    # Use Moritz scenario settings
    # path = Dataset.get(dataset_id=PATH_SCENARIO).get_local_copy()
    path = PATH_SCENARIO_DIR
    settings = settings = load_settings(path)

    # Get base model
    base_model = InputModel(model_id=PATH_MODEL).get_local_copy(raise_on_error=True)
    print("base model: ", base_model)

    logdir = pathlib.Path(config.logdir).expanduser()
    config.traindir = config.traindir or logdir / "train_eps"
    config.evaldir = config.evaldir or logdir / "eval_eps"
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat

    print("Logdir", logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)
    step = count_steps(config.traindir)
    logger = tools.Logger(logdir, config.action_repeat * step)

    print("Create envs.")
    if config.offline_traindir:
        directory = config.offline_traindir.format(**vars(config))
    else:
        directory = config.traindir
    train_eps = tools.load_episodes(directory, limit=config.dataset_size)
    if config.offline_evaldir:
        directory = config.offline_evaldir.format(**vars(config))
    else:
        directory = config.evaldir
    eval_eps = tools.load_episodes(directory, limit=1)
    make = lambda mode: make_env(config, logger, mode, train_eps, eval_eps, settings)
    train_envs = [make("train") for _ in range(config.envs)]
    eval_envs = [make("eval") for _ in range(config.envs)]
    acts = train_envs[0].action_space
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    if not config.offline_traindir:
        prefill = max(0, config.prefill - count_steps(config.traindir))
        print(f"Prefill dataset ({prefill} steps).")
        if hasattr(acts, "discrete"):
            random_actor = tools.OneHotDist(
                torch.zeros(config.num_actions).repeat(config.envs, 1)
            )
        else:
            random_actor = torchd.independent.Independent(
                torchd.uniform.Uniform(
                    torch.Tensor(acts.low).repeat(config.envs, 1),
                    torch.Tensor(acts.high).repeat(config.envs, 1),
                ),
                1,
            )

        def random_agent(o, d, s, r):
            action = random_actor.sample()
            logprob = random_actor.log_prob(action)
            return {"action": action, "logprob": logprob}, None

        tools.simulate(random_agent, train_envs, prefill)
        logger.step = config.action_repeat * count_steps(config.traindir)

    print("Simulate agent.")
    train_dataset = make_dataset(train_eps, config)
    eval_dataset = make_dataset(eval_eps, config)
    agent = Dreamer(
        train_envs[0].observation_space,
        train_envs[0].action_space,
        config,
        logger,
        train_dataset,
    ).to(config.device)
    agent.requires_grad_(requires_grad=False)
    if base_model:
        agent.load_state_dict(torch.load(base_model))
        agent._should_pretrain._once = False

    state = None
    while agent._step < config.steps:
        logger.write()
        print("Start evaluation.")
        eval_policy = functools.partial(agent, training=False)
        tools.simulate(eval_policy, eval_envs, episodes=config.eval_episode_num)
        if config.video_pred_log:
            video_pred = agent._wm.video_pred(next(eval_dataset))
            # logger.video("eval_openl", to_np(video_pred))
        print("Start training.")
        state = tools.simulate(agent, train_envs, config.eval_every, state=state)
        torch.save(agent.state_dict(), "latest_model.pt")
    for env in train_envs + eval_envs:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    args, remaining = parser.parse_known_args()
    configs = yaml.safe_load(
        (pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text()
    )

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    name_list = ["defaults", *args.configs] if args.configs else ["defaults"]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    main(parser.parse_args(remaining))
