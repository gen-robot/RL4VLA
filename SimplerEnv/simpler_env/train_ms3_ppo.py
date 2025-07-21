import os
import pprint
import random
import gc
import signal
from collections import defaultdict
import time
from pathlib import Path
from typing import Annotated
import torch
import numpy as np
import tyro
import wandb
from dataclasses import dataclass
import yaml
from tqdm import tqdm
from mani_skill.utils import visualization
from mani_skill.utils.visualization.misc import images_to_video

from simpler_env.env.simpler_wrapper import SimlerWrapper
from simpler_env.utils.replay_buffer import SeparatedReplayBuffer

signal.signal(signal.SIGINT, signal.SIG_DFL)  # allow ctrl+c
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "PutCarrotOnPlateInScene-v1"
    """The environment ID of the task you want to simulate. Can be one of
    PutCarrotOnPlateInScene-v1, 
    PutSpoonOnTableClothInScene-v1, 
    StackGreenCubeOnYellowCubeBakedTexInScene-v1, 
    PutEggplantInBasketScene-v1"""

    """Number of environments to run. With more than 1 environment the environment will use the GPU backend 
    which runs faster enabling faster large-scale evaluations. Note that the overall behavior of the simulation
    will be slightly different between CPU and GPU backends."""

    seed: Annotated[int, tyro.conf.arg(aliases=["-s"])] = 0
    """Seed the model and environment. Default seed is 0"""

    name: str = "PPO-test"

    # env
    num_envs: int = 64
    episode_len: int = 80
    use_same_init: bool = False

    steps_max: int = 2000000
    steps_vh: int = 0  # episodes
    interval_eval: int = 10
    interval_save: int = 40

    # buffer
    buffer_inferbatch: int = 32
    buffer_minibatch: int = 8
    buffer_gamma: float = 0.99
    buffer_lambda: float = 0.95

    # vla
    vla_path: str = "Haozhan72/Openvla-oft-SFT-libero-spatial-traj1"
    vla_unnorm_key: str = "bridge_orig"
    vla_load_path: str = "../openvla_oft/checkpoints/warmup/Openvla-oft-SFT-libero-spatial-traj1+warmup+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--2000_chkpt"
    vla_lora_rank: int = 32

    vla_lr: float = 1e-4
    vla_vhlr: float = 3e-3
    vla_optim_beta1: float = 0.9
    vla_optim_beta2: float = 0.999
    vla_temperature: float = 1.0
    vla_temperature_eval: float = 0.6

    # ppo & grpo
    alg_name: str = "ppo"  # ppo, grpo
    alg_grpo_fix: bool = True
    alg_gradient_accum: int = 20
    alg_ppo_epoch: int = 1
    alg_entropy_coef: float = 0.0

    # other
    wandb: bool = True
    only_render: bool = False
    render_info: bool = False

ACTION_CHUNCK_SIZE = 8  # number of actions to execute in one step, for efficiency

class Runner:
    def __init__(self, all_args: Args):
        self.args = all_args

        # alg_name
        assert self.args.alg_name in ["ppo", "grpo"]

        # set seed
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        from inputimeout import inputimeout, TimeoutOccurred

        project_default = "RLVLA_oft_first_try"
        if all_args.wandb:
            try:
                project_input = inputimeout(prompt=f"Enter the project name for wandb (default: {project_default}): ", timeout=5)
                if project_input.strip():
                    project_default = project_input
            except TimeoutOccurred:
                print(f"\nNo input received. Using default project name: {project_default}")
        # set wandb
        wandb.init(
            config=all_args.__dict__,
            project=project_default,
            name=self.args.name,
            mode="online" if self.args.wandb else "offline",
        )
        self.save_dir = Path(wandb.run.dir)
        self.glob_dir = Path(wandb.run.dir) / ".." / "glob"
        self.glob_dir.mkdir(parents=True, exist_ok=True)

        yaml.dump(all_args.__dict__, open(self.glob_dir / "config.yaml", "w"))

        # policy
        # from simpler_env.policies.openvla.openvla_train import OpenVLAPolicy, OpenVLAPPO
        from simpler_env.policies.openvla_oft.openvla_oft_train import OpenVLAPolicy, OpenVLAPPO
        device_id = 0
        device_id_other = 1 if torch.cuda.device_count() > 1 else 0
        self.device = torch.device("cuda:" + str(device_id))
        self.policy = OpenVLAPolicy(all_args, device_id_other)

        self.alg = OpenVLAPPO(all_args, self.policy)

        # env
        # unnorm_state = self.policy.vla.get_action_stats(self.args.vla_unnorm_key)
        self.env = SimlerWrapper(self.args)

        # buffer
        self.buffer = SeparatedReplayBuffer(
            all_args,
            obs_dim=(480, 640, 3),
            act_chunck=8,
            act_dim=7,
        )
        minibatch_count = self.buffer.get_minibatch_count()
        print(f"Buffer minibatch count: {minibatch_count}")

    @torch.no_grad()
    def _get_action(self, obs, deterministic=False):
        total_batch = obs["image"].shape[0]

        values = []
        actions = []
        logprobs = []
        generated_idss = []

        for i in range(0, total_batch, self.args.buffer_inferbatch):
            obs_batch = {k: v[i:i + self.args.buffer_inferbatch] for k, v in obs.items()}
            value, action, logprob, generated_ids = self.policy.get_action(obs_batch, deterministic)
            values.append(value)
            actions.append(action)
            logprobs.append(logprob)
            generated_idss.append(generated_ids)

        values = torch.cat(values, dim=0).to(device=self.device)
        actions = torch.cat(actions, dim=0)
        logprobs = torch.cat(logprobs, dim=0).to(device=self.device)
        generated_idss = torch.cat(generated_idss, dim=0).to(device=self.device)

        return values, actions, logprobs, generated_idss

    def collect(self):
        self.policy.prep_rollout()

        obs_image = self.buffer.obs[self.buffer.step]
        obs_image = torch.tensor(obs_image).to(self.device)
        obs = dict(image=obs_image, task_description=self.buffer.instruction)
        value, action, logprob, generated_ids = self._get_action(obs)

        return value, action, logprob, generated_ids

    def insert(self, data):
        obs_img, actions, logprob, value_preds, rewards, done, generated_ids = data
        masks = 1.0 - done.to(torch.float32)

        obs_img = obs_img.cpu().numpy()
        actions = actions.to(torch.int32).cpu().numpy()
        logprob = logprob.to(torch.float32).cpu().numpy()
        value_preds = value_preds.to(torch.float32).cpu().numpy()
        rewards = rewards.to(torch.float32).cpu().numpy()
        generated_ids = generated_ids.to(torch.int32).cpu().numpy()
        masks = masks.cpu().numpy()

        self.buffer.insert(obs_img, actions, logprob, value_preds, rewards, masks, generated_ids)

    def compute_endup(self):
        self.policy.prep_rollout()

        obs_image = torch.tensor(self.buffer.obs[-1]).to(self.device)
        obs = dict(image=obs_image, task_description=self.buffer.instruction)
        with torch.no_grad():
            next_value, _, _, _ = self._get_action(obs)
        next_value = next_value.to(torch.float32).cpu().numpy()

        self.buffer.endup(next_value)

    def train(self):
        self.policy.prep_training()

        if self.args.alg_name == "ppo":
            train_info = self.alg.train_ppo(self.buffer)
        elif self.args.alg_name == "grpo":
            train_info = self.alg.train_grpo(self.buffer)
        else:
            raise ValueError(f"Unknown alg_name: {self.args.alg_name}")

        info = {f"train/{k}": v for k, v in train_info.items()}
        info["buffer/reward_mean"] = np.mean(self.buffer.rewards)
        info["buffer/mask_mean"] = np.mean(1.0 - self.buffer.masks)

        return info

    @torch.no_grad()
    def eval(self, obj_set: str) -> dict:
        self.policy.prep_rollout()
        env_infos = defaultdict(lambda: [])

        obs_img, instruction, info = self.env.reset(obj_set=obj_set)

        for _ in range(self.args.episode_len):
            obs = dict(image=obs_img, task_description=instruction)
            value, action, logprob, _ = self._get_action(obs, deterministic=True)

            obs_img, reward, done, env_info = self.env.step(action)

            # info
            print({k: round(v.to(torch.float32).mean().tolist(), 4) for k, v in env_info.items() if k != "episode"})
            if "episode" in env_info.keys():
                for k, v in env_info["episode"].items():
                    env_infos[f"{k}"] += v

        # infos
        env_stats = {k: np.mean(v) for k, v in env_infos.items()}
        env_stats = env_stats.copy()

        print(pprint.pformat({k: round(v, 4) for k, v in env_stats.items()}))
        print(f"")

        return env_stats

    @torch.no_grad()
    def render(self, epoch: int, obj_set: str) -> dict:
        self.policy.prep_rollout()

        # init logger
        env_infos = defaultdict(lambda: [])
        datas = [{
            "image": [],  # obs_t: [0, T-1]
            "instruction": "",
            "action": [],  # a_t: [0, T-1]
            "info": [],  # info after executing a_t: [1, T]
        } for idx in range(self.args.num_envs)]

        obs_img, instruction, info = self.env.reset(obj_set)
        print("instruction[:3]:", instruction[:3])

        # data dump: instruction
        for idx in range(self.args.num_envs):
            datas[idx]["instruction"] = instruction[idx]

        for _ in range(self.args.episode_len):
            obs = dict(image=obs_img, task_description=instruction)
            value, action, logprob, _ = self._get_action(obs, deterministic=True)

            obs_img_new, reward, done, env_info = self.env.step(action)

            # info
            print({k: round(v.to(torch.float32).mean().tolist(), 4) for k, v in env_info.items() if k != "episode"})
            if "episode" in env_info.keys():
                for k, v in env_info["episode"].items():
                    env_infos[f"{k}"] += v

            for i in range(self.args.num_envs):
                post_action = self.env._process_action(action)
                log_image = obs_img[i].cpu().numpy()
                log_action = post_action[i].cpu().numpy().tolist()
                log_info = {k: v[i].tolist() for k, v in env_info.items() if k != "episode"}
                datas[i]["image"].append(log_image)
                datas[i]["action"].append(log_action)
                datas[i]["info"].append(log_info)

            # update obs_img
            obs_img = obs_img_new

        # data dump: last image
        for i in range(self.args.num_envs):
            log_image = obs_img[i].cpu().numpy()
            datas[i]["image"].append(log_image)

        # save video
        exp_dir = Path(self.glob_dir) / f"vis_{epoch}_{obj_set}"
        exp_dir.mkdir(parents=True, exist_ok=True)

        for i in range(self.args.num_envs):
            images = datas[i]["image"]
            infos = datas[i]["info"]
            assert len(images) == len(infos) + 1

            if self.args.render_info:
                for j in range(len(infos)):
                    images[j + 1] = visualization.put_info_on_image(
                        images[j + 1], infos[j],
                        extras=[f"Ins: {instruction[i]}"]
                    )

            success = int(infos[-1]["success"])
            images_to_video(images, str(exp_dir), f"video_{i}-s_{success}",
                            fps=10, verbose=False)

        # infos
        env_stats = {k: np.mean(v) for k, v in env_infos.items()}
        env_stats_ret = env_stats.copy()

        print(pprint.pformat({k: round(v, 4) for k, v in env_stats.items()}))
        print(f"")

        # save stats
        last_info = {
            idx: {k: env_infos[k][idx] for k in env_infos.keys()}
            for idx in range(self.args.num_envs)
        }

        save_stats = {}
        save_stats["env_name"] = self.args.env_id
        save_stats["ep_len"] = self.args.episode_len
        save_stats["epoch"] = epoch
        save_stats["stats"] = {k: v.item() for k, v in env_stats.items()}
        save_stats["instruction"] = {idx: ins for idx, ins in enumerate(instruction)}
        save_stats["last_info"] = last_info

        yaml.dump(save_stats, open(exp_dir / "stats.yaml", "w"))

        return env_stats_ret

    def run(self):
        max_episodes = self.args.steps_max // self.args.episode_len // self.args.num_envs

        for episode in range(max_episodes):
            env_infos = defaultdict(lambda: [])
            ep_time = time.time()

            obs_img, instruction, info = self.env.reset(obj_set="train", same_init=self.args.use_same_init)
            self.buffer.warmup(obs_img.cpu().numpy(), instruction)

            for _ in tqdm(range(self.args.episode_len), desc="rollout"):
                value, actions, logprob, generated_ids = self.collect()
                rewards = torch.zeros_like(value)  # [B, 1]
                # 如果generated_ids有小于30000的值打断点
                if torch.any(generated_ids < 30000):
                    breakpoint()
                for ac_idx in range(ACTION_CHUNCK_SIZE):
                    action = actions[:, ac_idx, :]
                    obs_img, reward, done, env_info = self.env.step(action)
                    rewards += reward  # [B, 1]
                data = (obs_img, actions, logprob, value, rewards, done, generated_ids)
                self.insert(data)

                # info
                if "episode" in env_info.keys():
                    for k, v in env_info["episode"].items():
                        env_infos[f"{k}"] += v

            # steps
            steps = (episode + 1) * self.args.episode_len * self.args.num_envs 
            print(pprint.pformat({k: round(np.mean(v), 4) for k, v in env_infos.items()}))

            # train and process infos
            self.compute_endup()
            del value, action, logprob, obs_img, reward, done
            gc.collect()
            torch.cuda.empty_cache()

            # train
            infos = self.train()
            for k, v in env_infos.items():
                infos[f"env/{k}"] = np.mean(v)

            # log
            wandb.log(infos, step=steps)

            elapsed_time = time.time() - ep_time
            print(f"{self.args.name}: ep {episode:0>4d} | steps {steps} | e {elapsed_time:.2f}s")
            print(pprint.pformat({k: round(v, 4) for k, v in infos.items()}))

            # eval
            if episode % self.args.interval_eval == self.args.interval_eval - 1 or episode == max_episodes - 1:
                print(f"Evaluating at {steps}")
                sval_stats = self.eval(obj_set="train")
                sval_stats = {f"eval/{k}": v for k, v in sval_stats.items()}
                wandb.log(sval_stats, step=steps)

                sval_stats = self.eval(obj_set="test")
                sval_stats = {f"eval/{k}_ood": v for k, v in sval_stats.items()}
                wandb.log(sval_stats, step=steps)

            # save
            if episode % self.args.interval_save == self.args.interval_save - 1 or episode == max_episodes - 1:
                print(f"Saving model at {steps}")
                save_path = self.glob_dir / f"steps_{episode:0>4d}"
                self.policy.save(save_path)

                self.render(epoch=episode, obj_set="train")
                self.render(epoch=episode, obj_set="test")


def main():
    args = tyro.cli(Args)
    runner = Runner(args)

    if args.only_render:
        ll = [
            "PutOnPlateInScene25VisionImage-v1",
            "PutOnPlateInScene25VisionTexture03-v1",
            "PutOnPlateInScene25VisionTexture05-v1",
            "PutOnPlateInScene25VisionWhole03-v1",
            "PutOnPlateInScene25VisionWhole05-v1",

            "PutOnPlateInScene25Instruct-v1",
            "PutOnPlateInScene25Plate-v1",
            "PutOnPlateInScene25Position-v1",
            "PutOnPlateInScene25EEPose-v1",
            "PutOnPlateInScene25PositionChange-v1",
            "PutOnPlateInScene25PositionChangeTo-v1"
        ]
        if args.env_id not in ll:
            runner.render(epoch=0, obj_set="train")
        runner.render(epoch=0, obj_set="test")
    else:
        runner.run()


if __name__ == "__main__":
    main()
