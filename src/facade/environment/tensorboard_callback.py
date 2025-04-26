import numpy as np

from stable_baselines3.common.callbacks import BaseCallback


class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.__abs_max_ratio = float("-inf")
        self.__abs_max_diff = float("-inf")

    def _on_rollout_start(self) -> None:
        self.__rollout_normalized_rewards = np.zeros(shape=(self.training_env.num_envs,), dtype=np.float32)
        self.__rollout_rewards = np.zeros(shape=(self.training_env.num_envs,), dtype=np.float32)
        self.__rollout_step_capacity = np.zeros(shape=(self.training_env.num_envs,), dtype=np.float32)
        self.__rollout_phase_capacity = np.zeros(shape=(self.training_env.num_envs,), dtype=np.float32)
        self.__rollout_reward_ratio = np.zeros(shape=(self.training_env.num_envs,), dtype=np.float32)

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        (reward, step_capacity, phase_capacity, group_reward, group_step_capacity,
         group_phase_capacity, reward_ratio, group_reward_ratio) = (
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        for i, info in enumerate(infos):
            timestep_rewards = info["timestep_rewards"]
            group_rewards = info["group_rewards"]
            self.__rollout_rewards[i] = self.__rollout_rewards[i] + (1 / (self.locals["n_steps"] + 1)) * (
                    timestep_rewards["sum_reward"] - self.__rollout_rewards[i])
            self.__rollout_step_capacity[i] = self.__rollout_step_capacity[i] + (1 / (self.locals["n_steps"] + 1)) * (
                    timestep_rewards["step_capacity"] - self.__rollout_step_capacity[i])
            self.__rollout_phase_capacity[i] = self.__rollout_phase_capacity[i] + (1 / (self.locals["n_steps"] + 1)) * (
                    timestep_rewards["phase_capacity"] - self.__rollout_phase_capacity[i])
            self.__rollout_reward_ratio[i] = self.__rollout_reward_ratio[i] + (1 / (self.locals["n_steps"] + 1)) * (
                    timestep_rewards["reward_ratio"] - self.__rollout_reward_ratio[i])

            reward = reward + (1 / len(infos)) * (timestep_rewards["sum_reward"] - reward)
            step_capacity = step_capacity + (1 / len(infos)) * (timestep_rewards["step_capacity"] - step_capacity)
            phase_capacity = phase_capacity + (1 / len(infos)) * (timestep_rewards["phase_capacity"] - phase_capacity)
            reward_ratio = reward_ratio + (1 / len(infos)) * (timestep_rewards["reward_ratio"] - reward_ratio)

            group_reward = group_reward + (1 / len(infos)) * (group_rewards["sum_reward"] - group_reward)
            group_step_capacity = group_step_capacity + (1 / len(infos)) * (
                    group_rewards["step_capacity"] - group_step_capacity)
            group_phase_capacity = group_phase_capacity + (1 / len(infos)) * (
                    group_rewards["phase_capacity"] - group_phase_capacity)
            group_reward_ratio = group_reward_ratio + (1 / len(infos)) * (
                    group_rewards["reward_ratio"] - group_reward_ratio)
            if group_rewards["step_capacity"] != 0 and abs(group_rewards["phase_capacity"] / group_rewards[
                "step_capacity"]) > self.__abs_max_ratio:
                self.__abs_max_ratio = abs(group_rewards["phase_capacity"] / group_rewards["step_capacity"])
            if abs(group_rewards["phase_capacity"] - group_rewards["step_capacity"]) > self.__abs_max_diff:
                self.__abs_max_diff = abs(group_rewards["phase_capacity"] - group_rewards["step_capacity"])
        self.__rollout_normalized_rewards = self.__rollout_normalized_rewards + (1 / (self.locals["n_steps"] + 1)) * (
                self.locals["rewards"] - self.__rollout_normalized_rewards)
        self.__log_rewards(reward,
                           step_capacity,
                           phase_capacity,
                           reward_ratio,
                           group_reward,
                           group_step_capacity,
                           group_phase_capacity,
                           group_reward_ratio)
        self.__log_norms()
        return True

    def _on_rollout_end(self) -> None:
        mean_normalized_reward = np.mean(self.__rollout_normalized_rewards)
        mean_rollout_reward = np.mean(self.__rollout_rewards)
        mean_rollout_step_capacity = np.mean(self.__rollout_step_capacity)
        mean_rollout_phase_capacity = np.mean(self.__rollout_phase_capacity)
        mean_rollout_reward_ratio = np.mean(self.__rollout_reward_ratio)
        self.logger.record("rollout/mean_normalized_reward", mean_normalized_reward)
        self.logger.record("rollout/mean_reward", mean_rollout_reward)
        self.logger.record("rollout/mean_step_capacity", mean_rollout_step_capacity)
        self.logger.record("rollout/mean_phase_capacity", mean_rollout_phase_capacity)
        self.logger.record("rollout/mean_reward_ratio", mean_rollout_reward_ratio)
        self.logger.record("rollout/max_ratio", self.__abs_max_ratio)
        self.logger.record("rollout/max_diff", self.__abs_max_diff)

    def __log_rewards(self, reward: float,
                      step_capacity: float,
                      phase_capacity: float,
                      reward_ratio: float,
                      group_reward: float,
                      group_step_capacity: float,
                      group_phase_capacity: float,
                      group_reward_ratio: float) -> None:
        self.logger.record("step/reward", reward)
        self.logger.record("step/step_capacity", step_capacity)
        self.logger.record("step/phase_capacity", phase_capacity)
        self.logger.record("step/reward_ratio", reward_ratio)

        self.logger.record("step/group_reward", group_reward)
        self.logger.record("step/group_step_capacity", group_step_capacity)
        self.logger.record("step/group_phase_capacity", group_phase_capacity)
        self.logger.record("step/group_reward_ratio", group_reward_ratio)

    def __log_norms(self) -> None:
        total_policy_norm = 0.0
        for p in self.model.policy.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_policy_norm += param_norm.item() ** 2
        total_policy_norm = total_policy_norm ** 0.5
        self.logger.record("step/policy_norm", total_policy_norm)
        total_value_norm = 0.0
        for p in self.model.policy.value_net.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_value_norm += param_norm.item() ** 2
        total_value_norm = total_value_norm ** 0.5
        self.logger.record("step/value norm", total_value_norm)
