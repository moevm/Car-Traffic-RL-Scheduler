import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_rollout_start(self) -> None:
        self.__rollout_rewards = np.zeros(shape=(self.training_env.num_envs,), dtype=np.float32)
        self.__rollout_capacity = np.zeros(shape=(self.training_env.num_envs,), dtype=np.int32)
        self.__rollout_tls_rewards = np.zeros(shape=(self.training_env.num_envs,), dtype=np.float32)
        self.__rollout_step_capacity = np.zeros(shape=(self.training_env.num_envs,), dtype=np.float32)
        self.__rollout_phase_capacity = np.zeros(shape=(self.training_env.num_envs,), dtype=np.float32)
        self.__rollout_halting_reward = np.zeros(shape=(self.training_env.num_envs,), dtype=np.float32)

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        reward, tls_reward, step_capacity, phase_capacity, halting_reward = 0, 0, 0, 0, 0
        for i, info in enumerate(infos):
            # self.__rollout_capacity[i] += info["capacity"]
            self.__rollout_rewards[i] = self.__rollout_rewards[i] + (1 / (self.locals["n_steps"] + 1)) * (
                    info["reward"] - self.__rollout_rewards[i])
            self.__rollout_tls_rewards[i] = self.__rollout_tls_rewards[i] + (1 / (self.locals["n_steps"] + 1)) * (
                    info["tls_reward"] - self.__rollout_tls_rewards[i])
            self.__rollout_step_capacity[i] = self.__rollout_step_capacity[i] + (1 / (self.locals["n_steps"] + 1)) * (
                    info["step_capacity"] - self.__rollout_step_capacity[i])
            self.__rollout_phase_capacity[i] = self.__rollout_phase_capacity[i] + (1 / (self.locals["n_steps"] + 1)) * (
                    info["phase_capacity"] - self.__rollout_phase_capacity[i])
            self.__rollout_halting_reward[i] = self.__rollout_halting_reward[i] + (1 / (self.locals["n_steps"] + 1)) * (
                    info["halting_reward"] - self.__rollout_halting_reward[i])
            reward = reward + (1 / len(infos)) * (info["reward"] - reward)
            tls_reward = tls_reward + (1 / len(infos)) * (info["tls_reward"] - tls_reward)
            step_capacity = step_capacity + (1 / len(infos)) * (info["step_capacity"] - step_capacity)
            phase_capacity = phase_capacity + (1 / len(infos)) * (info["phase_capacity"] - phase_capacity)
            halting_reward = halting_reward + (1 / len(infos)) * (info["halting_reward"] - halting_reward)
        self.logger.record("step/reward", reward)
        self.logger.record("step/tls_reward", tls_reward)
        self.logger.record("step/step_capacity", step_capacity)
        self.logger.record("step/phase_capacity", phase_capacity)
        self.logger.record("step/halting_reward", halting_reward)
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
        if infos[0]["phase_duration"] != -1:
            self.logger.record("step/phase_duration", infos[0]["phase_duration"])
        return True

    def _on_rollout_end(self) -> None:
        mean_rollout_reward = np.mean(self.__rollout_rewards)
        mean_rollout_tls_reward = np.mean(self.__rollout_tls_rewards)
        mean_rollout_step_capacity = np.mean(self.__rollout_step_capacity)
        mean_rollout_phase_capacity = np.mean(self.__rollout_phase_capacity)
        # mean_capacity = np.mean(self.__rollout_capacity)
        mean_rollout_halting_reward = np.mean(self.__rollout_halting_reward)
        self.logger.record("rollout/mean_reward", mean_rollout_reward)
        self.logger.record("rollout/mean_tls_reward", mean_rollout_tls_reward)
        self.logger.record("rollout/mean_step_capacity", mean_rollout_step_capacity)
        self.logger.record("rollout/mean_phase_capacity", mean_rollout_phase_capacity)
        self.logger.record("rollout/mean_halting_reward", mean_rollout_halting_reward)
        # self.logger.record("rollout/mean_capacity", mean_capacity)