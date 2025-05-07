import numpy as np

from stable_baselines3.common.callbacks import BaseCallback


class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0, group_size=4):
        super().__init__(verbose)
        self.__abs_phase_div_step = float("-inf")
        self.__abs_phase_div_tls = float("-inf")
        self.__abs_tls_div_step = float("-inf")
        self.__max_abs_phase = float("-inf")
        self.__max_abs_step = float("-inf")
        self.__max_abs_tls = float("-inf")
        self.__group_size = group_size
        self.__rollout_counter = 0

    def _on_rollout_start(self) -> None:
        self.__rollout_normalized_rewards = np.zeros(shape=(self.training_env.num_envs,), dtype=np.float32)
        self.__rollout_rewards = np.zeros(shape=(self.training_env.num_envs,), dtype=np.float32)
        self.__rollout_step_capacity = np.zeros(shape=(self.training_env.num_envs,), dtype=np.float32)
        self.__rollout_phase_capacity = np.zeros(shape=(self.training_env.num_envs,), dtype=np.float32)
        self.__rollout_tls_reward = np.zeros(shape=(self.training_env.num_envs,), dtype=np.float32)

        self.__rollout_normalized_timestep_rewards = np.zeros(shape=(self.training_env.num_envs,), dtype=np.float32)
        self.__rollout_timestep_rewards = np.zeros(shape=(self.training_env.num_envs,), dtype=np.float32)
        self.__rollout_timestep_step_capacity = np.zeros(shape=(self.training_env.num_envs,), dtype=np.float32)
        self.__rollout_timestep_phase_capacity = np.zeros(shape=(self.training_env.num_envs,), dtype=np.float32)
        self.__rollout_timestep_tls_reward = np.zeros(shape=(self.training_env.num_envs,), dtype=np.float32)

        self.__reset_timestep_rewards()

    def __reset_timestep_rewards(self) -> None:
        self.__normalized_timestep_rewards = np.zeros(shape=(self.training_env.num_envs,), dtype=np.float32)
        self.__timestep_rewards = np.zeros(shape=(self.training_env.num_envs,), dtype=np.float32)
        self.__timestep_step_capacity = np.zeros(shape=(self.training_env.num_envs,), dtype=np.float32)
        self.__timestep_phase_capacity = np.zeros(shape=(self.training_env.num_envs,), dtype=np.float32)
        self.__timestep_tls_reward = np.zeros(shape=(self.training_env.num_envs,), dtype=np.float32)

    def __update_statistics_from_info(self, info, i):
        self.__rollout_rewards[i] = self.__rollout_rewards[i] + (1 / (self.locals["n_steps"] + 1)) * (
                info["sum_reward"] - self.__rollout_rewards[i])
        self.__rollout_step_capacity[i] = self.__rollout_step_capacity[i] + (1 / (self.locals["n_steps"] + 1)) * (
                info["step_capacity"] - self.__rollout_step_capacity[i])
        self.__rollout_phase_capacity[i] = self.__rollout_phase_capacity[i] + (1 / (self.locals["n_steps"] + 1)) * (
                info["phase_capacity"] - self.__rollout_phase_capacity[i])
        self.__rollout_tls_reward[i] = self.__rollout_tls_reward[i] + (1 / (self.locals["n_steps"] + 1)) * (
                info["tls_reward"] - self.__rollout_tls_reward[i])
        self.__timestep_rewards[i] += info["sum_reward"]
        self.__timestep_step_capacity[i] += info["step_capacity"]
        self.__timestep_phase_capacity[i] += info["phase_capacity"]
        self.__timestep_tls_reward[i] += info["tls_reward"]

        if info["step_capacity"] != 0 and abs(info["phase_capacity"] / info[
            "step_capacity"]) > self.__abs_phase_div_step:
            self.__abs_phase_div_step = abs(info["phase_capacity"] / info["step_capacity"])

        if info["tls_reward"] != 0 and abs(info["phase_capacity"] / info[
            "tls_reward"]) > self.__abs_phase_div_tls:
            self.__abs_phase_div_tls = abs(info["phase_capacity"] / info["tls_reward"])

        if info["step_capacity"] != 0 and abs(info["tls_reward"] / info[
            "step_capacity"]) > self.__abs_tls_div_step:
            self.__abs_tls_div_step = abs(info["tls_reward"] / info["step_capacity"])

        if info["step_capacity"] > self.__max_abs_step:
            self.__max_abs_step = info["step_capacity"]

        if info["phase_capacity"] > self.__max_abs_phase:
            self.__max_abs_phase = info["phase_capacity"]

        if info["tls_reward"] > self.__max_abs_tls:
            self.__max_abs_tls = info["tls_reward"]

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        for i, info in enumerate(infos):
            self.__update_statistics_from_info(info, i)
        self.__rollout_normalized_rewards = self.__rollout_normalized_rewards + (1 / (self.locals["n_steps"] + 1)) * (
                self.locals["rewards"] - self.__rollout_normalized_rewards)
        self.__normalized_timestep_rewards += self.locals["rewards"]
        if (self.locals["n_steps"] + 1) % self.__group_size == 0:
            self.__rollout_normalized_timestep_rewards += self.__normalized_timestep_rewards
            self.__rollout_timestep_rewards += self.__timestep_rewards
            self.__rollout_timestep_step_capacity += self.__timestep_step_capacity
            self.__rollout_timestep_phase_capacity += self.__timestep_phase_capacity
            self.__rollout_timestep_tls_reward += self.__timestep_tls_reward

            self.__log_timestep_rewards(np.mean(self.__normalized_timestep_rewards), np.mean(self.__timestep_rewards),
                                        np.mean(self.__timestep_step_capacity), np.mean(self.__timestep_phase_capacity),
                                        np.mean(self.__timestep_tls_reward))
            self.__reset_timestep_rewards()
            self.__log_norms()
        return True

    def _on_rollout_end(self) -> None:
        mean_rollout_normalized_reward = np.mean(self.__rollout_normalized_rewards)
        mean_rollout_reward = np.mean(self.__rollout_rewards)
        mean_rollout_step_capacity = np.mean(self.__rollout_step_capacity)
        mean_rollout_phase_capacity = np.mean(self.__rollout_phase_capacity)
        mean_rollout_tls_reward = np.mean(self.__rollout_tls_reward)

        mean_rollout_normalized_timestep_reward = np.mean(self.__rollout_normalized_timestep_rewards)
        mean_rollout_timestep_reward = np.mean(self.__rollout_timestep_rewards)
        mean_rollout_timestep_step_capacity = np.mean(self.__rollout_timestep_step_capacity)
        mean_rollout_timestep_phase_capacity = np.mean(self.__rollout_timestep_phase_capacity)
        mean_rollout_timestep_tls_reward = np.mean(self.__rollout_timestep_tls_reward)

        self.logger.record("rollout/mean_normalized_reward", mean_rollout_normalized_reward)
        self.logger.record("rollout/mean_reward", mean_rollout_reward)
        self.logger.record("rollout/mean_step_capacity", mean_rollout_step_capacity)
        self.logger.record("rollout/mean_phase_capacity", mean_rollout_phase_capacity)
        self.logger.record("rollout/mean_rollout_tls_reward", mean_rollout_tls_reward)

        self.logger.record("rollout/mean_normalized_timestep_reward", mean_rollout_normalized_timestep_reward)
        self.logger.record("rollout/mean_timestep_reward", mean_rollout_timestep_reward)
        self.logger.record("rollout/mean_timestep_step_capacity", mean_rollout_timestep_step_capacity)
        self.logger.record("rollout/mean_timestep_phase_capacity", mean_rollout_timestep_phase_capacity)
        self.logger.record("rollout/mean_timestep_tls_reward", mean_rollout_timestep_tls_reward)

        self.logger.record("rollout/abs_phase_div_step", self.__abs_phase_div_step)
        self.logger.record("rollout/abs_phase_div_tls", self.__abs_phase_div_tls)
        self.logger.record("rollout/abs_tls_div_step", self.__abs_tls_div_step)
        self.logger.record("rollout/max_abs_phase", self.__max_abs_phase)
        self.logger.record("rollout/max_abs_step", self.__max_abs_step)
        self.logger.record("rollout/max_abs_tls", self.__max_abs_tls)

        if self.__rollout_counter % 100 == 0:
            self.model.save(
                f"./pre_trained_models/pre_trained_model_{self.__rollout_counter * (self.locals["n_steps"] + 1) * self.training_env.num_envs}")
            self.model.get_vec_normalize_env().save(
                f'./pre_trained_models/pre_vec_normalize_{self.__rollout_counter * (self.locals["n_steps"] + 1) * self.training_env.num_envs}.pkl')
        self.__rollout_counter += 1

    def __log_timestep_rewards(self, normalized_reward: np.float32, reward: np.float32, step_capacity: np.float32,
                               phase_capacity: np.float32, tls_reward: np.float32) -> None:
        self.logger.record("timestep/normalized_reward", normalized_reward)
        self.logger.record("timestep/reward", reward)
        self.logger.record("timestep/step_capacity", step_capacity)
        self.logger.record("timestep/phase_capacity", phase_capacity)
        self.logger.record("timestep/tls_reward", tls_reward)

    def __log_norms(self) -> None:
        total_policy_norm = 0.0
        for p in self.model.policy.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_policy_norm += param_norm.item() ** 2
        total_policy_norm = total_policy_norm ** 0.5
        self.logger.record("timestep/policy_norm", total_policy_norm)
        total_value_norm = 0.0
        for p in self.model.policy.value_net.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_value_norm += param_norm.item() ** 2
        total_value_norm = total_value_norm ** 0.5
        self.logger.record("timestep/value_norm", total_value_norm)
