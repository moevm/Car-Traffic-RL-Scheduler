import numpy as np

from stable_baselines3.common.callbacks import BaseCallback


class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0, group_size=4):
        super().__init__(verbose)
        self.__group_size = group_size
        self.__rollout_counter = 0

    def _on_rollout_start(self) -> None:
        self.__rollout_rewards = {
            "normalized_sum_reward": np.zeros(shape=(self.training_env.num_envs,), dtype=np.float32),
            "sum_reward": np.zeros(shape=(self.training_env.num_envs,), dtype=np.float32),
            "step_capacity": np.zeros(shape=(self.training_env.num_envs,), dtype=np.float32),
            "phase_capacity": np.zeros(shape=(self.training_env.num_envs,), dtype=np.float32),
            "tls_reward": np.zeros(shape=(self.training_env.num_envs,), dtype=np.float32),
            "halting_reward": np.zeros(shape=(self.training_env.num_envs,), dtype=np.float32),
        }

    def __update_statistics_from_info(self, info, i):
        for reward_type in info:
            if reward_type != 'TimeLimit.truncated':
                reward = self.__rollout_rewards[reward_type][i]
                self.__rollout_rewards[reward_type][i] = (reward + (1 / (self.locals["n_steps"] + 1)) *
                                                          (info[reward_type] - reward))

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        for i, info in enumerate(infos):
            self.__update_statistics_from_info(info, i)
        self.__rollout_rewards["normalized_sum_reward"] = self.__rollout_rewards["normalized_sum_reward"] + (
                1 / (self.locals["n_steps"] + 1)) * (self.locals["rewards"] - self.__rollout_rewards[
            "normalized_sum_reward"])
        self.__log_norms()
        return True

    def _on_rollout_end(self) -> None:
        for reward_type in self.__rollout_rewards:
            mean_reward_type = np.mean(self.__rollout_rewards[reward_type])
            self.logger.record(f"rollout/mean_{reward_type}", mean_reward_type)
        if self.__rollout_counter % 100 == 0:
            self.model.save(
                f"./pre_trained_models/pre_trained_model_{self.__rollout_counter * (self.locals["n_steps"] + 1) * 
                                                          self.training_env.num_envs}")
            self.model.get_vec_normalize_env().save(
                f'./pre_trained_models/pre_vec_normalize_{self.__rollout_counter * (self.locals["n_steps"] + 1) * 
                                                          self.training_env.num_envs}.pkl')
        self.__rollout_counter += 1

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
        self.logger.record("step/value_norm", total_value_norm)
