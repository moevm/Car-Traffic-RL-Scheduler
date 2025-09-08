from typing import Union, Optional

from sb3_contrib.common.recurrent.policies import RecurrentMultiInputActorCriticPolicy
from sb3_contrib.common.recurrent.type_aliases import RNNStates

from stable_baselines3.common.distributions import Distribution, DiagGaussianDistribution, CategoricalDistribution, \
    MultiCategoricalDistribution, BernoulliDistribution, StateDependentNoiseDistribution
from stable_baselines3.common.policies import ActorCriticPolicy
import torch as th
import numpy as np
from gymnasium import spaces


class MaskableRecurrentActorCriticPolicy(RecurrentMultiInputActorCriticPolicy):
    def _get_action_dist_from_latent_maskable(self, latent_pi: th.Tensor, action_mask: th.Tensor) -> Distribution:
        mean_actions = self.action_net(latent_pi)
        rows, columns = mean_actions.shape
        for i in range(rows):
            for j in range(columns):
                if action_mask[i, j] == 0:
                    mean_actions[i, j] = -1e9
                elif action_mask[i, j] == 2:
                    mean_actions[i, j] = 1e9
        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, BernoulliDistribution):
            # Here mean_actions are the logits (before rounding to get the binary actions)
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_pi)
        else:
            raise ValueError("Invalid action distribution")

    def forward_maskable(
            self,
            obs: th.Tensor,
            lstm_states: RNNStates,
            episode_starts: th.Tensor,
            action_mask: th.Tensor,
            deterministic: bool = False
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, RNNStates]:
        features = self.extract_features(obs)
        if self.share_features_extractor:
            pi_features = vf_features = features  # alis
        else:
            pi_features, vf_features = features
        # latent_pi, latent_vf = self.mlp_extractor(features)
        latent_pi, lstm_states_pi = self._process_sequence(pi_features, lstm_states.pi, episode_starts, self.lstm_actor)
        if self.lstm_critic is not None:
            latent_vf, lstm_states_vf = self._process_sequence(vf_features, lstm_states.vf, episode_starts,
                                                               self.lstm_critic)
        elif self.shared_lstm:
            # Re-use LSTM features but do not backpropagate
            latent_vf = latent_pi.detach()
            lstm_states_vf = (lstm_states_pi[0].detach(), lstm_states_pi[1].detach())
        else:
            # Critic only has a feedforward network
            latent_vf = self.critic(vf_features)
            lstm_states_vf = lstm_states_pi

        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent_maskable(latent_pi, action_mask)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob, RNNStates(lstm_states_pi, lstm_states_vf)

    def evaluate_actions_maskable(
            self, obs: th.Tensor, actions: th.Tensor, lstm_states: RNNStates, episode_starts: th.Tensor,
            action_mask: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        features = self.extract_features(obs)
        if self.share_features_extractor:
            pi_features = vf_features = features  # alias
        else:
            pi_features, vf_features = features
        latent_pi, _ = self._process_sequence(pi_features, lstm_states.pi, episode_starts, self.lstm_actor)
        if self.lstm_critic is not None:
            latent_vf, _ = self._process_sequence(vf_features, lstm_states.vf, episode_starts, self.lstm_critic)
        elif self.shared_lstm:
            latent_vf = latent_pi.detach()
        else:
            latent_vf = self.critic(vf_features)

        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        distribution = self._get_action_dist_from_latent_maskable(latent_pi, action_mask)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()

    def get_distribution_maskable(
            self,
            obs: th.Tensor,
            lstm_states: tuple[th.Tensor, th.Tensor],
            episode_starts: th.Tensor,
            action_mask: th.Tensor
    ) -> tuple[Distribution, tuple[th.Tensor, ...]]:
        features = super(ActorCriticPolicy, self).extract_features(obs, self.pi_features_extractor)
        latent_pi, lstm_states = self._process_sequence(features, lstm_states, episode_starts, self.lstm_actor)
        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        return self._get_action_dist_from_latent_maskable(latent_pi, action_mask), lstm_states

    def predict_maskable(
            self,
            observation: Union[np.ndarray, dict[str, np.ndarray]],
            action_mask: th.Tensor,
            state: Optional[tuple[np.ndarray, ...]] = None,
            episode_start: Optional[np.ndarray] = None,
            deterministic: bool = False,
    ) -> tuple[np.ndarray, Optional[tuple[np.ndarray, ...]]]:
        self.set_training_mode(False)

        observation, vectorized_env = self.obs_to_tensor(observation)

        if isinstance(observation, dict):
            n_envs = observation[next(iter(observation.keys()))].shape[0]
        else:
            n_envs = observation.shape[0]
        # state : (n_layers, n_envs, dim)
        if state is None:
            # Initialize hidden states to zeros
            state = np.concatenate([np.zeros(self.lstm_hidden_state_shape) for _ in range(n_envs)], axis=1)
            state = (state, state)

        if episode_start is None:
            episode_start = np.array([False for _ in range(n_envs)])

        with th.no_grad():
            # Convert to PyTorch tensors
            states = th.tensor(state[0], dtype=th.float32, device=self.device), th.tensor(
                state[1], dtype=th.float32, device=self.device
            )
            episode_starts = th.tensor(episode_start, dtype=th.float32, device=self.device)
            actions, states = self._predict_maskable(
                observation, action_mask, lstm_states=states, episode_starts=episode_starts, deterministic=deterministic
            )
            states = (states[0].cpu().numpy(), states[1].cpu().numpy())

        # Convert to numpy
        actions = actions.cpu().numpy()

        if isinstance(self.action_space, spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.action_space.low, self.action_space.high)

        # Remove batch dimension if needed
        if not vectorized_env:
            actions = actions.squeeze(axis=0)

        return actions, states

    def _predict_maskable(
            self,
            observation: th.Tensor,
            action_mask: th.Tensor,
            lstm_states: tuple[th.Tensor, th.Tensor],
            episode_starts: th.Tensor,
            deterministic: bool = False,
    ) -> tuple[th.Tensor, tuple[th.Tensor, ...]]:
        distribution, lstm_states = self.get_distribution_maskable(observation, lstm_states, episode_starts,
                                                                   action_mask)
        return distribution.get_actions(deterministic=deterministic), lstm_states
