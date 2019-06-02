import gym
from gym import spaces
import numpy as np

from model import Model
from env import WallAvoidingAgent
from model_memory import MDNRNN, hps
from model_vision import BetaVAE

hps = hps._replace(batch_size=1, max_seq_len=2, use_recurrent_dropout=0, is_training=0)
class ControllerEnv(gym.Env):
    def __init__(self, stage=2, vae_load=None, rnn_load=None, temperature=1.0):
        assert vae_load or rnn_load
        self.stage = stage
        self.vae_load = vae_load
        self.rnn_load = rnn_load
        self.temperature = temperature

        self.model = None
        self.vae = BetaVAE(batch_size=1)
        self.rnn = None
        self._seed = None

        self.step_count = 0
        self.obs = None
        self.z = None
        self.reward = 0.0
        self.done = None
        self.info = None
        self.zero_state = None
        self.rnn_state = None

        self.action_scale = 0.4

        self.reset()

        self.action_space = spaces.Box(-5.0 / self.action_scale, 5.0 / self.action_scale, shape=())
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(hps.seq_width, ))

        if vae_load:
            self.model = Model(stage=stage)
            self.vae.load_json(vae_load)


        if rnn_load:
            self.rnn = MDNRNN(hps)
            self.rnn.load_json(rnn_load)
            self.zero_state = self.rnn.sess.run(self.rnn.zero_state)

            self.observation_space = spaces.Box(-np.inf, np.inf, shape=(hps.seq_width + hps.rnn_size * 2, ))

    def seed(self, seed=None):
        #self._seed = seed
        self._seed = None

    def reset(self):
        if self.model:
            self.obs = self.model.env.reset(self._seed)
            self._seed = None
            self.z = self._encode(self.obs)
        else:
            self.z = np.clip(np.random.rand(self.rnn.hps.seq_width), -1, 1)

        self.rnn_state = self.zero_state
        self.done = True
        return self.get_obs()

    def step(self, action):
        action = np.clip(action[0] * self.action_scale, -5.0, 5.0)

        prev_z = np.zeros((1, 1, hps.seq_width))
        prev_z[0][0] = self.z

        prev_action = np.zeros((1, 1))
        prev_action[0] = action

        prev_restart = np.zeros((1, 1))
        prev_restart[0] = 1.0 if self.done else 0.0

        s_model = self.rnn

        if s_model:
            feed = {s_model.input_z: prev_z,
                    s_model.input_action: prev_action,
                    s_model.input_restart: prev_restart,
                    s_model.initial_state: self.rnn_state
                    }

            logmix, mean, logstd, self.rnn_state = s_model.sess.run(s_model.target_z, s_model.final_state, feed)
            self.z = s_model.get_next_z(logmix, mean, logstd, self.temperature)

        if self.model:
            self.obs, self.reward, self.done, self.info = self.model.env.step(action)
            self.z = self._encode(self.obs)


        # punish if the agent stay on the wall
        self.step_count += 1
        if WallAvoidingAgent.on_wall(self.obs, th=0.5):
            self.reward -= 0.01

        return self.get_obs(), self.reward, self.done, self.info

    def render(self, mode='human'):
        return

    def close(self):
        self.model.env.lab.close()

    def _encode(self, obs):
        obs = np.array(obs).astype(np.float) / 255.0

        return self.vae.encode(obs[np.newaxis, ...])[0]

    def get_obs(self):
        if self.rnn:
            return np.concatenate([self.z, self.rnn_state.c.flatten(), self.rnn_state.h.flatten()], axis=0)
        else:
            return self.z

    def reproduce(self):
        return ControllerEnv(self.stage, self.vae_load, self.rnn_load)
