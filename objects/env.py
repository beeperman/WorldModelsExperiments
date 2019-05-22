import deepmind_lab
import numpy as np
from scipy.misc import imresize as resize

SCREEN_X = 64
SCREEN_Y = 64


def _process_frame(frame):
    # obs = np.array(frame[0:400, :, :]).astype(np.float)/255.0
    obs = np.array(frame).astype(np.float) / 255.0
    obs = np.array(resize(obs, (SCREEN_Y, SCREEN_X)))
    obs = ((1.0 - obs) * 255).round().astype(np.uint8)
    return obs


class CollectGoodObjectsTrainEnv(object):
    """Collect good object for training
    will test with unseen environment"""

    def __init__(self, continuous=True):
        """ continuous or discrete action space """
        self.continuous = continuous
        self.old_obs = None
        if continuous:
            self.action_space = "scalar continuous from -5 to 5"
        else:
            self.action_space = "scalar discrete: left, right, move forward"

        self.lab = deepmind_lab.Lab("contributed/dmlab30/rooms_collect_good_objects_train", ['RGB_INTERLEAVED'],
                                    {'fps': '30', })

    def reset(self, seed=None):
        if seed:
            self.lab.reset(seed)
        else:
            self.lab.reset()
        return self.get_obs()

    def step(self, action):
        action_array = np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.intc)

        if self.continuous:
            assert isinstance(action, np.float)
            assert action >= -5.1 and action <= 5.1
            action = np.intc(action * 100)
            action_array[3] = 1  # move forward
            action_array[0] = action  # control direction
        else:
            assert isinstance(action, np.int)
            assert action >= 0 and action < 3
            if action == 0:  # turn left
                action_array[0] = -64
            elif action == 1:  # turn right
                action_array[0] = 64
            else:  # move forward
                action_array[3] = 1

        reward = self.lab.step(action_array)
        done = not self.lab.is_running()
        if done:
            obs = self.old_obs
        else:
            obs = self.get_obs()
            self.old_obs = obs
        info = self.lab.events()

        return obs, reward, done, info

    def get_obs(self):
        return _process_frame(self.lab.observations()['RGB_INTERLEAVED'])

    def remake(self):
        self.lab.close()
        self.lab = deepmind_lab.Lab("contributed/dmlab30/rooms_collect_good_objects_train", ['RGB_INTERLEAVED'],
                                    {'fps': '30', })


class WallAvoidingAgent(object):
    def __init__(self, env, repeat=0, var=0.2):
        assert env.continuous, "must be in continuous setting"
        self.repeat = repeat
        self.count = repeat
        self.action = 0.0
        self.var = var

    def get_action(self, obs):
        if self.on_wall(obs, th=0.5) and np.random.random_sample() < 0.3:
            return 5.1
        else:
            if self.count >= self.repeat:
                self.action = np.clip(np.random.normal(scale=self.var), -5, 5)
                self.count = 0
            else:
                self.action = 0.0
                self.count += 1
            return self.action

    def on_wall(self, obs, th):
        obs = np.reshape(obs, [np.prod(obs.shape[:-1]), obs.shape[-1]])

        def color_match(p):
            if 85 <= p[0] <= 105 and 115 <= p[1] <= 135 and 75 <= p[2] <= 95 \
                    and 183 <= p[0] <= 133 and 103 <= p[1] <= 123 and 85 <= p[2] <= 105:
                return True
            else:
                return False

        count = np.sum([1.0 if color_match(obs[i]) else 0.0 for i in range(obs.shape[0])], dtype=np.float)
        return True if count / obs.shape[0] > th else False
