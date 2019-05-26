import deepmind_lab
import numpy as np
import random
from scipy.misc import imresize as resize

SCREEN_X = 64
SCREEN_Y = 64


def _process_frame(frame):
    # obs = np.array(frame[0:400, :, :]).astype(np.float)/255.0
    obs = np.array(frame).astype(np.float) / 255.0
    obs = np.array(resize(obs, (SCREEN_Y, SCREEN_X)))
    obs = ((1.0 - obs) * 255).round().astype(np.uint8)
    return obs


class CollectGoodObjectsEnv(object):
    """Collect good object environment choose from
    train
    test
    balloon_cake_false
    balloon_cake_true
    balloon_can_false
    balloon_can_true
    balloon_false
    balloon_hat_false
    balloon_hat_true
    balloon_true
    cake_can_false
    cake_can_true
    cake_false
    cake_hat_false
    cake_hat_true
    cake_true
    can_false
    can_true
    hat_can_false
    hat_can_true
    hat_false
    hat_true
    """

    def __init__(self, name="train", continuous=True):
        """ continuous or discrete action space """
        self.names = name if isinstance(name, list) else [name]
        self.continuous = continuous
        self.old_obs = None
        if continuous:
            self.action_space = "scalar continuous from -5 to 5"
        else:
            self.action_space = "scalar discrete: left, right, move forward"

        #self.lab = deepmind_lab.Lab("contributed/dmlab30/rooms_collect_good_objects_train", ['RGB_INTERLEAVED'],
        #                            {'fps': '30', })
        self.labs = None
        self.remake(close=False)

        self.lab = None
        self.lab_name = None

    def reset(self, seed=None):
        sampled_lab = random.sample(self.labs, 1)[0]
        self.lab_name = sampled_lab[0]
        self.lab = sampled_lab[1]
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

    def remake(self, close=True):
        if close:
            [lab[1].close() for lab in self.labs]

        self.labs = [(n, deepmind_lab.Lab(self.translate(n), ['RGB_INTERLEAVED'], {'fps': '30', 'allowHoldOutLevels': 'true'})) for n in self.names]

    def translate(self, name):
        if name in ["train", "test"]:
            return "contributed/dmlab30/rooms_collect_good_objects_{}".format(name)
        else:
            return "contributed/dmlab30/objects/{}".format(name)


class WallAvoidingAgent(object):
    def __init__(self, env, repeat=0, var=0.16, shift=0.4, wall_steps=15):
        assert env.continuous, "must be in continuous setting"
        self.repeat = repeat
        self.count = repeat
        self.action = 0.0

        self.std = np.sqrt(var)
        self.mean = 0.0
        self.shift = shift

        self.wall_avoiding_steps = wall_steps
        self.wall_count = 0

    def get_action(self, obs):
        # wall avoiding
        if self.wall_count > 0:
            self.wall_count -= 1
        else:
            self.mean = 0.0

        # wall detecting
        if self.on_wall(obs, th=0.2) and np.random.random_sample() < 0.3 and self.wall_count <= 0:
            self.wall_count = self.wall_avoiding_steps
            self.mean = random.choice([-1.0, 1.0]) * self.shift

        # action generation
        if self.count >= self.repeat:
            self.action = np.clip(np.random.normal(loc=self.mean, scale=self.std), -5, 5)
            self.count = 0
        else:
            self.action = 0.0
            self.count += 1
        return self.action


    def on_wall(self, obs, th):
        obs = np.reshape(obs, [np.prod(obs.shape[:-1]), obs.shape[-1]])

        def color_match(p):
            if (85 <= p[0] <= 105 and 115 <= p[1] <= 135 and 75 <= p[2] <= 95) \
                    or (183 <= p[0] <= 233 and 103 <= p[1] <= 123 and 85 <= p[2] <= 105):
                return True
            else:
                return False

        count = np.sum([1.0 if color_match(obs[i]) else 0.0 for i in range(obs.shape[0])], dtype=np.float)
        return True if count / obs.shape[0] > th else False
