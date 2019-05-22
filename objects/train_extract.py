import os

import argparse

from env import WallAvoidingAgent
from model import Model
import numpy as np
import random

parser = argparse.ArgumentParser(description=('collect data for training and testing'))
parser.add_argument("-t", type=int, default=200, help="how many trials it will run for")
parser.add_argument("-s", type=int, default=1, help="the stage to extract, choose from 1,2,3,4,5")
args = parser.parse_args()

MAX_FRAMES = 1800
MAX_TRIALS = args.t
MIN_LENGTH = 100

DIR_NAME = 'train_record'
if not os.path.exists(DIR_NAME):
    os.makedirs(DIR_NAME)

model = Model(stage=args.s, initialize=False)

total_frames = 0

for trial in range(MAX_TRIALS):  # 200 trials per worker
    try:
        random_generated_int = random.randint(0, 2 ** 31 - 1)
        filename = DIR_NAME + "/{}_" + str(random_generated_int) + ".npz"
        recording_obs = []
        recording_action = []

        np.random.seed(random_generated_int)
        random.seed(random_generated_int)
        pixel_obs = model.env.reset(random_generated_int)
        agent = WallAvoidingAgent(model.env)

        # random policy
        # more diverse random policy, works slightly better:

        for frame in range(MAX_FRAMES):
            action = agent.get_action(pixel_obs)
            recording_obs.append(pixel_obs)
            recording_action.append(action)
            pixel_obs, reward, done, info = model.env.step(action)

            if done:
                break

        total_frames += frame
        print("dead at", frame, "total recorded frames for this worker", total_frames)
        recording_obs = np.array(recording_obs, dtype=np.uint8)
        recording_action = np.array(recording_action, dtype=np.float16)
        if (len(recording_obs) > MIN_LENGTH):
            np.savez_compressed(filename.format(model.env.lab_name), obs=recording_obs, action=recording_action)
    except Exception as e:
        print("environment error, remaking")
        model.env.remake()
        continue
