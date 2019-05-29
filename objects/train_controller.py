#!/usr/bin/env python3
"""
This is an example to train a task with PPO algorithm.

Here it creates InvertedDoublePendulum using gym. And uses a PPO with 1M
steps.

Results:
    AverageDiscountedReturn: 528.3
    RiseTime: itr 250
"""
import gym
import tensorflow as tf
import numpy as np

import datetime
now = datetime.datetime.now()

import argparse

from model_controller import ControllerEnv

parser = argparse.ArgumentParser(description='train the controller. choose to load rnn or not')
parser.add_argument('--loadrnn', action='store_true', help='load the rnn model')
parser.add_argument('--name', type=str, default="all", help='name of the series file used to train rnn')
parser.add_argument('--int', type=int, default=0, help='the id of the vision model to load an integer default: 0')
parser.add_argument('--seed', type=int, default=0, help='seed used default: 0')
parser.add_argument('--beta', type=float, default=10.0, help='the beta value of the model to load')
args = parser.parse_args()

# load vision model
vision_load_dir = "train_beta_vae"
vision_load_path = "{}/b{}_{}.json".format(vision_load_dir, args.beta, args.int)
# save memory model
memory_load_dir = "train_rnn"
memory_load_path = "{}/b{}_{}.json".format(memory_load_dir, args.beta, args.int) if args.loadrnn else None


import os
import sys
# locate the garage source directory
__garage_path = os.path.abspath("../../garage/src/")
sys.path.append(__garage_path)
os.environ["PYTHONPATH"] = "{}:$PYTHONPATH".format(__garage_path)

from garage.envs import normalize
from garage.experiment import LocalRunner, run_experiment
from garage.tf.algos import PPO
from garage.tf.baselines import GaussianMLPBaseline
from garage.tf.envs import TfEnv
from garage.tf.policies import GaussianMLPPolicy


def run_task(*_, **__):
    config = tf.ConfigProto(intra_op_parallelism_threads=1)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config).as_default():
        with LocalRunner() as runner:
            #env = TfEnv(normalize(gym.make("InvertedDoublePendulum-v2")))
            #env = TfEnv(gym.make("Pendulum-v0"))
            env = TfEnv(ControllerEnv(vae_load=vision_load_path, rnn_load=memory_load_path))

            policy = GaussianMLPPolicy(
                env_spec=env.spec,
                hidden_sizes=(),
                #hidden_sizes=(64, 64),
                hidden_nonlinearity=tf.nn.tanh,
                output_nonlinearity=None,
            )

            baseline = GaussianMLPBaseline(
                env_spec=env.spec,
                regressor_args=dict(
                    #hidden_sizes=(32, 32),
                    hidden_sizes=(32,),
                    use_trust_region=True,
                ),
            )

            algo = PPO(
                env_spec=env.spec,
                policy=policy,
                baseline=baseline,
                max_path_length=100,
                discount=0.9,
                gae_lambda=0.95,
                lr_clip_range=0.2,
                policy_ent_coeff=0.0,
                optimizer_args=dict(
                    batch_size=32,
                    max_epochs=10,
                    learning_rate=0.001
                ),
                plot=False,
            )

            runner.setup(algo, env)

            runner.train(n_epochs=1200, batch_size=2048, plot=False)

experiment_dir = os.path.abspath("train_ctrler_experiment")
experiment_path = os.path.join(experiment_dir, "b{}_{}_{}_s{}_{}".format(args.beta, args.int, args.name, args.seed,
                    "{}-{}-{}_{}:{}:{}".format(now.year, now.month, now.day, now.hour, now.minute, now.second, now.microsecond)))

run_experiment(run_task, snapshot_mode="last", log_dir=experiment_path, seed=args.seed)

#from env import WallAvoidingAgent
#total_frames = 0
#env = ControllerEnv(vae_load=vision_load_path, rnn_load=memory_load_path)
#try:
#    recording_obs = []
#    recording_action = []
#
#    pixel_obs = env.reset()
#    pixel_obs = env.obs
#    agent = WallAvoidingAgent(env.model.env)
#
#    # random policy
#    # more diverse random policy, works slightly better:
#
#    for frame in range(1800):
#        action = agent.get_action(pixel_obs)
#        recording_obs.append(pixel_obs)
#        recording_action.append(action)
#        pixel_obs, reward, done, info = env.step(action)
#        pixel_obs = env.obs
#
#        if done:
#            break
#
#    total_frames += frame
#    print("dead at", frame, "total recorded frames for this worker", total_frames)
#except Exception as e:
#    print("environment error, resetting")
#    env.reset()

