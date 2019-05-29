'''
Uses pretrained VAE to process dataset to get mu and logvar for each frame, and stores
all the dataset files into one dataset called series/series.npz
'''

import argparse
import time

from model_memory import MDNRNN, hps

from model_vision import BetaVAE
from model import DataSet, SeriesDataSet
import numpy as np
import os
import json
import tensorflow as tf
np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)

# Hyperparameters for BetaVAE
z_size=hps.seq_width
batch_size=hps.batch_size
learning_rate=hps.learning_rate
kl_tolerance=0.5

import random
parser = argparse.ArgumentParser(description='process image data and learn mdn_rnn. choose to load processed data or to generate')
parser.add_argument('--load', action='store_true', help='load the series file corresponding to the name or process image data from datadir')
parser.add_argument('--name', type=str, default="all", help='name of the series file to load or save')
parser.add_argument('--datadir', type=str, default="train_record/stageall", help='directory to load image data')
parser.add_argument('--int', type=int, default=0, help='the id of the vision model to load an integer default: 0')
parser.add_argument('--beta', type=float, default=10.0, help='the beta value of the model to load')
args = parser.parse_args()

DATA_DIR = args.datadir
SERIES_DIR = "train_series"
series_save_path = "{}/b{}_{}_{}.npz".format(SERIES_DIR, args.beta, args.int, args.name)
# load vision model
model_load_dir = "train_beta_vae"
model_load_path = "{}/b{}_{}.json".format(model_load_dir, args.beta, args.int)
# save memory model
model_save_dir = "train_rnn"
model_save_path = "{}/b{}_{}.json".format(model_save_dir, args.beta, args.int)

#os.environ["CUDA_VISIBLE_DEVICES"]="-1" # disable GPU

if not os.path.exists(SERIES_DIR):
    os.makedirs(SERIES_DIR)

if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

if args.load:
    series_dataset = SeriesDataSet(batch_size=batch_size, seq_length=hps.max_seq_len, load_path=series_save_path)
else:
    vae = BetaVAE(z_size=z_size, batch_size=batch_size, learning_rate=learning_rate, kl_tolerance=kl_tolerance,
                  beta=args.beta)
    vae.load_json(model_load_path)
    dataset = DataSet(DATA_DIR, batch_size, div=100)
    series_dataset = SeriesDataSet(batch_size=batch_size, seq_length=hps.max_seq_len, dataset=dataset, vae=vae)
    series_dataset.save_to_path(series_save_path)

mdn_rnn = MDNRNN(hps)
start = time.time()

for epoch in range(1, 401):
    series_dataset.load_new_epoch()
    print("epoch {}, number of batches {}".format(epoch, series_dataset.num_batches))
    batch_state = mdn_rnn.sess.run(mdn_rnn.initial_state)

    while not series_dataset.is_end():
        batch_z, info, batch_action, batch_restart = series_dataset.next_batch()
        step = mdn_rnn.sess.run(mdn_rnn.global_step)
        curr_learning_rate = (hps.learning_rate - hps.min_learning_rate) * (
            hps.decay_rate) ** step + hps.min_learning_rate

        feed = {mdn_rnn.batch_z: batch_z,
                mdn_rnn.batch_action: batch_action,
                mdn_rnn.batch_restart: batch_restart,
                mdn_rnn.initial_state: batch_state,
                mdn_rnn.lr: curr_learning_rate}

        (train_cost, z_cost, batch_state, train_step, _) = mdn_rnn.sess.run(
            [mdn_rnn.cost, mdn_rnn.z_cost, mdn_rnn.final_state, mdn_rnn.global_step, mdn_rnn.train_op], feed)
        if (step % 20 == 0 and step > 0):
            end = time.time()
            time_taken = end - start
            start = time.time()
            r_cost=0.0
            output_log = "step: %d, lr: %.6f, cost: %.4f, z_cost: %.4f, r_cost: %.4f, train_time_taken: %.4f" % (
            step, curr_learning_rate, train_cost, z_cost, r_cost, time_taken)
            print(output_log)

mdn_rnn.save_json(model_save_path)
