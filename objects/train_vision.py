from model_vision import BetaVAE
from model import DataSet
import numpy as np
import os

import argparse

parser = argparse.ArgumentParser(description='The ID (integer) of the model')
parser.add_argument('--int', type=int, default=0, help='an integer default: 0')
parser.add_argument('--beta', type=float, default=10.0, help='a float default: 10')
parser.add_argument('--datadir', type=str, default="train_record", help='directory to load data')
args = parser.parse_args()

model_save_dir = "train_beta_vae"
model_save_path = "{}/b{}_{}.json".format(model_save_dir, args.beta, args.int)

if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)


np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)
# Hyperparameters for BetaVAE
z_size = 64
batch_size = 100
learning_rate = 0.0001
kl_tolerance = 0.0

NUM_EPOCH = 3
DATA_DIR = args.datadir

vae = BetaVAE(z_size=z_size, batch_size=batch_size, learning_rate=learning_rate, kl_tolerance=kl_tolerance, beta=args.beta)
dataset = DataSet(DATA_DIR, batch_size, div=100)

print("train", "step", "loss", "recon_loss", "kl_loss")
for epoch in range(NUM_EPOCH):
    #np.random.shuffle(dataset)
    dataset.load_new_file_batch(new_epoch=True)
    while not dataset.is_end():
        #batch = dataset[idx * batch_size:(idx + 1) * batch_size]

        # TODO: deal with extra info
        batch = dataset.next_batch()

        obs = batch[0].astype(np.float) / 255.0

        feed = {vae.x: obs, }

        (train_loss, r_loss, kl_loss, train_step, _) = vae.sess.run([
            vae.loss, vae.r_loss, vae.kl_loss, vae.global_step, vae.train_op
        ], feed)

        if ((train_step + 1) % 500 == 0):
            print("step", (train_step + 1), train_loss, r_loss, kl_loss)
        if ((train_step + 1) % 30000 == 0):
            vae.save_json(model_save_path)

# finished, final model:
vae.save_json(model_save_path)
