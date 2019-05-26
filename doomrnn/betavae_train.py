'''
Train VAE model on data created using extract.py
final model saved into tf_vae/vae.json
'''

import argparse

parser = argparse.ArgumentParser(description='The ID (integer) of the model')
parser.add_argument('--int', type=int, default=0, help='an integer default: 0')
parser.add_argument('--beta', type=float, default=10.0, help='a float default: 10')
args = parser.parse_args()

model_save_dir = "tf_beta_vae"
model_save_path = "{}/b{}_{}.json".format(model_save_dir, args.beta, args.int)

import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0" # can just override for multi-gpu systems

import tensorflow as tf
import random
import numpy as np

np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)

from doomrnn import reset_graph, ConvVAE

# Hyperparameters for ConvVAE
z_size = 64
batch_size = 100
learning_rate = 0.0001
kl_tolerance = 0.5

# Parameters for training
NUM_EPOCH = 10
DATA_DIR = "record"

if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)


def load_raw_data_list(filelist):
    data_list = []
    counter = 0
    for i in range(len(filelist)):
        filename = filelist[i]
        raw_data = np.load(os.path.join(DATA_DIR, filename))['obs']
        data_list.append(raw_data)
        if ((i + 1) % 1000 == 0):
            print("loading file", (i + 1))
    return data_list


def count_length_of_raw_data(raw_data_list):
    min_len = 100000
    max_len = 0
    N = len(raw_data_list)
    total_length = 0
    for i in range(N):
        l = len(raw_data_list[i])
        if l > max_len:
            max_len = l
        if l < min_len:
            min_len = l
        if l < 10:
            print(i)
        total_length += l
    return total_length


def create_dataset(raw_data_list):
    N = len(raw_data_list)
    M = count_length_of_raw_data(raw_data_list)
    data = np.zeros((M, 64, 64, 3), dtype=np.uint8)
    idx = 0
    for i in range(N):
        raw_data = raw_data_list[i]
        l = len(raw_data)
        if (idx + l) > M:
            data = data[0:idx]
            break
        data[idx:idx + l] = raw_data
        idx += l
    return data


class Dataset(object):
    def __init__(self, DATA_DIR, batch_size, div=10):
        self.data_dir = DATA_DIR
        self.batch_size = batch_size

        self.div = div
        self.file_batch_count = div - 1
        self.filename_batch_list = []

        self.file_batch_dataset = None
        self.num_batches = 0
        self.batch_count = 0

        # load filename_batch_list
        filelist = os.listdir(DATA_DIR)
        # filelist.sort()
        filelist = filelist[0:10000]
        np.random.shuffle(filelist)
        file_batch_size = len(filelist) // div
        self.filename_batch_list = [filelist[i * file_batch_size: (i + 1) * file_batch_size] for i in range(div)]
        # Call the following method for new epoch (including the first one)
        # self.load_new_file_batch(new_epoch=True)

    def load_new_file_batch(self, new_epoch=False):
        if self.is_end() and not new_epoch:
            raise ValueError("epoch ended!")

        self.file_batch_count = 0 if new_epoch else self.file_batch_count + 1
        self.file_batch_dataset = create_dataset(
            load_raw_data_list(self.filename_batch_list[self.file_batch_count]))
        self.batch_count = 0
        self.num_batches = len(self.file_batch_dataset) // self.batch_size
        print("num_batches", self.num_batches)

    def is_end(self):
        if self.file_batch_count >= self.div - 1 and self.batch_count >= self.num_batches:
            return True
        else:
            return False

    def next_batch(self):
        if self.is_end():
            raise ValueError("epoch ended!")
        if self.batch_count >= self.num_batches:
            self.load_new_file_batch()
        batch = self.file_batch_dataset[self.batch_count * self.batch_size:(self.batch_count + 1) * self.batch_size]
        self.batch_count += 1
        return batch


# dataset class
dataset = Dataset(DATA_DIR, batch_size, div=10)
# load dataset from record/*. only use first 10K, sorted by filename.
# filelist = os.listdir(DATA_DIR)
# filelist.sort()
# filelist = filelist[0:10000]
# dataset = load_raw_data_list(filelist)
# dataset = create_dataset(dataset)

# split into batches:
#total_length = len(dataset)
#num_batches = int(np.floor(total_length / batch_size))
#print("num_batches", num_batches)

reset_graph()

vae = ConvVAE(z_size=z_size,
              batch_size=batch_size,
              learning_rate=learning_rate,
              kl_tolerance=kl_tolerance,
              is_training=True,
              reuse=False,
              gpu_mode=True,
              beta=args.beta)

# train loop:
print("train", "step", "loss", "recon_loss", "kl_loss")
for epoch in range(NUM_EPOCH):
    #np.random.shuffle(dataset)
    dataset.load_new_file_batch(new_epoch=True)
    #for idx in range(num_batches):
    while not dataset.is_end():
        #batch = dataset[idx * batch_size:(idx + 1) * batch_size]
        batch = dataset.next_batch()

        obs = batch.astype(np.float) / 255.0

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