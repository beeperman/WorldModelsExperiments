'''
Uses pretrained VAE to process dataset to get mu and logvar for each frame, and stores
all the dataset files into one dataset called series/series.npz
'''

import argparse

parser = argparse.ArgumentParser(description='model loading parameters. the series will be saved with the same name if modeldir is not tf_vae')
parser.add_argument('--modeldir', type=str, default="tf_vae", help='directory to load the vae model')
parser.add_argument('--name', type=str, default="vae", help='name of the json file e.g. load name.json under model dir')
args = parser.parse_args()

import numpy as np
import os
import json
import tensorflow as tf
np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)

import random
from doomrnn import reset_graph, ConvVAE

DATA_DIR = "record"
SERIES_DIR = "series"
model_path_name = args.modeldir
save_name = "series" if "tf_vae" == args.modeldir else args.name

os.environ["CUDA_VISIBLE_DEVICES"]="-1" # disable GPU

if not os.path.exists(SERIES_DIR):
  os.makedirs(SERIES_DIR)

def load_raw_data_list(filelist):
  data_list = []
  action_list = []
  counter = 0
  for i in range(len(filelist)):
    filename = filelist[i]
    raw_data = np.load(os.path.join(DATA_DIR, filename))
    data_list.append(raw_data['obs'])
    action_list.append(raw_data['action'])
    if ((i+1) % 1000 == 0):
      print("loading file", (i+1))
  return data_list, action_list

def encode(img):
  simple_obs = np.copy(img).astype(np.float)/255.0
  simple_obs = simple_obs.reshape(1, 64, 64, 3)
  mu, logvar = vae.encode_mu_logvar(simple_obs)
  z = (mu + np.exp(logvar/2.0) * np.random.randn(*logvar.shape))[0]
  return mu[0], logvar[0], z

def decode(z):
  # decode the latent vector
  img = vae.decode(z.reshape(1, 64)) * 255.
  img = np.round(img).astype(np.uint8)
  img = img.reshape(64, 64, 3)
  return img


# Hyperparameters for ConvVAE
z_size=64
batch_size=1
learning_rate=0.0001
kl_tolerance=0.5


class Dataset(object):
  def __init__(self, DATA_DIR, batch_size=100, div=10):
    self.data_dir = DATA_DIR
    #self.batch_size = batch_size

    self.div = div
    self.file_batch_count = div - 1
    self.filename_batch_list = []

    self.deknamefile_batch_dataset = None
    self.file_batch_action_dataset = None
    self.num_batches = 0
    self.batch_count = 0

    # load filename_batch_list
    filelist = os.listdir(DATA_DIR)
    filelist.sort()
    filelist = filelist[0:10000]
    #np.random.shuffle(filelist)
    file_batch_size = len(filelist) // div
    assert file_batch_size * div == len(filelist), "not divisible"
    self.filename_batch_list = [filelist[i * file_batch_size: (i + 1) * file_batch_size] for i in range(div)]
    # Call the following method for new epoch (including the first one)
    # self.load_new_file_batch(new_epoch=True)

  def load_new_file_batch(self, new_epoch=False):
    if self.is_end() and not new_epoch:
      raise ValueError("epoch ended!")

    self.file_batch_count = 0 if new_epoch else self.file_batch_count + 1
    self.file_batch_dataset, self.file_batch_action_dataset = load_raw_data_list(self.filename_batch_list[self.file_batch_count])
    self.batch_count = 0
    self.num_batches = len(self.file_batch_dataset)
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
    batch = self.file_batch_dataset[self.batch_count]
    batch_action = self.file_batch_action_dataset[self.batch_count]
    self.batch_count += 1
    return batch, batch_action

dataset = Dataset(DATA_DIR, div=100)
dataset.load_new_file_batch(new_epoch=True)
#filelist = os.listdir(DATA_DIR)
#filelist.sort()
#filelist = filelist[0:10000]

#dataset, action_dataset = load_raw_data_list(filelist)

reset_graph()

vae = ConvVAE(z_size=z_size,
              batch_size=batch_size,
              learning_rate=learning_rate,
              kl_tolerance=kl_tolerance,
              is_training=False,
              reuse=False,
              gpu_mode=False)

vae.load_json(os.path.join(model_path_name, "{}.json".format(args.name)))

mu_dataset = []
logvar_dataset = []
action_dataset = []
#for i in range(len(dataset)):
i = 0
while not dataset.is_end():
  #data = dataset[i]
  data, action_data = dataset.next_batch()
  datalen = len(data)
  mu_data = []
  logvar_data = []
  for j in range(datalen):
    img = data[j]
    mu, logvar, z = encode(img)
    mu_data.append(mu)
    logvar_data.append(logvar)
  mu_data = np.array(mu_data, dtype=np.float16)
  logvar_data = np.array(logvar_data, dtype=np.float16)
  action_data = np.array(action_data)
  mu_dataset.append(mu_data)
  logvar_dataset.append(logvar_data)
  action_dataset.append(action_data)
  if (i+1) % 100 == 0:
    print(i+1)

#dataset = np.array(dataset)
action_dataset = np.array(action_dataset)
mu_dataset = np.array(mu_dataset)
logvar_dataset = np.array(logvar_dataset)

np.savez_compressed(os.path.join(SERIES_DIR, "{}.npz".format(save_name)), action=action_dataset, mu=mu_dataset, logvar=logvar_dataset)
