import os

from env import CollectGoodObjectsEnv
import numpy as np
import tensorflow as tf
import json

STAGE_ENV_NAME = {
    1: ["cake_hat_true", "balloon_can_true", "cake_can_false", "balloon_hat_false"], # unsuperivsed vae learning
    2: ["train"],
    3: ["test"],
    4: ["hat_false", "cake_false", "balloon_true", "can_true"], # metrics train
    5: ["hat_true", "cake_true", "balloon_false", "can_false"]  # metrics test
}

OBJECT_INDEX = {
    "hat": 0,
    "balloon": 1,
    "cake": 2,
    "can": 3,
    "train": 4,
    "test": 5
}

class Model(object):
    """The complete model"""
    def __init__(self,
                 stage=1,
                 initialize=False):
        self.env = None

        self.model_v = None
        self.model_m = None
        self.model_c = None
        self.model_metrics = None


        self.env = self.initialize_env(stage)

    def initialize_env(self, stage):
        return CollectGoodObjectsEnv(name=STAGE_ENV_NAME[stage])

    # TODO: train TransferMetics model and test it under testing environment
    def compute_vae_transfer_metics(self, train_data_dir, test_data_dir):
        # load train
        # process data to match the ph in TransferMetrics model
        # train model till convergence
        # load test
        # process data to match the ph in TransferMetrics model
        # obtain success rate (background, object, combined)
        pass


class DataSet(object):
    def __init__(self, DATA_DIR, batch_size, div=10, shuffle=True, file_size=10000):
        self.data_dir = DATA_DIR
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.div = div
        self.file_batch_count = div - 1
        self.filename_batch_list = []

        self.file_batch_dataset = None
        self.num_batches = 0
        self.batch_count = 0

        # load filename_batch_list
        filelist = os.listdir(DATA_DIR)
        np.random.shuffle(filelist)
        #filelist.sort()
        filelist = filelist[0:file_size]
        file_batch_size = len(filelist) // div
        self.filename_batch_list = [filelist[i * file_batch_size: (i + 1) * file_batch_size] for i in range(div)]
        # Call the following method for new epoch (including the first one)
        #self.load_new_file_batch(new_epoch=True)

    def load_new_file_batch(self, new_epoch=False, suppress=False):
        if self.is_end() and not new_epoch:
            raise ValueError("epoch ended!")

        self.file_batch_count = 0 if new_epoch else self.file_batch_count + 1
        if not (self.file_batch_dataset and self.div == 1): # don't load if div is 1
            self.file_batch_dataset = self.create_dataset(self.load_raw_data_list(self.filename_batch_list[self.file_batch_count]), self.shuffle)
        self.batch_count = 0
        self.num_batches = len(self.file_batch_dataset[0]) // self.batch_size
        if not suppress:
            print("num_batches", self.num_batches)

    def is_end(self):
        if self.file_batch_count >= self.div - 1 and self.batch_count >= self.num_batches:
            return True
        else:
            return False

    # TODO: test include environment info in the batch
    def next_batch(self):
        if self.is_end():
            raise ValueError("epoch ended!")
        if self.batch_count >= self.num_batches:
            self.load_new_file_batch()
        indices = self.file_batch_dataset[0][self.batch_count * self.batch_size:(self.batch_count + 1) * self.batch_size]
        data = self.file_batch_dataset[1][indices]
        info = [self.file_batch_dataset[2][i] for i in indices]
        actions = self.file_batch_dataset[3][indices]
        restart = self.file_batch_dataset[4][indices]
        self.batch_count += 1
        return [data, info, actions, restart]

    def load_raw_data_list(self, filelist):
        data_list = []
        file_info = []
        action_list = []
        # any potential info
        other_info = []
        counter = 0
        for i in range(len(filelist)):
            filename = filelist[i]
            if '.' not in filename or filename.split('.', 2)[1] != 'npz':
                continue
            raw_data = np.load(os.path.join(self.data_dir, filename))
            data_list.append(raw_data['obs'])
            raw_file_info = self.parse_filename(filename)
            file_info.append([raw_file_info for _ in range(len(data_list[-1]))])
            action_list.append(raw_data['action'])
            if ((i + 1) % 1000 == 0):
                print("loading file", (i + 1))
        assert len(data_list) == len(file_info)
        return [data_list, file_info, action_list]


    # TODO: test include environment info in the dataset
    @classmethod
    def create_dataset(cls, raw_data_list, shuffle):
        N = len(raw_data_list[0])
        M = cls.count_length_of_raw_data(raw_data_list[0])
        data = np.zeros((M, 64, 64, 3), dtype=np.uint8)
        info = []
        actions = np.zeros((M), dtype=np.float)
        restart = np.zeros((M), dtype=np.uint8)
        idx = 0
        for i in range(N):
            raw_data = raw_data_list[0][i]
            l = len(raw_data)
            if (idx + l) > M:
                data = data[0:idx]
                actions = actions[0:idx]
                restart = restart[0:idx]
                break
            data[idx:idx + l] = raw_data
            info += raw_data_list[1][i]
            actions[idx:idx + l] = raw_data_list[2][i]
            restart[idx] = 1
            idx += l
        assert len(info) == idx
        permutation = np.arange(idx)
        if shuffle:
            np.random.shuffle(permutation)
        return [permutation, data, info, actions, restart]

    @staticmethod
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

    @staticmethod
    def parse_filename(filename):
        filename_list = filename.split('_')
        dict = {
            'A': OBJECT_INDEX[filename_list[0]],
            'B': OBJECT_INDEX[filename_list[0]],
            'Wall': None,
            'FileListLen': len(filename_list)
        }
        if len(filename_list) < 3:
            return dict

        if filename_list[1] in ["true", "false"]:
            dict["Wall"] = True if filename_list[1] == "true" else False
        else:
            dict["Wall"] = True if filename_list[2] == "true" else False
            dict["B"] = OBJECT_INDEX[filename_list[1]]
        return dict

class SeriesDataSet(object):
    def __init__(self, batch_size=100, seq_length=500, load_path=None, dataset=None, vae=None):
        assert load_path or (dataset and vae)

        self.load_path = load_path
        self.dataset = dataset
        self.vae = vae

        self.batch_size = batch_size
        self.seq_length = seq_length
        self.num_batches = 0
        self.batch_count = 0

        self.mu = None
        self.logvar = None
        self.info = None
        self.actions = None
        self.restart = None

        self.reshaped_mu = None
        self.reshaped_logvar = None
        self.reshaped_info = None
        self.reshaped_actions = None
        self.reshaped_restart = None

        if load_path:
            self.load_from_path(load_path)
        else:
            self.generate(dataset, vae)

        self.reshape()

    def load_new_epoch(self):
        self.batch_count = 0

    def is_end(self):
        return self.batch_count >= self.num_batches

    def next_batch(self):
        i = self.batch_count
        self.batch_count += 1
        batch_z = np.random.normal(loc=self.reshaped_mu[i], scale=np.exp(self.reshaped_logvar[i]/2.0))
        return [batch_z, self.reshaped_info[i], self.reshaped_actions[i], self.reshaped_restart[i]]

    def generate(self, dataset, vae):
        list_mu = []
        list_logvar = []
        list_info = []
        list_actions = []
        list_restart = []

        # collect 1 epoch of data
        batch_size = dataset.batch_size
        assert batch_size == vae.batch_size
        dataset.load_new_file_batch(new_epoch=True)
        while not dataset.is_end():

            batch = dataset.next_batch()

            obs = batch[0].astype(np.float) / 255.0

            mu, logvar = vae.encode_mu_logvar(obs)
            info = batch[1]
            actions = batch[2]
            restart = batch[3]

            list_mu.append(mu)
            list_logvar.append(logvar)
            list_info += info
            list_actions.append(actions)
            list_restart.append(restart)

        # organize the data
        length = len(list_info)
        self.num_batches = length // (self.batch_size * self.seq_length)
        num_datapoints = self.num_batches * self.batch_size * self.seq_length

        self.mu = np.zeros((num_datapoints, np.shape(mu)[-1]), dtype=np.float16)
        self.logvar = np.zeros((num_datapoints, np.shape(logvar)[-1]), dtype=np.float16)
        self.info = list_info[:num_datapoints]
        # discrete action ready
        self.actions = np.zeros((num_datapoints), dtype=actions[0].dtype)
        self.restart = np.zeros((num_datapoints), dtype=np.uint8)

        idx = 0
        for i in range(len(list_mu)):
            l = len(list_mu[i])
            if idx >= num_datapoints:
                break
            if idx + l >= num_datapoints:
                l = num_datapoints - idx

            self.mu[idx:idx+l] = list_mu[i]
            self.logvar[idx:idx+l] = list_logvar[i]
            self.actions[idx:idx+l] = list_actions[i]
            self.restart[idx:idx+l] = list_restart[i]

            idx += l


    def load_from_path(self, load_path):
        data = np.load(load_path, allow_pickle=True)
        self.mu = data['mu']
        self.logvar = data['logvar']
        self.info = data['info']
        self.actions = data['action']
        self.restart = data['restart']

        self.num_batches = len(self.mu) // (self.batch_size * self.seq_length)

    def reshape(self):
        self.reshaped_mu = np.split(self.mu.reshape(self.batch_size, -1, self.mu.shape[-1]), self.num_batches, 1)
        self.reshaped_logvar = np.split(self.logvar.reshape(self.batch_size, -1, self.logvar.shape[-1]), self.num_batches, 1)
        self.reshaped_info = np.split(np.array(self.info).reshape(self.batch_size, -1), self.num_batches, 1)
        self.reshaped_actions = np.split(self.actions.reshape(self.batch_size, -1), self.num_batches, 1)
        self.reshaped_restart = np.split(self.restart.reshape(self.batch_size, -1), self.num_batches, 1)

    def save_to_path(self, save_path):
        np.savez_compressed(save_path, mu=self.mu, logvar=self.logvar, info=self.info, action=self.actions, restart=self.restart)


# TODO: Refactor the code
class TensorFlowModel(object):

    def __init__(self, scope_name, reuse=False, gpu_mode=True):
        self.scope_name = scope_name
        self.json_multiplier = 10000.
        self.init = None # must build in _build_graph
        self.sess = None # will generate in init_session
        self.g = None
        self.reuse = reuse
        self.gpu_mode = gpu_mode

        self.setup_graph_sess()

    def setup_graph_sess(self, seed=None):
        if not self.gpu_mode:
            with tf.device('/cpu:0'):
                tf.logging.info('Model using cpu.')
                self.build_graph(seed)
        else:
            tf.logging.info('Model using gpu.')
            self.build_graph(seed)
        self._init_session()

    def _build_graph(self):
        pass

    def build_graph(self, seed=None):
        self.g = tf.Graph()
        with self.g.as_default():
            if seed:
                tf.set_random_seed(seed)
            with tf.variable_scope(self.scope_name, reuse=self.reuse):
                self._build_graph()

                # initialize vars
                self.init = tf.global_variables_initializer()

                # Create assign opsfor VAE
                t_vars = tf.trainable_variables()
                self.assign_ops = {}
                for var in t_vars:
                    if var.name.startswith(self.scope_name):
                        pshape = var.get_shape()
                        pl = tf.placeholder(tf.float32, pshape, var.name[:-2] + '_placeholder')
                        assign_op = var.assign(pl)
                        self.assign_ops[var] = (assign_op, pl)

    def _init_session(self):
        """Launch TensorFlow session and initialize variables. allow growth"""
        config = tf.ConfigProto(intra_op_parallelism_threads=1)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config, graph=self.g)
        self.sess.run(self.init)

    def close_sess(self):
        """ Close TensorFlow session """
        self.sess.close()

    def get_model_params(self):
        # get trainable params.
        model_names = []
        model_params = []
        model_shapes = []
        with self.g.as_default():
            t_vars = tf.trainable_variables()
            for var in t_vars:
                if var.name.startswith(self.scope_name):
                    param_name = var.name
                    p = self.sess.run(var)
                    model_names.append(param_name)
                    params = np.round(p * self.json_multiplier).astype(np.int).tolist()
                    model_params.append(params)
                    model_shapes.append(p.shape)
        return model_params, model_shapes, model_names

    def get_random_model_params(self, stdev=0.5):
        # get random params.
        _, mshape, _ = self.get_model_params()
        rparam = []
        for s in mshape:
            # rparam.append(np.random.randn(*s)*stdev)
            rparam.append(np.random.standard_cauchy(s) * stdev)  # spice things up!
        return rparam

    def set_model_params(self, params):
        with self.g.as_default():
            t_vars = tf.trainable_variables()
            idx = 0
            for var in t_vars:
                if var.name.startswith(self.scope_name):
                    pshape = tuple(var.get_shape().as_list())
                    p = np.array(params[idx])
                    assert pshape == p.shape, "inconsistent shape"
                    assign_op, pl = self.assign_ops[var]
                    self.sess.run(assign_op, feed_dict={pl.name: p / self.json_multiplier})
                    idx += 1

    def load_json(self, jsonfile):
        with open(jsonfile, 'r') as f:
            params = json.load(f)
        self.set_model_params(params)

    def save_json(self, jsonfile):
        model_params, model_shapes, model_names = self.get_model_params()
        qparams = []
        for p in model_params:
            qparams.append(p)
        with open(jsonfile, 'wt') as outfile:
            json.dump(qparams, outfile, sort_keys=True, indent=0, separators=(',', ': '))

    def set_random_params(self, stdev=0.5):
        rparam = self.get_random_model_params(stdev)
        self.set_model_params(rparam)

    # will not use
    def save_model(self, model_save_path):
        sess = self.sess
        with self.g.as_default():
            saver = tf.train.Saver(tf.global_variables())
        checkpoint_path = os.path.join(model_save_path, 'vae')
        tf.logging.info('saving model %s.', checkpoint_path)
        saver.save(sess, checkpoint_path, 0)  # just keep one

    def load_checkpoint(self, checkpoint_path):
        sess = self.sess
        with self.g.as_default():
            saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(checkpoint_path)
        print('loading model', ckpt.model_checkpoint_path)
        tf.logging.info('Loading model %s.', ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)