import tensorflow as tf
import numpy as np

from model import TensorFlowModel
from collections import namedtuple

# rnn hyper parameters
HyperParams = namedtuple('HyperParams', ['max_seq_len',
                                         'seq_width',
                                         'rnn_size',
                                         'batch_size',
                                         'grad_clip',
                                         'num_mixture',
                                         'restart_factor',
                                         'learning_rate',
                                         'decay_rate',
                                         'min_learning_rate',
                                         'use_layer_norm',
                                         'use_recurrent_dropout',
                                         'recurrent_dropout_prob',
                                         'use_input_dropout',
                                         'input_dropout_prob',
                                         'use_output_dropout',
                                         'output_dropout_prob',
                                         'is_training',
                                         ])
hps = HyperParams(max_seq_len=500, # train on sequences of 500 (found it worked better than 1000)
                     seq_width=64,    # width of our data (64)
                     rnn_size=512,    # number of rnn cells
                     batch_size=100,   # minibatch sizes
                     grad_clip=1.0,
                     num_mixture=5,   # number of mixtures in MDN
                     restart_factor=10.0, # factor of importance for restart=1 rare case for loss.
                     learning_rate=0.001,
                     decay_rate=0.99999,
                     min_learning_rate=0.00001,
                     use_layer_norm=0, # set this to 1 to get more stable results (less chance of NaN), but slower
                     use_recurrent_dropout=0,
                     recurrent_dropout_prob=0.90,
                     use_input_dropout=0,
                     input_dropout_prob=0.90,
                     use_output_dropout=0,
                     output_dropout_prob=0.90,
                     is_training=1)

class MDNRNN(TensorFlowModel):
    def __init__(self, hps, reuse=False, gpu_mode=True):
        self.hps = hps

        self.rnn_state = None
        super().__init__('mdn_rnn', reuse, gpu_mode)

    def _build_graph(self):
        self.num_mixture = self.hps.num_mixture
        KMIX = self.num_mixture  # 5 mixtures
        WIDTH = self.hps.seq_width  # 64 channels
        LENGTH = self.hps.max_seq_len - 1  # 999 timesteps

        if self.hps.is_training:
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

        cell_fn = tf.contrib.rnn.LayerNormBasicLSTMCell  # use LayerNormLSTM

        use_recurrent_dropout = False if self.hps.use_recurrent_dropout == 0 else True
        use_input_dropout = False if self.hps.use_input_dropout == 0 else True
        use_output_dropout = False if self.hps.use_output_dropout == 0 else True
        is_training = False if self.hps.is_training == 0 else True
        use_layer_norm = False if self.hps.use_layer_norm == 0 else True

        if use_recurrent_dropout:
            cell = cell_fn(self.hps.rnn_size, layer_norm=use_layer_norm,
                           dropout_keep_prob=self.hps.recurrent_dropout_prob)
        else:
            cell = cell_fn(self.hps.rnn_size, layer_norm=use_layer_norm)

        # multi-layer, and dropout:
        print("input dropout mode =", use_input_dropout)
        print("output dropout mode =", use_output_dropout)
        print("recurrent dropout mode =", use_recurrent_dropout)
        if use_input_dropout:
            print("applying dropout to input with keep_prob =", self.hps.input_dropout_prob)
            cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.hps.input_dropout_prob)
        if use_output_dropout:
            print("applying dropout to output with keep_prob =", self.hps.output_dropout_prob)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.hps.output_dropout_prob)
        self.cell = cell

        self.sequence_lengths = LENGTH  # assume every sample has same length.
        self.batch_z = tf.placeholder(dtype=tf.float32, shape=[self.hps.batch_size, self.hps.max_seq_len, WIDTH])
        self.batch_action = tf.placeholder(dtype=tf.float32, shape=[self.hps.batch_size, self.hps.max_seq_len])
        self.batch_restart = tf.placeholder(dtype=tf.float32, shape=[self.hps.batch_size, self.hps.max_seq_len])

        self.input_z = self.batch_z[:, :LENGTH, :]
        self.input_action = self.batch_action[:, :LENGTH]
        self.input_restart = self.batch_restart[:, :LENGTH]

        self.target_z = self.batch_z[:, 1:, :]
        self.target_restart = self.batch_restart[:, 1:]

        self.input_seq = tf.concat([self.input_z,
                                    tf.reshape(self.input_action, [self.hps.batch_size, LENGTH, 1]),
                                    tf.reshape(self.input_restart, [self.hps.batch_size, LENGTH, 1])], axis=2)

        self.zero_state = cell.zero_state(batch_size=self.hps.batch_size, dtype=tf.float32)
        self.initial_state = self.zero_state

        inputs = tf.unstack(self.input_seq, axis=1)

        def custom_rnn_autodecoder(decoder_inputs, input_restart, initial_state, cell, scope=None):
            # customized rnn_decoder for the task of dealing with restart
            with tf.variable_scope(scope or "RNN"):
                state = initial_state
                zero_c, zero_h = self.zero_state
                outputs = []
                prev = None

                for i in range(LENGTH):
                    inp = decoder_inputs[i]
                    if i > 0:
                        tf.get_variable_scope().reuse_variables()

                    # if restart is 1, then set lstm state to zero
                    restart_flag = tf.greater(input_restart[:, i], 0.5)

                    c, h = state

                    c = tf.where(restart_flag, zero_c, c)
                    h = tf.where(restart_flag, zero_h, h)

                    output, state = cell(inp, tf.nn.rnn_cell.LSTMStateTuple(c, h))
                    outputs.append(output)

            return outputs, state

        outputs, final_state = custom_rnn_autodecoder(inputs, self.input_restart, self.initial_state, self.cell)
        output = tf.reshape(tf.concat(outputs, axis=1), [-1, self.hps.rnn_size])

        NOUT = WIDTH * KMIX * 3 + 1  # plus 1 to predict the restart state.

        with tf.variable_scope('RNN'):
            output_w = tf.get_variable("output_w", [self.hps.rnn_size, NOUT])
            output_b = tf.get_variable("output_b", [NOUT])

        output = tf.reshape(output, [-1, self.hps.rnn_size])
        output = tf.nn.xw_plus_b(output, output_w, output_b)

        self.out_restart_logits = output[:, 0]
        output = output[:, 1:]

        output = tf.reshape(output, [-1, KMIX * 3])
        self.final_state = final_state

        logSqrtTwoPI = np.log(np.sqrt(2.0 * np.pi))

        def tf_lognormal(y, mean, logstd):
            return -0.5 * ((y - mean) / tf.exp(logstd)) ** 2 - logstd - logSqrtTwoPI

        def get_lossfunc(logmix, mean, logstd, y):
            v = logmix + tf_lognormal(y, mean, logstd)
            v = tf.reduce_logsumexp(v, 1, keepdims=True)
            return -tf.reduce_mean(v)

        def get_mdn_coef(output):
            logmix, mean, logstd = tf.split(output, 3, 1)
            logmix = logmix - tf.reduce_logsumexp(logmix, 1, keepdims=True)
            return logmix, mean, logstd

        out_logmix, out_mean, out_logstd = get_mdn_coef(output)

        self.out_logmix = out_logmix
        self.out_mean = out_mean
        self.out_logstd = out_logstd

        # reshape target data so that it is compatible with prediction shape
        flat_target_data = tf.reshape(self.target_z, [-1, 1])

        lossfunc = get_lossfunc(out_logmix, out_mean, out_logstd, flat_target_data)

        self.z_cost = tf.reduce_mean(lossfunc)

        flat_target_restart = tf.reshape(self.target_restart, [-1, 1])

        #self.r_cost = tf.nn.sigmoid_cross_entropy_with_logits(labels=flat_target_restart,
        #                                                      logits=tf.reshape(self.out_restart_logits, [-1, 1]))

        #factor = tf.ones_like(self.r_cost) + flat_target_restart * (self.hps.restart_factor - 1.0)

        #self.r_cost = tf.reduce_mean(tf.multiply(factor, self.r_cost))

        self.cost = self.z_cost# + self.r_cost

        if self.hps.is_training == 1:
            self.lr = tf.Variable(self.hps.learning_rate, trainable=False)
            optimizer = tf.train.AdamOptimizer(self.lr)

            gvs = optimizer.compute_gradients(self.cost)
            capped_gvs = [(tf.clip_by_value(grad, -self.hps.grad_clip, self.hps.grad_clip), var) for grad, var in
                          gvs]
            self.train_op = optimizer.apply_gradients(capped_gvs, global_step=self.global_step, name='train_step')

    def predict(self, z, action, rnn_state=None, restart=False, temperature=1.0):
        if rnn_state:
            self.rnn_state = rnn_state
        if not self.rnn_state or restart:
            self.rnn_state = self.sess.run(self.zero_state)

        prev_z = np.zeros((1, 1, self.hps.seq_width))
        prev_z[0][0] = z

        prev_action = np.zeros((1, 1))
        prev_action[0] = action

        prev_restart = np.zeros((1, 1))

        s_model = self

        feed = {s_model.input_z: prev_z,
                s_model.input_action: prev_action,
                s_model.input_restart: prev_restart,
                s_model.initial_state: self.rnn_state
                }

        [logmix, mean, logstd, next_state] = s_model.sess.run([s_model.out_logmix,
                                                                           s_model.out_mean,
                                                                           s_model.out_logstd,
                                                                           s_model.final_state],
                                                                          feed)

        self.rnn_state = next_state
        #OUTWIDTH = self.outwidth
        OUTWIDTH = self.hps.seq_width

        # adjust temperatures
        logmix2 = np.copy(logmix) / temperature
        logmix2 -= logmix2.max()
        logmix2 = np.exp(logmix2)
        logmix2 /= logmix2.sum(axis=1).reshape(OUTWIDTH, 1)

        mixture_idx = np.zeros(OUTWIDTH)
        chosen_mean = np.zeros(OUTWIDTH)
        chosen_logstd = np.zeros(OUTWIDTH)
        for j in range(OUTWIDTH):
            idx = self.get_pi_idx(np.random.rand(), logmix2[j])
            mixture_idx[j] = idx
            chosen_mean[j] = mean[j][idx]
            chosen_logstd[j] = logstd[j][idx]

        rand_gaussian = np.random.randn(OUTWIDTH) * np.sqrt(temperature)
        next_z = chosen_mean + np.exp(chosen_logstd) * rand_gaussian

        return next_z

    @staticmethod
    def get_pi_idx(x, pdf):
        # samples from a categorial distribution
        N = pdf.size
        accumulate = 0
        for i in range(0, N):
            accumulate += pdf[i]
            if (accumulate >= x):
                return i
        random_value = np.random.randint(N)
        # print('error with sampling ensemble, returning random', random_value)
        return random_value