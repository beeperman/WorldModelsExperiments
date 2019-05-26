import tensorflow as tf

from model import TensorFlowModel

# TODO: test two softmax classifiers
class TransferMetrics(TensorFlowModel):
    def __init__(self, z_size=64, batch_size=100, learning_rate=0.0001, reuse=False, gpu_mode=True):
        self.z_size = z_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        super().__init__('metrics', reuse, gpu_mode)


    def _build_graph(self):
        self.x = tf.placeholder(tf.float32, shape=[None, 64])
        self.yo_label = tf.placeholder(tf.float32, shape=[None])
        self.yw_label = tf.placeholder(tf.float32, shape=[None])

        self.yo = tf.layers.dense(self.x, 4, activation=tf.identity)
        self.yw = tf.layers.dense(self.x, 2, activation=tf.identity)

        self.obj_loss = tf.losses.softmax_cross_entropy(tf.one_hot(self.yo_label, 4), self.yo, reduction=tf.losses.Reduction.MEAN)
        self.wal_loss = tf.losses.softmax_cross_entropy(tf.one_hot(self.yw_label, 2), self.yw, reduction=tf.losses.Reduction.MEAN)

        self.obj_train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.obj_loss)
        self.wal_train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.wal_loss)

        self.pred_obj = tf.argmax(self.yo, -1)
        self.pred_wal = tf.argmax(self.yw, -1)

        self.correctness_obj = tf.cast(tf.equal(self.yo_label, self.pred_obj), tf.float32)
        self.correctness_wal = tf.cast(tf.equal(self.yw_label, self.pred_wal), tf.float32)

        self.acc_obj = tf.reduce_mean(self.correctness_obj)
        self.acc_wal = tf.reduce_mean(self.correctness_wal)
        self.acc_joint = tf.reduce_mean(self.correctness_obj * self.correctness_wal)

    def get_accuracy(self, x, yo, yw):
        return self.sess.run([self.acc_obj, self.acc_wal, self.acc_joint], feed_dict={self.x: x, self.yo_label: yo, self.yw_label: yw})
