from model_metrics import TransferMetrics
from model_vision import BetaVAE
from model import DataSet
import numpy as np
import os



data_dir = "train_record/stagem"
test_dir = "train_record/stagemt"
model_save_dir = "train_beta_vae"
metric_save_dir = "train_bvae_metrics"

result = {}
if os.path.exists(os.path.join(metric_save_dir, "result.npz")):
    result = np.load(os.path.join(metric_save_dir, "result.npz"))['dict'].item()

if not os.path.exists(metric_save_dir):
    os.makedirs(metric_save_dir)

np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)
# Hyperparameters for BetaVAE
z_size = 64
batch_size = 100
learning_rate = 0.0001
kl_tolerance = 0.5

NUM_EPOCH=2

vae = BetaVAE(z_size=z_size, batch_size=batch_size, learning_rate=learning_rate, kl_tolerance=kl_tolerance)
metric = TransferMetrics(z_size=z_size, batch_size=batch_size)
metric.close_sess()
dataset = DataSet(data_dir, batch_size, div=100)
dataset_test = DataSet(test_dir, batch_size, div=100)


model_list = os.listdir(model_save_dir)
for model_path in model_list:
    if model_path in result.keys():
        continue

    # load
    metric._init_session()
    vae.load_json(model_path)
    train_step = 0

    # train
    for epoch in range(NUM_EPOCH):
        # np.random.shuffle(dataset)
        dataset.load_new_file_batch(new_epoch=True)
        while not dataset.is_end():
            # batch = dataset[idx * batch_size:(idx + 1) * batch_size]

            batch = dataset.next_batch()

            obs = batch[0].astype(np.float) / 255.0

            code = vae.sess.run(vae.z, {vae.x: obs})
            yo = np.array([d['A'] for d in batch[1]], np.int)
            yw = np.array([d['Wall'] for d in batch[1]], np.int)

            (obj_loss, wal_loss, obj_train, wal_train, acc_obj, acc_wal, acc_joint) = metric.sess.run([
                metric.obj_loss, metric.wal_loss, metric.obj_train_op, metric.wal_train_op, metric.acc_obj, metric.acc_wal, metric.acc_joint],
                {metric.x: code, metric.yo: yo, metric.yw: yw})

            train_step += 1
            if ((train_step + 1) % 500 == 0):
                print("step", (train_step + 1), obj_loss, acc_obj, wal_loss, acc_wal, acc_joint)
    # test
    dataset_test.load_new_file_batch(new_epoch=True)
    accuracy_obj = []
    accuracy_wal = []
    accuracy_joint = []
    losses_obj = []
    losses_wal = []
    while not dataset_test.is_end():
        batch = dataset_test.next_batch()

        obs = batch[0].astype(np.float) / 255.0

        code = vae.sess.run(vae.z, {vae.x: obs})
        yo = np.array([d['A'] for d in batch[1]], np.int)
        yw = np.array([d['Wall'] for d in batch[1]], np.int)

        (obj_loss, wal_loss, obj_train, wal_train, acc_obj, acc_wal, acc_joint) = metric.sess.run([
            metric.obj_loss, metric.wal_loss, metric.obj_train_op, metric.wal_train_op, metric.acc_obj, metric.acc_wal,
            metric.acc_joint],
            {metric.x: code, metric.yo: yo, metric.yw: yw})
        losses_obj.append(obj_loss)
        losses_wal.append(wal_loss)
        accuracy_obj.append(acc_obj)
        accuracy_wal.append(acc_wal)
        accuracy_joint.append(acc_joint)


    # store
    result[model_path] = [np.mean(losses_obj), np.mean(accuracy_obj), np.mean(losses_wal),
                          np.mean(accuracy_wal), np.mean(accuracy_joint)]

    np.savez(os.path.join(metric_save_dir, "result.npz"), dict=result)

    print("test result", model_path, result[model_path])
