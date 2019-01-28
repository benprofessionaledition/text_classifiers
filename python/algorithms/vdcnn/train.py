import logging
import os
import time

import tensorflow as tf

from algorithms.dataset import JSONDataset
from algorithms.vdcnn.model import VDCNN_9, VDCNN_17, VDCNN_29, vdcnn_fn

# Parameters settings
# Data loading params

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

tf.flags.DEFINE_string("training_file", "train.csv", "Path for the training data")
tf.flags.DEFINE_string("alphabet_file", "", "File containing the newline-delimited character alphabet")
tf.flags.DEFINE_float("test_pct", 0.1, "Percentage of data to be used for testing")

tf.flags.DEFINE_string("logdir", os.path.join(os.getcwd(), "logs"), "logdir for summary data")
tf.flags.DEFINE_string("checkpoints", os.path.join(os.getcwd(), "checkpoints"), "checkpoint directory")
tf.flags.DEFINE_string("cachedir", os.path.join(os.getcwd(), "cache"), "cache directory")
tf.flags.DEFINE_string("modeldir", os.path.join(os.getcwd(), "model"), "models directory")
tf.flags.DEFINE_string("name", str(int(time.time())), "run name")
tf.flags.DEFINE_string("mode", "train", "The mode (train, eval, predict)")
tf.flags.DEFINE_string("cross_entropy", "sigmoid", "The cross entropy type, sigmoid or softmax")
tf.flags.DEFINE_integer("num_classes", -1, "The number of classes")

# Model Hyperparameters
tf.flags.DEFINE_string("arch", "vdcnn9", "The architecture of VDCNN to use {vdcnn9, vdcnn17, vdcnn29}")
tf.flags.DEFINE_float("dropout_keep_prob", 1.0, "Dropout keep probability (default: 1.0)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_integer("sequence_max_length", 250, "Sequence Max Length (default: 250)")

# Training Parameters
tf.flags.DEFINE_float("learning_rate", 1e-2, "Starter Learning Rate (default: 1e-2)")
tf.flags.DEFINE_float("lr_decay_rate", .1, "Learning rate decay rate (default: 1e-1)")
tf.flags.DEFINE_integer("lr_decay_steps", 100, "Number of steps before decaying learning rate")
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 50, "Evaluate model on dev set after this many steps (default: 50)")
tf.flags.DEFINE_integer("max_iterations", 80000, "Maximum iterations")

architecture_dict = {"vdcnn9": VDCNN_9,
                     "vdcnn17": VDCNN_17,
                     "vdcnn29": VDCNN_29}

mode_dict = {"train": tf.estimator.ModeKeys.TRAIN,
             "eval": tf.estimator.ModeKeys.EVAL,
             "predict": tf.estimator.ModeKeys.PREDICT}
if __name__ == '__main__':

    FLAGS = tf.flags.FLAGS
    print("Parameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr, value))
    print("")
    maxiter = FLAGS.max_iterations
    mode = mode_dict[FLAGS.mode]
    architecture = architecture_dict[FLAGS.arch]
    checkpointdir = FLAGS.checkpoints
    model_path = FLAGS.modeldir
    sess = tf.Session()
    # load the dataset
    dset = JSONDataset(FLAGS.training_file)

    params = {
        "architecture": FLAGS.arch,
        "dropout_keep_prob": FLAGS.dropout_keep_prob,
        "l2_reg_lambda": FLAGS.l2_reg_lambda,
        "char_dim": FLAGS.sequence_max_length,
        "alphabet_file": FLAGS.alphabet_file,
        "num_oov_buckets": 1,
        "embedding_dim": 16,
        "num_classes": FLAGS.num_classes,
        "learning_rate": FLAGS.learning_rate,
        "lr_decay_steps": FLAGS.lr_decay_steps,
        "lr_decay_rate": FLAGS.lr_decay_rate,
        "spec_hooks": [],
        "cross_entropy": FLAGS.cross_entropy
    }

    config = tf.estimator.RunConfig(model_dir=checkpointdir,
                                    save_summary_steps=100,
                                    save_checkpoints_steps=FLAGS.evaluate_every,
                                    keep_checkpoint_max=3)

    est = tf.estimator.Estimator(model_fn=vdcnn_fn,
                                 params=params,
                                 config=config)

    hooks = []

    # if FLAGS.debug:
        # hooks.append(tf.python.debug.LocalCLIDebugHook())
    # hooks.append(tf.train.LoggingTensorHook(tensors=['IteratorGetNext:0', 'IteratorGetNext:1'], every_n_iter=1))


    if mode == tf.estimator.ModeKeys.TRAIN:
        est.train(dset.input_func, hooks=None)
