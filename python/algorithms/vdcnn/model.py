"""
Model file
"""
import tensorflow as tf
import numpy as np

# constants
VDCNN_9 = [2, 2, 2, 2]
VDCNN_17 = [4, 4, 4, 4]
VDCNN_29 = [10, 10, 4, 4]

def vdcnn_fn(features: tf.Tensor, labels: tf.Tensor, mode: tf.estimator.ModeKeys, params: dict):
    """
    A function suitable for use with tf.Estimator
    :param features:
    :param labels:
    :param mode:
    :param params:
    :return:
    """

    arch_ = params.get("architecture", VDCNN_9)
    window_size_ = params.get("char_dim", 256)
    batch_size_ = params.get("batch_size", 32)
    char_alphabet_file_ = params.get("alphabet_file")
    num_oov_ = params.get("num_oov_buckets") # out of vocabulary buckets
    char_embedding_dim_ = params.get("embedding_dim", 16)
    filter_multiplier_ = params.get("filter_multiplier", 64)
    dropout_keep_prob_ = params.get("dropout_keep_prob", .5)
    l2_reg_lambda_ = params.get("l2_reg_lambda", 0)
    num_classes_ = params.get("num_classes")
    cross_entropy_ = params.get("cross_entropy", "softmax")
    learning_rate_ = params.get("learning_rate", .1)
    lr_decay_steps_ = params.get("lr_decay_steps", 10000)
    lr_decay_rate_ = params.get("lr_decay_rate", .9)
    spec_hooks_ = params.get("spec_hooks", [])

    # initialize some random bits and stuff
    he_initializer = tf.contrib.keras.initializers.he_normal()
    l2_loss = tf.constant(0.0)
    if labels is None:
        labels = tf.placeholder(tf.float64, [num_classes_])
    input_y = tf.reshape(labels, [-1, num_classes_])
    # define the alphabet
    with open(char_alphabet_file_, 'rb') as charf:
        num_chars = sum(1 for _ in charf) + num_oov_
    char_table = tf.contrib.lookup.index_table_from_file(char_alphabet_file_, num_oov_)
    dropout_keep_prob = tf.constant(dropout_keep_prob_, name="dropout_keep_prob")
    is_training = tf.constant(mode == tf.estimator.ModeKeys.TRAIN, dtype=tf.bool)

    predictions = tf.placeholder(tf.float32, [num_classes_], name="predictions")


    with tf.device("/cpu:0"), tf.variable_scope("char_embeddings"):
        char_ids = char_table.lookup(features)
        embedding_W = tf.Variable(tf.random_uniform([num_chars + 1, char_embedding_dim_], -1.0, 1.0),
                                  name="embedding_W")
        char_embeddings = tf.nn.embedding_lookup(embedding_W, char_ids)
        embedded_characters_expanded = tf.expand_dims(char_embeddings, -1, name="embedding_input")

    # First Conv Layer
    with tf.variable_scope("convolutions"):
        with tf.variable_scope("first_convolution"):
            filter_shape = [3, char_embedding_dim_, 1, filter_multiplier_]
            w = tf.get_variable(name='W', shape=filter_shape,
                                initializer=he_initializer, trainable=is_training)
            conv = tf.nn.conv2d(embedded_characters_expanded, w, strides=[1, 1, char_embedding_dim_, 1], padding="SAME")
            b = tf.get_variable(name='b', shape=[filter_multiplier_],
                                initializer=tf.constant_initializer(0.0), trainable=is_training)
            out = tf.nn.bias_add(conv, b)
            first_conv = tf.nn.relu(out)

        def __convolutional_block(inputs, num_layers, num_filters, name, is_training):
            """
            A convolutional block which will be initialized with varying parameters a few times in the network
            :param inputs: the previous tensor
            :param num_layers: the number of layers
            :param num_filters: the number of filters
            :param name: a name for the tf name scope
            :param is_training: sets the 'is_training' parameter in the batch normalization
            :return:
            """
            with tf.variable_scope("conv_block_%s" % name):
                out = inputs
                for i in range(0, num_layers):
                    filter_shape = [3, 1, out.get_shape()[3], num_filters]
                    w = tf.get_variable(name='W_' + str(i), shape=filter_shape,
                                        initializer=he_initializer, trainable=is_training)
                    b = tf.get_variable(name='b_' + str(i), shape=[num_filters],
                                        initializer=tf.constant_initializer(0.0), trainable=is_training)
                    conv = tf.nn.conv2d(out, w, strides=[1, 1, 1, 1], padding="SAME")
                    conv = tf.nn.bias_add(conv, b)
                    batch_norm = tf.contrib.layers.batch_norm(conv, center=True, scale=True, is_training=is_training)
                    out = tf.nn.relu(batch_norm)
            return out

        # all convolutional blocks
        conv_in = first_conv
        for i in range(4):
            conv_block = __convolutional_block(conv_in, num_layers=arch_[i], num_filters=filter_multiplier_ * (2**i), name=('conv_%d' % (i + 1)), is_training=is_training)
            pool = tf.nn.max_pool(conv_block, ksize=[1, 3, 1, 1], strides=[1, 2, 1, 1], padding='SAME',
                               name="pool_%d" % (i + 1))
            conv_in = conv_block
        shape = int(np.prod(pool.get_shape()[1:]))
        conv_max_pool_out = tf.reshape(pool, (-1, shape), name="conv_output")

    with tf.variable_scope("dropout"):
        dropout = tf.nn.dropout(conv_max_pool_out, dropout_keep_prob)

    fc3_in = dropout

    # fc3
    with tf.variable_scope('fc_aggregate'):
        w = tf.get_variable('W_fc_aggregate', [fc3_in.get_shape()[1], num_classes_], initializer=he_initializer)
        b = tf.get_variable('b_fc_aggregate', [num_classes_], initializer=tf.constant_initializer(0.0))
        if l2_reg_lambda_ > 0:
            l2_loss += tf.nn.l2_loss(w) + tf.nn.l2_loss(b)
        fc3 = tf.matmul(fc3_in, w) + b

        loss = None  # to shut up the compiler

    def __terminal_block_softmax(block_name="loss"):
        """
        This is the terminal block for a softmax, ostensibly single-class estimator.

        :return: nothing, just modifies the tensorflow graph
            """
        # Calculate mean cross-entropy loss
        with tf.name_scope(block_name):
            nonlocal predictions
            predictions = tf.nn.softmax(fc3, name="predictions")
            cross_entropy_losses = tf.nn.softmax_cross_entropy_with_logits(logits=fc3, labels=input_y)
            nonlocal loss
            loss = tf.reduce_mean(cross_entropy_losses) + l2_reg_lambda_ * l2_loss

    def __terminal_block_sigmoid(block_name="loss"):
        """
        This function represents the terminal block for sigmoid cross-entropy.

        :return: nothing, just modifies the tensorflow graph
        """
        with tf.name_scope(block_name):
            nonlocal predictions
            predictions = tf.sigmoid(fc3, name="predictions")
            cross_entropy_losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=fc3, labels=tf.to_float(input_y))
            nonlocal loss
            loss = tf.reduce_mean(cross_entropy_losses) + l2_reg_lambda_ * l2_loss

    if cross_entropy_ == 'softmax':
        __terminal_block_softmax()
    elif cross_entropy_ == 'sigmoid':
        __terminal_block_sigmoid()
    else:
        raise NotImplementedError("Unrecognized cross entropy: {}\t Available types are sigmoid, softmax")

    with tf.name_scope('metrics'):
        """
        A note on accuracy: since accuracy is essentially the ratio of true positives and negatives to all data points, we can interpret it effectively 
        as the "top-1" accuracy. This means we specifically need the argmax of the final output, NOT the rounded value. While softmax does guarantee that 
        all the values will sum up to 1 so at most only one value in the rounded tensor will be 1, it doesn't guarantee that any values will be greater than 
        0.5. Imagine this case:

        predictions: [0.1, 0.2, 0.2, 0.1, 0.4]
        truth: [0, 0, 0, 0, 1]
        predictions, rounded: [0, 0, 0, 0, 0]
        predictions, argmax (effectively): [0, 0, 0, 0, 1]

        This is a subjective issue that depends on how the algorithm will ultimately be implemented, but at AddThis I opted to use the argmax
        since my unscientific observations seemed to indicate that these results were usually correct. It's usually a moot point, and again,
        the usecase should be kept in mind (it's adtech, a massive scale, and all things equal, quantity is generally better than quality).
        
        For this reason, you should ignore this metric entirely if you're using sigmoid loss--argmax is indeterminate if there are multiple 
        true labels. Furthermore, with large numbers of classes you can expect to have extremely high accuracy; given 100 classes in a single-class 
        problem, you can have 99% accuracy with 0% recall. 
        """
        predictions_rounded = tf.round(predictions)
        # note: accuracy is meaningless for multilabel
        accuracy, accuracy_op = tf.metrics.accuracy(input_y, predictions_rounded)
        precision, precision_op = tf.metrics.precision(input_y, predictions_rounded)
        recall, recall_op = tf.metrics.recall(input_y, predictions_rounded)
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('precision', precision)
        tf.summary.scalar('recall', recall)
        tf.summary.merge_all()

    if mode == tf.estimator.ModeKeys.PREDICT:
        output = tf.estimator.export.ClassificationOutput(scores=predictions)
        spec = tf.estimator.EstimatorSpec(mode=mode,
                                          export_outputs={'predictions':output},
                                          predictions=predictions,
                                          training_hooks=spec_hooks_)
    else:  # TRAIN or EVAL
        # define the training function based on what we've used previously to train VDCNN
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            global_step = tf.train.get_global_step()
            optimizer = tf.train.MomentumOptimizer(learning_rate_, 0.9)
            lr_decay_fn = lambda lr, global_step: tf.train.exponential_decay(lr, global_step, lr_decay_steps_, lr_decay_rate_, staircase=True)
            train_op = tf.contrib.layers.optimize_loss(loss=loss, global_step=global_step, clip_gradients=4.0,
                                                       learning_rate=learning_rate_, optimizer=lambda lr: optimizer,
                                                       update_ops=update_ops, learning_rate_decay_fn=lr_decay_fn)
            train_op = tf.group(accuracy_op, precision_op, recall_op, train_op)

        spec = tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          train_op=train_op,
                                          eval_metric_ops={
                                              'accuracy': (accuracy, accuracy_op),
                                              'precision': (precision, precision_op),
                                              'recall': (recall, recall_op)
                                          },
                                          training_hooks=spec_hooks_)
    return spec









