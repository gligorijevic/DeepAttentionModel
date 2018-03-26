import numpy as np
import tensorflow as tf


class WordEmbeddingModel(object):
    """ Deep Attention Model for Text Classification. """
    def __init__(self,
                 session,
                 batch_size=64,
                 embedding_size=256,
                 vocabulary_size=1000,
                 number_of_words=35,
                 weights_init_strategy=tf.contrib.layers.variance_scaling_initializer(),
                 keep_probability=0.5,
                 learning_rate=0.001,
                 number_of_classes=1,
                 phase_train=True):

        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.number_of_words = number_of_words
        self.vocabulary_size = vocabulary_size
        self.weights_init_strategy = weights_init_strategy
        self.weights_init_strategy = tf.contrib.layers.variance_scaling_initializer()
        self.keep_prob = keep_probability
        self.learning_rate = learning_rate
        self.number_of_classes = number_of_classes
        self.phase_train = phase_train

        self._initializer = tf.truncated_normal

        self._build_params()
        self._build_inputs()

        self.inference_op = self.build_inference_ops()
        self.loss_op = self.build_loss_ops()
        self.train_op = self.build_opt_ops()
        self.predict_op = self.build_predict_ops()

        self._init_op = tf.global_variables_initializer()
        self._sess = session
        self._sess.run(self._init_op)

        self._saver = tf.train.Saver()

    def _build_inputs(self):
        self.textPlaceHolder = tf.placeholder(tf.int32, shape=[None, self.number_of_words])
        if self.number_of_classes > 1:
            self.labelPlaceHolder = tf.placeholder(tf.float32, shape=[None, self.number_of_classes])
        else:
            self.labelPlaceHolder = tf.placeholder(tf.float32, shape=[None, ])

    def _build_params(self):
        self.word_embeddings = tf.Variable(self._initializer([self.vocabulary_size, self.embedding_size]), name='W_emb')

    def build_inference_ops(self):
        """
        Forward pass of the model
        """
        print('###: textPlaceHolder:', self.textPlaceHolder.get_shape().as_list())
        a = tf.nn.embedding_lookup(self.word_embeddings, self.textPlaceHolder)
        print('###: textEmbedding1:', a.get_shape().as_list())
        a = tf.reduce_sum(a, axis=2)
        a = tf.layers.dense(inputs=a, units=self.number_of_classes, activation=tf.nn.relu, trainable=True,
                            name='joint_linear_projection')
        print('###: textEmbedding2:', a.get_shape().as_list())

        # --------------------------------------------------------------------------------
        y = tf.squeeze(a)

        return y

    def build_loss_ops(self):
        """
        Loss
        """
        if self.number_of_classes == 1:
            loss_op = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.inference_op, labels=self.labelPlaceHolder))
        else:
            loss_op = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=self.inference_op, labels=self.labelPlaceHolder))
        return loss_op

    def build_opt_ops(self):
        train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss_op)
        return train_op

    def build_predict_ops(self):
        if self.number_of_classes == 1:
            predict_op = tf.nn.sigmoid(self.inference_op)
        else:
            predict_op = tf.nn.softmax(self.inference_op)
        return predict_op

    def batch_fit(self, text_ids, labels):
        feed_dict = {self.textPlaceHolder: text_ids,
                     self.labelPlaceHolder: labels}

        loss, _ = self._sess.run([self.loss_op, self.train_op], feed_dict=feed_dict)

        return loss

    def predict(self, text_ids, labels):
        feed_dict = {self.textPlaceHolder: text_ids,
                     self.labelPlaceHolder: labels}

        loss, predict = self._sess.run([self.loss_op, self.predict_op], feed_dict=feed_dict)

        return loss, predict


class WordEmbeddingModelWide(object):
    """ Deep Attention Model for Text Classification. """
    def __init__(self,
                 session,
                 batch_size=64,
                 embedding_size=256,
                 vocabulary_size=1000,
                 number_of_words=35,
                 total_number_of_features=50,
                 weights_init_strategy=tf.contrib.layers.variance_scaling_initializer(),
                 keep_probability=0.5,
                 learning_rate=0.001,
                 number_of_classes=1,
                 phase_train=True):

        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.number_of_words = number_of_words
        self.vocabulary_size = vocabulary_size
        self.total_number_of_features = total_number_of_features
        self.weights_init_strategy = weights_init_strategy
        self.weights_init_strategy = tf.contrib.layers.variance_scaling_initializer()
        self.keep_prob = keep_probability
        self.learning_rate = learning_rate
        self.number_of_classes = number_of_classes
        self.phase_train = phase_train

        self._initializer = tf.truncated_normal

        self._build_params()
        self._build_inputs()

        self.inference_op = self.build_inference_ops()
        self.loss_op = self.build_loss_ops()
        self.train_op = self.build_opt_ops()
        self.predict_op = self.build_predict_ops()

        self._init_op = tf.global_variables_initializer()
        self._sess = session
        self._sess.run(self._init_op)

        self._saver = tf.train.Saver()

    def _build_inputs(self):
        self.textPlaceHolder = tf.placeholder(tf.int32, shape=[None, self.number_of_words])
        self.binaryFeaturesPlaceHolder = tf.placeholder(tf.float32, shape=[None, self.total_number_of_features])
        if self.number_of_classes > 1:
            self.labelPlaceHolder = tf.placeholder(tf.float32, shape=[None, self.number_of_classes])
        else:
            self.labelPlaceHolder = tf.placeholder(tf.float32, shape=[None, ])


    def _build_params(self):
        self.word_embeddings = tf.Variable(self._initializer([self.vocabulary_size, self.embedding_size]), name='W_emb')

    def build_inference_ops(self):
        """
        Forward pass of the model
        """
        print('###: textPlaceHolder:', self.textPlaceHolder.get_shape().as_list())
        a = tf.nn.embedding_lookup(self.word_embeddings, self.textPlaceHolder)
        print('###: textEmbedding1:', a.get_shape().as_list())
        a = tf.reduce_sum(a, axis=2)
        a = tf.concat([a, self.binaryFeaturesPlaceHolder], 1)
        print('###: textEmbedding2:', a.get_shape().as_list())
        a = tf.layers.dense(inputs=a, units=self.number_of_classes, activation=tf.nn.relu, trainable=True,
                            name='joint_linear_projection')
        print('###: textEmbedding3:', a.get_shape().as_list())

        # --------------------------------------------------------------------------------
        y = tf.squeeze(a)

        return y

    def build_loss_ops(self):
        """
        Loss
        """
        if self.number_of_classes == 1:
            loss_op = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.inference_op, labels=self.labelPlaceHolder))
        else:
            loss_op = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=self.inference_op, labels=self.labelPlaceHolder))
        return loss_op

    def build_opt_ops(self):
        train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss_op)
        return train_op

    def build_predict_ops(self):
        if self.number_of_classes == 1:
            predict_op = tf.nn.sigmoid(self.inference_op)
        else:
            predict_op = tf.nn.softmax(self.inference_op)
        return predict_op

    def batch_fit(self, text_ids, binary_features, labels):
        feed_dict = {self.textPlaceHolder: text_ids,
                     self.binaryFeaturesPlaceHolder: binary_features,
                     self.labelPlaceHolder: labels}

        loss, _ = self._sess.run([self.loss_op, self.train_op], feed_dict=feed_dict)

        return loss

    def predict(self, text_ids, binary_features, labels):
        feed_dict = {self.textPlaceHolder: text_ids,
                     self.binaryFeaturesPlaceHolder: binary_features,
                     self.labelPlaceHolder: labels}

        loss, predict = self._sess.run([self.loss_op, self.predict_op], feed_dict=feed_dict)

        return loss, predict


class VDCNN(object):
    """ Very Deep Convolutional Networks for Text Classification. """
    def __init__(self,
                 session,
                 base_factor=64,
                 batch_size=64,
                 embedding_size=200,
                 vocabulary_size=1000,
                 number_of_words=35,
                 total_number_of_characters=100,
                 weights_init_strategy=tf.contrib.layers.variance_scaling_initializer(),
                 keep_probability=0.5,
                 learning_rate=0.001,
                 number_of_classes=1,
                 phase_train=True):

        self.base_factor = base_factor
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.number_of_words = number_of_words
        self.vocabulary_size = vocabulary_size
        self.total_number_of_characters = total_number_of_characters
        self.weights_init_strategy = weights_init_strategy
        self.weights_init_strategy = tf.contrib.layers.variance_scaling_initializer()
        self.keep_prob = keep_probability
        self.learning_rate = learning_rate
        self.number_of_classes = number_of_classes
        self.phase_train = phase_train

        # self._initializer = tf.random_normal_initializer(stddev=0.1)
        self._initializer = tf.truncated_normal
        # if options['initializer'] == 0 else tf.random_uniform_initializer(minval=-0.01, maxval=0.01)

        self._build_params()
        self._build_inputs()

        self.inference_op = self.build_inference_ops()
        self.loss_op = self.build_loss_ops()
        self.train_op = self.build_opt_ops()
        self.predict_op = self.build_predict_ops()

        self._init_op = tf.global_variables_initializer()
        self._sess = session
        self._sess.run(self._init_op)

        self._saver = tf.train.Saver()

    def _convBlock(self, x, filterSize, numberOfFilters, numOfLayers, tra, weights_init_strategy):
        net = x
        for i in range(numOfLayers):
            net = tf.layers.conv1d(inputs=net, filters=numberOfFilters, kernel_size=filterSize, strides=1,
                                   padding='same', activation=None, kernel_initializer=weights_init_strategy,
                                   trainable=tra)

            net = tf.layers.batch_normalization(inputs=net, trainable=tra)

            net = tf.nn.relu(net)

        return net

    def _resBlock(self, x, convBlock, poolBlock=None):
        """ Residual block """
        # TODO: seems that this operation prevents the model to optimize
        if poolBlock:
            return x  # TODO: figure this residual part out
        else:
            return tf.add(x, convBlock)

    def _build_inputs(self):
        self.textPlaceHolder = tf.placeholder(tf.int32, shape=[None, self.number_of_words])
        if self.number_of_classes > 1:
            self.labelPlaceHolder = tf.placeholder(tf.float32, shape=[None, self.number_of_classes])
        else:
            self.labelPlaceHolder = tf.placeholder(tf.float32, shape=[None, ])

    def _build_params(self):
        self.word_embeddings = tf.Variable(self._initializer([self.vocabulary_size, self.embedding_size]), name='W_emb')

    def build_inference_ops(self):
        """
        Forward pass of the model
        """
        print('###: textPlaceHolder:', self.textPlaceHolder.get_shape().as_list())

        a = tf.nn.embedding_lookup(self.word_embeddings, self.textPlaceHolder)
        print('###: textEmbedding1:', a.get_shape().as_list())

        if self.keep_prob < 1:
            a = tf.nn.dropout(a, self.keep_prob)

        a = tf.layers.conv1d(inputs=a, filters=self.base_factor, kernel_size=3, strides=1, padding='same',
                             activation=None, kernel_initializer=self.weights_init_strategy, trainable=True)
        print('###: textEmbedding after first conv:', a.get_shape().as_list())

        a = self._convBlock(a, filterSize=3, numberOfFilters=self.base_factor, numOfLayers=2, tra=True,
                            weights_init_strategy=self.weights_init_strategy)
        # a = self._resBlock(a, self._convBlock(a, filterSize=3, numberOfFilters=self.base_factor, numOfLayers=2, tra=True,
        #                     weights_init_strategy=self.weights_init_strategy))
        print('###: textEmbedding after first convBlock:', a.get_shape().as_list())

        a = self._convBlock(a, filterSize=3, numberOfFilters=self.base_factor, numOfLayers=2, tra=True,
                            weights_init_strategy=self.weights_init_strategy)
        print('###: textEmbedding after second convBlock:', a.get_shape().as_list())

        a = tf.layers.max_pooling1d(inputs=a, pool_size=2, strides=2, padding='same')
        print('###: textEmbedding_final (after max-pool):', a.get_shape().as_list())

        a = self._convBlock(a, filterSize=3, numberOfFilters=(2*self.base_factor), numOfLayers=2, tra=True,
                            weights_init_strategy=self.weights_init_strategy)
        print('###: textEmbedding after third convBlock:', a.get_shape().as_list())

        a = self._convBlock(a, filterSize=3, numberOfFilters=(2*self.base_factor), numOfLayers=2, tra=True,
                            weights_init_strategy=self.weights_init_strategy)
        print('###: textEmbedding after fourth convBlock:', a.get_shape().as_list())

        a = tf.layers.max_pooling1d(inputs=a, pool_size=2, strides=2, padding='same')
        print('###: textEmbedding_final (after max-pool):', a.get_shape().as_list())

        a = self._convBlock(a, filterSize=3, numberOfFilters=(4*self.base_factor), numOfLayers=2, tra=True,
                            weights_init_strategy=self.weights_init_strategy)
        print('###: textEmbedding after fifth convBlock:', a.get_shape().as_list())

        a = self._convBlock(a, filterSize=3, numberOfFilters=(4*self.base_factor), numOfLayers=2, tra=True,
                            weights_init_strategy=self.weights_init_strategy)
        print('###: textEmbedding after sixth convBlock:', a.get_shape().as_list())

        a = tf.layers.max_pooling1d(inputs=a, pool_size=2, strides=2, padding='same')
        print('###: textEmbedding_final (after max-pool):', a.get_shape().as_list())

        a = self._convBlock(a, filterSize=3, numberOfFilters=(8*self.base_factor), numOfLayers=2, tra=True,
                            weights_init_strategy=self.weights_init_strategy)
        print('###: textEmbedding after seventh convBlock:', a.get_shape().as_list())

        a = self._convBlock(a, filterSize=3, numberOfFilters=(8*self.base_factor), numOfLayers=2, tra=True,
                            weights_init_strategy=self.weights_init_strategy)
        print('###: textEmbedding after eighth convBlock:', a.get_shape().as_list())
        #
        a = tf.layers.max_pooling1d(inputs=a, pool_size=8, strides=8, padding='same')
        print('###: textEmbedding_final (after max-pool):', a.get_shape().as_list())

        qA = a
        n_inputs = int(np.prod(qA.get_shape().as_list()[1:]))
        qA = tf.reshape(qA, [-1, n_inputs])
        if self.keep_prob == 1:
            qA = tf.layers.dense(inputs=qA, units=(self.base_factor * 32), activation=tf.nn.relu, trainable=True,
                                 use_bias=True, bias_regularizer=tf.nn.l2_loss, kernel_regularizer=tf.nn.l2_loss)
            print('### Dimensionality of the pair after first dense layer:', qA.get_shape().as_list())
            qA = tf.layers.dense(inputs=qA, units=(self.base_factor * 32), activation=tf.nn.relu, trainable=True,
                                 use_bias=True, bias_regularizer=tf.nn.l2_loss, kernel_regularizer=tf.nn.l2_loss)
            print('### Dimensionality of the pair after second layer:', qA.get_shape().as_list())
            qA = tf.layers.dense(inputs=qA, units=self.number_of_classes, activation=None, trainable=True,
                                 use_bias=True, bias_regularizer=tf.nn.l2_loss, kernel_regularizer=tf.nn.l2_loss)
            print('### Dimensionality of the pair after dense layers:', qA.get_shape().as_list())
        elif self.keep_prob < 1:
            # qA = tf.nn.dropout(qA, self.keep_prob)
            qA = tf.layers.dense(inputs=qA, units=(self.base_factor * 32), activation=tf.nn.relu, trainable=True,
                                 use_bias=True)
            print('### Dimensionality of the pair after first dense layer:', qA.get_shape().as_list())
            # qA = tf.nn.dropout(qA, self.keep_prob)
            qA = tf.layers.dense(inputs=qA, units=(self.base_factor * 32), activation=tf.nn.relu, trainable=True,
                                 use_bias=True)
            print('### Dimensionality of the pair after second layer:', qA.get_shape().as_list())
            qA = tf.nn.dropout(qA, self.keep_prob)
            qA = tf.layers.dense(inputs=qA, units=self.number_of_classes, activation=None, trainable=True,
                                 use_bias=True)
            print('### Dimensionality of the pair after dense layers:', qA.get_shape().as_list())

        # --------------------------------------------------------------------------------
        # y = tf.reshape(qA, [-1])
        y = tf.squeeze(qA)

        return y

    def build_loss_ops(self):
        """
        Loss
        """
        if self.number_of_classes == 1:
            loss_op = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.inference_op, labels=self.labelPlaceHolder))
        else:
            loss_op = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=self.inference_op, labels=self.labelPlaceHolder))
        return loss_op

    def build_opt_ops(self):
        train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss_op)
        return train_op

    def build_predict_ops(self):
        if self.number_of_classes == 1:
            predict_op = tf.nn.sigmoid(self.inference_op)
        else:
            predict_op = tf.nn.softmax(self.inference_op)
        return predict_op

    def batch_fit(self, text_ids, labels):
        feed_dict = {self.textPlaceHolder: text_ids,
                     self.labelPlaceHolder: labels}

        loss, _ = self._sess.run([self.loss_op, self.train_op], feed_dict=feed_dict)

        return loss

    def predict(self, text_ids, labels):
        feed_dict = {self.textPlaceHolder: text_ids,
                     self.labelPlaceHolder: labels}

        loss, predict = self._sess.run([self.loss_op, self.predict_op], feed_dict=feed_dict)

        return loss, predict


class VDWCNN(object):
    """ Very Deep Convolutional Networks for Text Classification. """
    def __init__(self,
                 session,
                 base_factor=64,
                 batch_size=64,
                 embedding_size=200,
                 vocabulary_size=1000,
                 number_of_words=35,
                 total_number_of_characters=100,
                 total_number_of_features=50,
                 weights_init_strategy=tf.contrib.layers.variance_scaling_initializer(),
                 keep_probability=0.5,
                 learning_rate=0.001,
                 number_of_classes=1,
                 phase_train=True):

        self.base_factor = base_factor
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.number_of_words = number_of_words
        self.vocabulary_size = vocabulary_size
        self.total_number_of_characters = total_number_of_characters
        self.total_number_of_features = total_number_of_features
        self.weights_init_strategy = weights_init_strategy
        self.weights_init_strategy = tf.contrib.layers.variance_scaling_initializer()
        self.keep_prob = keep_probability
        self.learning_rate = learning_rate
        self.number_of_classes = number_of_classes
        self.phase_train = phase_train

        # self._initializer = tf.random_normal_initializer(stddev=0.1)
        self._initializer = tf.truncated_normal
        # if options['initializer'] == 0 else tf.random_uniform_initializer(minval=-0.01, maxval=0.01)

        self._build_params()
        self._build_inputs()

        self.inference_op = self.build_inference_ops()
        self.loss_op = self.build_loss_ops()
        self.train_op = self.build_opt_ops()
        self.predict_op = self.build_predict_ops()

        self._init_op = tf.global_variables_initializer()
        self._sess = session
        self._sess.run(self._init_op)

        self._saver = tf.train.Saver()

    def _convBlock(self, x, filterSize, numberOfFilters, numOfLayers, tra, weights_init_strategy):
        net = x
        for i in range(numOfLayers):
            net = tf.layers.conv1d(inputs=net, filters=numberOfFilters, kernel_size=filterSize, strides=1,
                                   padding='same', activation=None, kernel_initializer=weights_init_strategy,
                                   trainable=tra)

            net = tf.layers.batch_normalization(inputs=net, trainable=tra)

            net = tf.nn.relu(net)

        return net

    def _resBlock(self, x, convBlock, poolBlock=None):
        """ Residual block """
        # TODO: seems that this operation prevents the model to optimize
        if poolBlock:
            return x  # TODO: figure this residual part out
        else:
            return tf.add(x, convBlock)

    def _build_inputs(self):
        self.textPlaceHolder = tf.placeholder(tf.int32, shape=[None, self.number_of_words])
        self.binaryFeaturesPlaceHolder = tf.placeholder(tf.float32, shape=[None, self.total_number_of_features])
        if self.number_of_classes > 1:
            self.labelPlaceHolder = tf.placeholder(tf.float32, shape=[None, self.number_of_classes])
        else:
            self.labelPlaceHolder = tf.placeholder(tf.float32, shape=[None, ])

    def _build_params(self):
        self.word_embeddings = tf.Variable(self._initializer([self.vocabulary_size, self.embedding_size]), name='W_emb')

    def build_inference_ops(self):
        """
        Forward pass of the model
        """
        print('###: textPlaceHolder:', self.textPlaceHolder.get_shape().as_list())

        a = tf.nn.embedding_lookup(self.word_embeddings, self.textPlaceHolder)
        print('###: textEmbedding1:', a.get_shape().as_list())

        if self.keep_prob < 1:
            a = tf.nn.dropout(a, self.keep_prob)

        a = tf.layers.conv1d(inputs=a, filters=self.base_factor, kernel_size=3, strides=1, padding='same',
                             activation=None, kernel_initializer=self.weights_init_strategy, trainable=True)
        print('###: textEmbedding after first conv:', a.get_shape().as_list())

        a = self._convBlock(a, filterSize=3, numberOfFilters=self.base_factor, numOfLayers=2, tra=True,
                            weights_init_strategy=self.weights_init_strategy)
        # a = self._resBlock(a, self._convBlock(a, filterSize=3, numberOfFilters=self.base_factor, numOfLayers=2, tra=True,
        #                     weights_init_strategy=self.weights_init_strategy))
        print('###: textEmbedding after first convBlock:', a.get_shape().as_list())

        a = self._convBlock(a, filterSize=3, numberOfFilters=self.base_factor, numOfLayers=2, tra=True,
                            weights_init_strategy=self.weights_init_strategy)
        print('###: textEmbedding after second convBlock:', a.get_shape().as_list())

        a = tf.layers.max_pooling1d(inputs=a, pool_size=2, strides=2, padding='same')
        print('###: textEmbedding_final (after max-pool):', a.get_shape().as_list())

        a = self._convBlock(a, filterSize=3, numberOfFilters=(2*self.base_factor), numOfLayers=2, tra=True,
                            weights_init_strategy=self.weights_init_strategy)
        print('###: textEmbedding after third convBlock:', a.get_shape().as_list())

        a = self._convBlock(a, filterSize=3, numberOfFilters=(2*self.base_factor), numOfLayers=2, tra=True,
                            weights_init_strategy=self.weights_init_strategy)
        print('###: textEmbedding after fourth convBlock:', a.get_shape().as_list())

        a = tf.layers.max_pooling1d(inputs=a, pool_size=2, strides=2, padding='same')
        print('###: textEmbedding_final (after max-pool):', a.get_shape().as_list())

        a = self._convBlock(a, filterSize=3, numberOfFilters=(4*self.base_factor), numOfLayers=2, tra=True,
                            weights_init_strategy=self.weights_init_strategy)
        print('###: textEmbedding after fifth convBlock:', a.get_shape().as_list())

        a = self._convBlock(a, filterSize=3, numberOfFilters=(4*self.base_factor), numOfLayers=2, tra=True,
                            weights_init_strategy=self.weights_init_strategy)
        print('###: textEmbedding after sixth convBlock:', a.get_shape().as_list())

        a = tf.layers.max_pooling1d(inputs=a, pool_size=2, strides=2, padding='same')
        print('###: textEmbedding_final (after max-pool):', a.get_shape().as_list())

        a = self._convBlock(a, filterSize=3, numberOfFilters=(8*self.base_factor), numOfLayers=2, tra=True,
                            weights_init_strategy=self.weights_init_strategy)
        print('###: textEmbedding after seventh convBlock:', a.get_shape().as_list())

        a = self._convBlock(a, filterSize=3, numberOfFilters=(8*self.base_factor), numOfLayers=2, tra=True,
                            weights_init_strategy=self.weights_init_strategy)
        print('###: textEmbedding after eighth convBlock:', a.get_shape().as_list())
        #
        a = tf.layers.max_pooling1d(inputs=a, pool_size=8, strides=8, padding='same')
        print('###: textEmbedding_final (after max-pool):', a.get_shape().as_list())

        qA = a
        n_inputs = int(np.prod(qA.get_shape().as_list()[1:]))
        qA = tf.reshape(qA, [-1, n_inputs])
        if self.keep_prob == 1:
            qA = tf.layers.dense(inputs=qA, units=(self.base_factor * 32), activation=tf.nn.relu, trainable=True,
                                 use_bias=True, bias_regularizer=tf.nn.l2_loss, kernel_regularizer=tf.nn.l2_loss)
            print('### Dimensionality of the pair after first dense layer:', qA.get_shape().as_list())

            qA = tf.concat([qA, self.binaryFeaturesPlaceHolder], 1)
            print('### Dimensionality of the pair after join with binary features:', qA.get_shape().as_list())

            qA = tf.layers.dense(inputs=qA, units=(self.base_factor * 32), activation=tf.nn.relu, trainable=True,
                                 use_bias=True, bias_regularizer=tf.nn.l2_loss, kernel_regularizer=tf.nn.l2_loss)
            print('### Dimensionality of the pair after second layer:', qA.get_shape().as_list())
            qA = tf.layers.dense(inputs=qA, units=self.number_of_classes, activation=None, trainable=True,
                                 use_bias=True, bias_regularizer=tf.nn.l2_loss, kernel_regularizer=tf.nn.l2_loss)
            print('### Dimensionality of the pair after dense layers:', qA.get_shape().as_list())
        elif self.keep_prob < 1:
            # qA = tf.nn.dropout(qA, self.keep_prob)
            qA = tf.layers.dense(inputs=qA, units=(self.base_factor * 32), activation=tf.nn.relu, trainable=True,
                                 use_bias=True)
            print('### Dimensionality of the pair after first dense layer:', qA.get_shape().as_list())

            qA = tf.concat([qA, self.binaryFeaturesPlaceHolder], 1)
            print('### Dimensionality of the pair after join with binary features:', qA.get_shape().as_list())

            # qA = tf.nn.dropout(qA, self.keep_prob)
            qA = tf.layers.dense(inputs=qA, units=(self.base_factor * 32), activation=tf.nn.relu, trainable=True,
                                 use_bias=True)
            print('### Dimensionality of the pair after second layer:', qA.get_shape().as_list())
            qA = tf.nn.dropout(qA, self.keep_prob)
            qA = tf.layers.dense(inputs=qA, units=self.number_of_classes, activation=None, trainable=True,
                                 use_bias=True)
            print('### Dimensionality of the pair after dense layers:', qA.get_shape().as_list())

        # --------------------------------------------------------------------------------
        # y = tf.reshape(qA, [-1])
        y = tf.squeeze(qA)

        return y

    def build_loss_ops(self):
        """
        Loss
        """
        if self.number_of_classes == 1:
            loss_op = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.inference_op, labels=self.labelPlaceHolder))
        else:
            loss_op = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=self.inference_op, labels=self.labelPlaceHolder))
        return loss_op

    def build_opt_ops(self):
        train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss_op)
        return train_op

    def build_predict_ops(self):
        if self.number_of_classes == 1:
            predict_op = tf.nn.sigmoid(self.inference_op)
        else:
            predict_op = tf.nn.softmax(self.inference_op)
        return predict_op

    def batch_fit(self, text_ids, binary_features, labels):
        feed_dict = {self.textPlaceHolder: text_ids,
                     self.binaryFeaturesPlaceHolder: binary_features,
                     self.labelPlaceHolder: labels}

        loss, _ = self._sess.run([self.loss_op, self.train_op], feed_dict=feed_dict)

        return loss

    def predict(self, text_ids, binary_features, labels):
        feed_dict = {self.textPlaceHolder: text_ids,
                     self.binaryFeaturesPlaceHolder: binary_features,
                     self.labelPlaceHolder: labels}

        loss, predict = self._sess.run([self.loss_op, self.predict_op], feed_dict=feed_dict)

        return loss, predict


class BiDirectionalRNNModel(object):
    """ Deep Attention Model for Text Classification. """
    def __init__(self,
                 session,
                 batch_size=64,
                 embedding_size=256,
                 bi_rnn_text_embedding=128,
                 linear_projection_embedding=128,
                 final_linear_projection_embedding=128,
                 vocabulary_size=1000,
                 number_of_words=35,
                 total_number_of_characters=100,
                 weights_init_strategy=tf.contrib.layers.variance_scaling_initializer(),
                 keep_probability=0.5,
                 learning_rate=0.001,
                 number_of_classes=1,
                 phase_train=True):

        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.bi_rnn_text_embedding = bi_rnn_text_embedding
        self.linear_projection_embedding = linear_projection_embedding
        self.final_linear_projection_embedding = final_linear_projection_embedding
        self.number_of_words = number_of_words
        self.vocabulary_size = vocabulary_size
        self.total_number_of_characters = total_number_of_characters
        self.weights_init_strategy = weights_init_strategy
        self.weights_init_strategy = tf.contrib.layers.variance_scaling_initializer()
        self.keep_prob = keep_probability
        self.learning_rate = learning_rate
        self.number_of_classes = number_of_classes
        self.phase_train = phase_train

        # self._initializer = tf.random_normal_initializer(stddev=0.1)
        self._initializer = tf.truncated_normal
        # if options['initializer'] == 0 else tf.random_uniform_initializer(minval=-0.01, maxval=0.01)

        self._build_params()
        self._build_inputs()

        self.inference_op = self.build_inference_ops()
        self.loss_op = self.build_loss_ops()
        self.train_op = self.build_opt_ops()
        self.predict_op = self.build_predict_ops()

        self._init_op = tf.global_variables_initializer()
        self._sess = session
        self._sess.run(self._init_op)

        self._saver = tf.train.Saver()

    def _build_inputs(self):
        self.textPlaceHolder = tf.placeholder(tf.int32, shape=[None, self.number_of_words])
        if self.number_of_classes > 1:
            self.labelPlaceHolder = tf.placeholder(tf.float32, shape=[None, self.number_of_classes])
        else:
            self.labelPlaceHolder = tf.placeholder(tf.float32, shape=[None, ])

    def _build_params(self):
        self.word_embeddings = tf.Variable(self._initializer([self.vocabulary_size, self.embedding_size]), name='W_emb')

    def build_inference_ops(self):
        """
        Forward pass of the model
        """
        print('###: textPlaceHolder:', self.textPlaceHolder.get_shape().as_list())
        a = tf.nn.embedding_lookup(self.word_embeddings, self.textPlaceHolder)
        print('###: textEmbedding1:', a.get_shape().as_list())

        with tf.variable_scope('birnn_text'):
            fw_lstm_cell_a = tf.contrib.rnn.LSTMCell(num_units=int(self.bi_rnn_text_embedding / 2), state_is_tuple=True)
            bw_lstm_cell_a = tf.contrib.rnn.LSTMCell(num_units=int(self.bi_rnn_text_embedding / 2), state_is_tuple=True)
            outputs, states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=fw_lstm_cell_a,
                cell_bw=bw_lstm_cell_a,
                dtype=tf.float32,
                inputs=a)

        a = tf.concat(outputs, 2)
        print('###: textEmbedding3:', a.get_shape().as_list())

        aDense = tf.reduce_sum(a, axis=2)
        print('###: textEmbedding5 (sum-pool layer):', aDense.get_shape().as_list())

        # aDense = tf.squeeze(aDense)
        # print('### Ad representation (after attention):', aDense.get_shape().as_list())

        qA = aDense
        if self.keep_prob < 1:
            # qA = tf.nn.dropout(qA, self.keep_prob)
            # qA = tf.layers.dense(inputs=qA, units=(self.final_linear_projection_embedding * 4),
            #                      activation=tf.nn.relu, trainable=True, use_bias=True)
            # # qA = tf.nn.dropout(qA, self.keep_prob)
            # print('### Dimensionality of the text after first dense layer:', qA.get_shape().as_list())
            # qA = tf.layers.dense(inputs=qA, units=(self.final_linear_projection_embedding * 4),
            #                      activation=tf.nn.relu, trainable=True, use_bias=True)
            # qA = tf.nn.dropout(qA, self.keep_prob)
            # print('### Dimensionality of the text after second layer:', qA.get_shape().as_list())
            qA = tf.layers.dense(inputs=qA, units=1, activation=None, trainable=True, use_bias=True)
            print('### Dimensionality of the text after dense layers:', qA.get_shape().as_list())
        elif self.keep_prob == 1:
            # qA = tf.layers.dense(inputs=qA, units=(self.final_linear_projection_embedding * 4),
            #                      activation=tf.nn.relu, trainable=True,
            #                      use_bias=True, bias_regularizer=tf.nn.l2_loss, kernel_regularizer=tf.nn.l2_loss)
            # print('### Dimensionality of the text after first dense layer:', qA.get_shape().as_list())
            # qA = tf.layers.dense(inputs=qA, units=(self.final_linear_projection_embedding * 4),
            #                      activation=tf.nn.relu, trainable=True,
            #                      use_bias=True, bias_regularizer=tf.nn.l2_loss, kernel_regularizer=tf.nn.l2_loss)
            # print('### Dimensionality of the text after second layer:', qA.get_shape().as_list())
            qA = tf.layers.dense(inputs=qA, units=self.number_of_classes, activation=None, trainable=True,
                                 use_bias=True, bias_regularizer=tf.nn.l2_loss, kernel_regularizer=tf.nn.l2_loss)
            print('### Dimensionality of the text after dense layers:', qA.get_shape().as_list())

        # --------------------------------------------------------------------------------
        y = tf.squeeze(qA)

        return y

    def build_loss_ops(self):
        """
        Loss
        """
        if self.number_of_classes == 1:
            loss_op = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.inference_op, labels=self.labelPlaceHolder))
        else:
            loss_op = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=self.inference_op, labels=self.labelPlaceHolder))
        return loss_op

    def build_opt_ops(self):
        train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss_op)
        return train_op

    def build_predict_ops(self):
        if self.number_of_classes == 1:
            predict_op = tf.nn.sigmoid(self.inference_op)
        else:
            predict_op = tf.nn.softmax(self.inference_op)
        return predict_op

    def batch_fit(self, text_ids, labels):
        feed_dict = {self.textPlaceHolder: text_ids,
                     self.labelPlaceHolder: labels}

        loss, _ = self._sess.run([self.loss_op, self.train_op], feed_dict=feed_dict)

        return loss

    def predict(self, text_ids, labels):
        feed_dict = {self.textPlaceHolder: text_ids,
                     self.labelPlaceHolder: labels}

        loss, predict = self._sess.run([self.loss_op, self.predict_op], feed_dict=feed_dict)

        return loss, predict


class BiDirectionalRNNWideModel(object):
    def __init__(self,
                 session,
                 batch_size=64,
                 embedding_size=256,
                 bi_rnn_text_embedding=128,
                 linear_projection_embedding=128,
                 final_linear_projection_embedding=128,
                 vocabulary_size=1000,
                 number_of_words=35,
                 total_number_of_features=50,
                 weights_init_strategy=tf.contrib.layers.variance_scaling_initializer(),
                 keep_probability=0.5,
                 learning_rate=0.001,
                 number_of_classes=1,
                 phase_train=True):

        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.bi_rnn_text_embedding = bi_rnn_text_embedding
        self.linear_projection_embedding = linear_projection_embedding
        self.final_linear_projection_embedding = final_linear_projection_embedding
        self.number_of_words = number_of_words
        self.vocabulary_size = vocabulary_size
        self.total_number_of_features = total_number_of_features
        self.weights_init_strategy = weights_init_strategy
        self.weights_init_strategy = tf.contrib.layers.variance_scaling_initializer()
        self.keep_prob = keep_probability
        self.learning_rate = learning_rate
        self.number_of_classes = number_of_classes
        self.phase_train = phase_train

        # self._initializer = tf.random_normal_initializer(stddev=0.1)
        self._initializer = tf.truncated_normal
        # if options['initializer'] == 0 else tf.random_uniform_initializer(minval=-0.01, maxval=0.01)

        self._build_params()
        self._build_inputs()

        self.inference_op = self.build_inference_ops()
        self.loss_op = self.build_loss_ops()
        self.train_op = self.build_opt_ops()
        self.predict_op = self.build_predict_ops()

        self._init_op = tf.global_variables_initializer()
        self._sess = session
        self._sess.run(self._init_op)

        self._saver = tf.train.Saver()

    def _build_inputs(self):
        self.textPlaceHolder = tf.placeholder(tf.int32, shape=[None, self.number_of_words])
        self.binaryFeaturesPlaceHolder = tf.placeholder(tf.float32, shape=[None, self.total_number_of_features])
        if self.number_of_classes > 1:
            self.labelPlaceHolder = tf.placeholder(tf.float32, shape=[None, self.number_of_classes])
        else:
            self.labelPlaceHolder = tf.placeholder(tf.float32, shape=[None, ])

    def _build_params(self):
        self.word_embeddings = tf.Variable(self._initializer([self.vocabulary_size, self.embedding_size]), name='W_emb')

    def build_inference_ops(self):
        """
        Forward pass of the model
        """
        print('###: textPlaceHolder:', self.textPlaceHolder.get_shape().as_list())
        a = tf.nn.embedding_lookup(self.word_embeddings, self.textPlaceHolder)
        print('###: textEmbedding1:', a.get_shape().as_list())

        # a = tf.layers.dense(inputs=a, units=self.linear_projection_embedding, activation=tf.nn.relu, trainable=True,
        #                     name='joint_linear_projection')
        # print('###: textEmbedding2:', a.get_shape().as_list())

        with tf.variable_scope('birnn_text'):
            fw_lstm_cell_a = tf.contrib.rnn.LSTMCell(num_units=int(self.bi_rnn_text_embedding / 2), state_is_tuple=True)
            bw_lstm_cell_a = tf.contrib.rnn.LSTMCell(num_units=int(self.bi_rnn_text_embedding / 2), state_is_tuple=True)
            outputs, states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=fw_lstm_cell_a,
                cell_bw=bw_lstm_cell_a,
                dtype=tf.float32,
                inputs=a)

        a = tf.concat(outputs, 2)
        print('###: textEmbedding3:', a.get_shape().as_list())

        # a = tf.layers.dense(inputs=a, units=self.final_linear_projection_embedding, activation=None, trainable=True,
        #                     name='second_ad_linear')
        # print('###: textEmbedding4:', a.get_shape().as_list())



        aDense = tf.reduce_sum(a, axis=2)
        print('###: textEmbedding5 (after sum-pool layer):', aDense.get_shape().as_list())

        # aDense = tf.squeeze(aDense)
        # print('### Ad representation (after attention):', aDense.get_shape().as_list())

        qA = aDense
        if self.keep_prob < 1:
            # qA = tf.nn.dropout(qA, self.keep_prob)
            # qA = tf.layers.dense(inputs=qA, units=(self.final_linear_projection_embedding * 4),
            #                      activation=tf.nn.relu, trainable=True, use_bias=True)
            # # qA = tf.nn.dropout(qA, self.keep_prob)
            # print('### Dimensionality of the text after first dense layer:', qA.get_shape().as_list())
            #
            # qA = tf.concat([qA, self.binaryFeaturesPlaceHolder], 1)
            # print('### Dimensionality of the text after join with binary features:', qA.get_shape().as_list())
            #
            # qA = tf.layers.dense(inputs=qA, units=(self.final_linear_projection_embedding * 4),
            #                      activation=tf.nn.relu, trainable=True, use_bias=True)
            # qA = tf.nn.dropout(qA, self.keep_prob)
            # print('### Dimensionality of the text after second layer:', qA.get_shape().as_list())
            qA = tf.layers.dense(inputs=qA, units=self.number_of_classes, activation=None, trainable=True, use_bias=True)
            print('### Dimensionality of the text after dense layers:', qA.get_shape().as_list())
        elif self.keep_prob == 1:
            # qA = tf.layers.dense(inputs=qA, units=(self.final_linear_projection_embedding * 4),
            #                      activation=tf.nn.relu, trainable=True,
            #                      use_bias=True, bias_regularizer=tf.nn.l2_loss, kernel_regularizer=tf.nn.l2_loss)
            # print('### Dimensionality of the text after first dense layer:', qA.get_shape().as_list())
            #
            # qA = tf.concat([qA, self.binaryFeaturesPlaceHolder], 1)
            # print('### Dimensionality of the text after join with binary features:', qA.get_shape().as_list())
            #
            # qA = tf.layers.dense(inputs=qA, units=(self.final_linear_projection_embedding * 4),
            #                      activation=tf.nn.relu, trainable=True,
            #                      use_bias=True, bias_regularizer=tf.nn.l2_loss, kernel_regularizer=tf.nn.l2_loss)
            # print('### Dimensionality of the text after second layer:', qA.get_shape().as_list())
            qA = tf.layers.dense(inputs=qA, units=self.number_of_classes, activation=None, trainable=True,
                                 use_bias=True, bias_regularizer=tf.nn.l2_loss, kernel_regularizer=tf.nn.l2_loss)
            print('### Dimensionality of the text after dense layers:', qA.get_shape().as_list())

        # --------------------------------------------------------------------------------
        y = tf.squeeze(qA)

        return y

    def build_loss_ops(self):
        """
        Loss
        """
        if self.number_of_classes == 1:
            loss_op = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.inference_op, labels=self.labelPlaceHolder))
        else:
            loss_op = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=self.inference_op, labels=self.labelPlaceHolder))
        return loss_op

    def build_opt_ops(self):
        train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss_op)
        return train_op

    def build_predict_ops(self):
        if self.number_of_classes == 1:
            predict_op = tf.nn.sigmoid(self.inference_op)
        else:
            predict_op = tf.nn.softmax(self.inference_op)
        return predict_op

    def batch_fit(self, text_ids, binary_features, labels):
        feed_dict = {self.textPlaceHolder: text_ids,
                     self.binaryFeaturesPlaceHolder: binary_features,
                     self.labelPlaceHolder: labels}

        loss, _ = self._sess.run([self.loss_op, self.train_op], feed_dict=feed_dict)

        return loss

    def predict(self, text_ids, binary_features, labels):
        feed_dict = {self.textPlaceHolder: text_ids,
                     self.binaryFeaturesPlaceHolder: binary_features,
                     self.labelPlaceHolder: labels}

        loss, predict = self._sess.run([self.loss_op, self.predict_op], feed_dict=feed_dict)

        return loss, predict


class DeepTextSumPoolModel(object):
    """ Deep Attention Model for Text Classification. """
    def __init__(self,
                 session,
                 batch_size=64,
                 embedding_size=256,
                 bi_rnn_text_embedding=128,
                 linear_projection_embedding=128,
                 final_linear_projection_embedding=128,
                 vocabulary_size=1000,
                 number_of_words=35,
                 total_number_of_characters=100,
                 weights_init_strategy=tf.contrib.layers.variance_scaling_initializer(),
                 keep_probability=0.5,
                 learning_rate=0.001,
                 number_of_classes=1,
                 phase_train=True):

        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.bi_rnn_text_embedding = bi_rnn_text_embedding
        self.linear_projection_embedding = linear_projection_embedding
        self.final_linear_projection_embedding = final_linear_projection_embedding
        self.number_of_words = number_of_words
        self.vocabulary_size = vocabulary_size
        self.total_number_of_characters = total_number_of_characters
        self.weights_init_strategy = weights_init_strategy
        self.weights_init_strategy = tf.contrib.layers.variance_scaling_initializer()
        self.keep_prob = keep_probability
        self.learning_rate = learning_rate
        self.number_of_classes = number_of_classes
        self.phase_train = phase_train

        # self._initializer = tf.random_normal_initializer(stddev=0.1)
        self._initializer = tf.truncated_normal
        # if options['initializer'] == 0 else tf.random_uniform_initializer(minval=-0.01, maxval=0.01)

        self._build_params()
        self._build_inputs()

        self.inference_op = self.build_inference_ops()
        self.loss_op = self.build_loss_ops()
        self.train_op = self.build_opt_ops()
        self.predict_op = self.build_predict_ops()

        self._init_op = tf.global_variables_initializer()
        self._sess = session
        self._sess.run(self._init_op)

        self._saver = tf.train.Saver()

    def _build_inputs(self):
        self.textPlaceHolder = tf.placeholder(tf.int32, shape=[None, self.number_of_words])
        if self.number_of_classes > 1:
            self.labelPlaceHolder = tf.placeholder(tf.float32, shape=[None, self.number_of_classes])
        else:
            self.labelPlaceHolder = tf.placeholder(tf.float32, shape=[None, ])

    def _build_params(self):
        self.word_embeddings = tf.Variable(self._initializer([self.vocabulary_size, self.embedding_size]), name='W_emb')

    def build_inference_ops(self):
        """
        Forward pass of the model
        """
        print('###: textPlaceHolder:', self.textPlaceHolder.get_shape().as_list())
        a = tf.nn.embedding_lookup(self.word_embeddings, self.textPlaceHolder)
        print('###: textEmbedding1:', a.get_shape().as_list())

        a = tf.layers.dense(inputs=a, units=self.linear_projection_embedding, activation=tf.nn.relu, trainable=True,
                            name='joint_linear_projection')
        print('###: textEmbedding2:', a.get_shape().as_list())

        with tf.variable_scope('birnn_text'):
            fw_lstm_cell_a = tf.contrib.rnn.LSTMCell(num_units=int(self.bi_rnn_text_embedding / 2), state_is_tuple=True)
            bw_lstm_cell_a = tf.contrib.rnn.LSTMCell(num_units=int(self.bi_rnn_text_embedding / 2), state_is_tuple=True)
            outputs, states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=fw_lstm_cell_a,
                cell_bw=bw_lstm_cell_a,
                dtype=tf.float32,
                inputs=a)

        a = tf.concat(outputs, 2)
        print('###: textEmbedding3:', a.get_shape().as_list())

        a = tf.layers.dense(inputs=a, units=self.final_linear_projection_embedding, activation=None, trainable=True,
                            name='second_ad_linear')
        print('###: textEmbedding4:', a.get_shape().as_list())

        aDense = tf.reduce_sum(a, axis=2)
        print('###: textEmbedding5 (sum-pool layer):', aDense.get_shape().as_list())

        # aDense = tf.squeeze(aDense)
        # print('### Ad representation (after attention):', aDense.get_shape().as_list())

        qA = aDense
        if self.keep_prob < 1:
            # qA = tf.nn.dropout(qA, self.keep_prob)
            qA = tf.layers.dense(inputs=qA, units=(self.final_linear_projection_embedding * 4),
                                 activation=tf.nn.relu, trainable=True, use_bias=True)
            # qA = tf.nn.dropout(qA, self.keep_prob)
            print('### Dimensionality of the text after first dense layer:', qA.get_shape().as_list())
            qA = tf.layers.dense(inputs=qA, units=(self.final_linear_projection_embedding * 4),
                                 activation=tf.nn.relu, trainable=True, use_bias=True)
            qA = tf.nn.dropout(qA, self.keep_prob)
            print('### Dimensionality of the text after second layer:', qA.get_shape().as_list())
            qA = tf.layers.dense(inputs=qA, units=1, activation=None, trainable=True, use_bias=True)
            print('### Dimensionality of the text after dense layers:', qA.get_shape().as_list())
        elif self.keep_prob == 1:
            qA = tf.layers.dense(inputs=qA, units=(self.final_linear_projection_embedding * 4),
                                 activation=tf.nn.relu, trainable=True,
                                 use_bias=True, bias_regularizer=tf.nn.l2_loss, kernel_regularizer=tf.nn.l2_loss)
            print('### Dimensionality of the text after first dense layer:', qA.get_shape().as_list())
            qA = tf.layers.dense(inputs=qA, units=(self.final_linear_projection_embedding * 4),
                                 activation=tf.nn.relu, trainable=True,
                                 use_bias=True, bias_regularizer=tf.nn.l2_loss, kernel_regularizer=tf.nn.l2_loss)
            print('### Dimensionality of the text after second layer:', qA.get_shape().as_list())
            qA = tf.layers.dense(inputs=qA, units=self.number_of_classes, activation=None, trainable=True,
                                 use_bias=True, bias_regularizer=tf.nn.l2_loss, kernel_regularizer=tf.nn.l2_loss)
            print('### Dimensionality of the text after dense layers:', qA.get_shape().as_list())

        # --------------------------------------------------------------------------------
        y = tf.squeeze(qA)

        return y

    def build_loss_ops(self):
        """
        Loss
        """
        if self.number_of_classes == 1:
            loss_op = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.inference_op, labels=self.labelPlaceHolder))
        else:
            loss_op = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=self.inference_op, labels=self.labelPlaceHolder))
        return loss_op

    def build_opt_ops(self):
        train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss_op)
        return train_op

    def build_predict_ops(self):
        if self.number_of_classes == 1:
            predict_op = tf.nn.sigmoid(self.inference_op)
        else:
            predict_op = tf.nn.softmax(self.inference_op)
        return predict_op

    def batch_fit(self, text_ids, labels):
        feed_dict = {self.textPlaceHolder: text_ids,
                     self.labelPlaceHolder: labels}

        loss, _ = self._sess.run([self.loss_op, self.train_op], feed_dict=feed_dict)

        return loss

    def predict(self, text_ids, labels):
        feed_dict = {self.textPlaceHolder: text_ids,
                     self.labelPlaceHolder: labels}

        loss, predict = self._sess.run([self.loss_op, self.predict_op], feed_dict=feed_dict)

        return loss, predict


class DeepAndWideTextSumPoolModel(object):
    """ Deep Attention Model for Text Classification. """
    def __init__(self,
                 session,
                 batch_size=64,
                 embedding_size=256,
                 bi_rnn_text_embedding=128,
                 linear_projection_embedding=128,
                 final_linear_projection_embedding=128,
                 vocabulary_size=1000,
                 number_of_words=35,
                 total_number_of_features=50,
                 weights_init_strategy=tf.contrib.layers.variance_scaling_initializer(),
                 keep_probability=0.5,
                 learning_rate=0.001,
                 number_of_classes=1,
                 phase_train=True):

        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.bi_rnn_text_embedding = bi_rnn_text_embedding
        self.linear_projection_embedding = linear_projection_embedding
        self.final_linear_projection_embedding = final_linear_projection_embedding
        self.number_of_words = number_of_words
        self.vocabulary_size = vocabulary_size
        self.total_number_of_features = total_number_of_features
        self.weights_init_strategy = weights_init_strategy
        self.weights_init_strategy = tf.contrib.layers.variance_scaling_initializer()
        self.keep_prob = keep_probability
        self.learning_rate = learning_rate
        self.number_of_classes = number_of_classes
        self.phase_train = phase_train

        # self._initializer = tf.random_normal_initializer(stddev=0.1)
        self._initializer = tf.truncated_normal
        # if options['initializer'] == 0 else tf.random_uniform_initializer(minval=-0.01, maxval=0.01)

        self._build_params()
        self._build_inputs()

        self.inference_op = self.build_inference_ops()
        self.loss_op = self.build_loss_ops()
        self.train_op = self.build_opt_ops()
        self.predict_op = self.build_predict_ops()

        self._init_op = tf.global_variables_initializer()
        self._sess = session
        self._sess.run(self._init_op)

        self._saver = tf.train.Saver()

    def _build_inputs(self):
        self.textPlaceHolder = tf.placeholder(tf.int32, shape=[None, self.number_of_words])
        self.binaryFeaturesPlaceHolder = tf.placeholder(tf.float32, shape=[None, self.total_number_of_features])
        if self.number_of_classes > 1:
            self.labelPlaceHolder = tf.placeholder(tf.float32, shape=[None, self.number_of_classes])
        else:
            self.labelPlaceHolder = tf.placeholder(tf.float32, shape=[None, ])

    def _build_params(self):
        self.word_embeddings = tf.Variable(self._initializer([self.vocabulary_size, self.embedding_size]), name='W_emb')

    def build_inference_ops(self):
        """
        Forward pass of the model
        """
        print('###: textPlaceHolder:', self.textPlaceHolder.get_shape().as_list())
        a = tf.nn.embedding_lookup(self.word_embeddings, self.textPlaceHolder)
        print('###: textEmbedding1:', a.get_shape().as_list())

        a = tf.layers.dense(inputs=a, units=self.linear_projection_embedding, activation=tf.nn.relu, trainable=True,
                            name='joint_linear_projection')
        print('###: textEmbedding2:', a.get_shape().as_list())

        with tf.variable_scope('birnn_text'):
            fw_lstm_cell_a = tf.contrib.rnn.LSTMCell(num_units=int(self.bi_rnn_text_embedding / 2), state_is_tuple=True)
            bw_lstm_cell_a = tf.contrib.rnn.LSTMCell(num_units=int(self.bi_rnn_text_embedding / 2), state_is_tuple=True)
            outputs, states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=fw_lstm_cell_a,
                cell_bw=bw_lstm_cell_a,
                dtype=tf.float32,
                inputs=a)

        a = tf.concat(outputs, 2)
        print('###: textEmbedding3:', a.get_shape().as_list())

        a = tf.layers.dense(inputs=a, units=self.final_linear_projection_embedding, activation=None, trainable=True,
                            name='second_ad_linear')
        print('###: textEmbedding4:', a.get_shape().as_list())



        aDense = tf.reduce_sum(a, axis=2)
        print('###: textEmbedding5 (after sum-pool layer):', aDense.get_shape().as_list())

        # aDense = tf.squeeze(aDense)
        # print('### Ad representation (after attention):', aDense.get_shape().as_list())

        qA = aDense
        if self.keep_prob < 1:
            # qA = tf.nn.dropout(qA, self.keep_prob)
            qA = tf.layers.dense(inputs=qA, units=(self.final_linear_projection_embedding * 4),
                                 activation=tf.nn.relu, trainable=True, use_bias=True)
            # qA = tf.nn.dropout(qA, self.keep_prob)
            print('### Dimensionality of the text after first dense layer:', qA.get_shape().as_list())

            qA = tf.concat([qA, self.binaryFeaturesPlaceHolder], 1)
            print('### Dimensionality of the text after join with binary features:', qA.get_shape().as_list())

            qA = tf.layers.dense(inputs=qA, units=(self.final_linear_projection_embedding * 4),
                                 activation=tf.nn.relu, trainable=True, use_bias=True)
            qA = tf.nn.dropout(qA, self.keep_prob)
            print('### Dimensionality of the text after second layer:', qA.get_shape().as_list())
            qA = tf.layers.dense(inputs=qA, units=self.number_of_classes, activation=None, trainable=True, use_bias=True)
            print('### Dimensionality of the text after dense layers:', qA.get_shape().as_list())
        elif self.keep_prob == 1:
            qA = tf.layers.dense(inputs=qA, units=(self.final_linear_projection_embedding * 4),
                                 activation=tf.nn.relu, trainable=True,
                                 use_bias=True, bias_regularizer=tf.nn.l2_loss, kernel_regularizer=tf.nn.l2_loss)
            print('### Dimensionality of the text after first dense layer:', qA.get_shape().as_list())

            qA = tf.concat([qA, self.binaryFeaturesPlaceHolder], 1)
            print('### Dimensionality of the text after join with binary features:', qA.get_shape().as_list())

            qA = tf.layers.dense(inputs=qA, units=(self.final_linear_projection_embedding * 4),
                                 activation=tf.nn.relu, trainable=True,
                                 use_bias=True, bias_regularizer=tf.nn.l2_loss, kernel_regularizer=tf.nn.l2_loss)
            print('### Dimensionality of the text after second layer:', qA.get_shape().as_list())
            qA = tf.layers.dense(inputs=qA, units=self.number_of_classes, activation=None, trainable=True,
                                 use_bias=True, bias_regularizer=tf.nn.l2_loss, kernel_regularizer=tf.nn.l2_loss)
            print('### Dimensionality of the text after dense layers:', qA.get_shape().as_list())

        # --------------------------------------------------------------------------------
        y = tf.squeeze(qA)

        return y

    def build_loss_ops(self):
        """
        Loss
        """
        if self.number_of_classes == 1:
            loss_op = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.inference_op, labels=self.labelPlaceHolder))
        else:
            loss_op = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=self.inference_op, labels=self.labelPlaceHolder))
        return loss_op

    def build_opt_ops(self):
        train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss_op)
        return train_op

    def build_predict_ops(self):
        if self.number_of_classes == 1:
            predict_op = tf.nn.sigmoid(self.inference_op)
        else:
            predict_op = tf.nn.softmax(self.inference_op)
        return predict_op

    def batch_fit(self, text_ids, binary_features, labels):
        feed_dict = {self.textPlaceHolder: text_ids,
                     self.binaryFeaturesPlaceHolder: binary_features,
                     self.labelPlaceHolder: labels}

        loss, _ = self._sess.run([self.loss_op, self.train_op], feed_dict=feed_dict)

        return loss

    def predict(self, text_ids, binary_features, labels):
        feed_dict = {self.textPlaceHolder: text_ids,
                     self.binaryFeaturesPlaceHolder: binary_features,
                     self.labelPlaceHolder: labels}

        loss, predict = self._sess.run([self.loss_op, self.predict_op], feed_dict=feed_dict)

        return loss, predict


class DeepTextAttentionModel(object):
    """ Deep Attention Model for Text Classification. """
    def __init__(self,
                 session,
                 batch_size=64,
                 embedding_size=256,
                 bi_rnn_text_embedding=128,
                 linear_projection_embedding=128,
                 final_linear_projection_embedding=128,
                 vocabulary_size=1000,
                 number_of_words=35,
                 total_number_of_characters=100,
                 weights_init_strategy=tf.contrib.layers.variance_scaling_initializer(),
                 keep_probability=0.5,
                 learning_rate=0.001,
                 number_of_classes=1,
                 phase_train=True):

        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.bi_rnn_text_embedding = bi_rnn_text_embedding
        self.linear_projection_embedding = linear_projection_embedding
        self.final_linear_projection_embedding = final_linear_projection_embedding
        self.number_of_words = number_of_words
        self.vocabulary_size = vocabulary_size
        self.total_number_of_characters = total_number_of_characters
        self.weights_init_strategy = weights_init_strategy
        self.weights_init_strategy = tf.contrib.layers.variance_scaling_initializer()
        self.keep_prob = keep_probability
        self.learning_rate = learning_rate
        self.number_of_classes = number_of_classes
        self.phase_train = phase_train

        # self._initializer = tf.random_normal_initializer(stddev=0.1)
        self._initializer = tf.truncated_normal
        # if options['initializer'] == 0 else tf.random_uniform_initializer(minval=-0.01, maxval=0.01)

        self._build_params()
        self._build_inputs()

        self.inference_op, self.attentions = self.build_inference_ops()
        self.loss_op = self.build_loss_ops()
        self.train_op = self.build_opt_ops()
        self.predict_op = self.build_predict_ops()

        self._init_op = tf.global_variables_initializer()
        self._sess = session
        self._sess.run(self._init_op)

        self._saver = tf.train.Saver()

    def _build_inputs(self):
        self.textPlaceHolder = tf.placeholder(tf.int32, shape=[None, self.number_of_words])
        if self.number_of_classes > 1:
            self.labelPlaceHolder = tf.placeholder(tf.float32, shape=[None, self.number_of_classes])
        else:
            self.labelPlaceHolder = tf.placeholder(tf.float32, shape=[None, ])

    def _build_params(self):
        self.word_embeddings = tf.Variable(self._initializer([self.vocabulary_size, self.embedding_size]), name='W_emb')

    def build_inference_ops(self):
        """
        Forward pass of the model
        """
        print('###: textPlaceHolder:', self.textPlaceHolder.get_shape().as_list())
        a = tf.nn.embedding_lookup(self.word_embeddings, self.textPlaceHolder)
        print('###: textEmbedding1:', a.get_shape().as_list())

        a = tf.layers.dense(inputs=a, units=self.linear_projection_embedding, activation=tf.nn.relu, trainable=True,
                            name='joint_linear_projection')
        print('###: textEmbedding2:', a.get_shape().as_list())

        with tf.variable_scope('birnn_text'):
            fw_lstm_cell_a = tf.contrib.rnn.LSTMCell(num_units=int(self.bi_rnn_text_embedding / 2), state_is_tuple=True)
            bw_lstm_cell_a = tf.contrib.rnn.LSTMCell(num_units=int(self.bi_rnn_text_embedding / 2), state_is_tuple=True)
            outputs, states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=fw_lstm_cell_a,
                cell_bw=bw_lstm_cell_a,
                dtype=tf.float32,
                inputs=a)

        a = tf.concat(outputs, 2)
        print('###: textEmbedding3:', a.get_shape().as_list())

        a = tf.layers.dense(inputs=a, units=self.final_linear_projection_embedding, activation=None, trainable=True,
                            name='second_ad_linear')
        print('###: textEmbedding4:', a.get_shape().as_list())

        print('### Learning attention for the text...')
        attentionA = tf.layers.dense(inputs=a, units=self.final_linear_projection_embedding, activation=None,
                                     trainable=True,
                                     name='attention_model_text_l1')
        attentionA = tf.layers.dense(inputs=attentionA, units=1, activation=None, trainable=True,
                                     name='attention_model_text_l2')
        attentionA = tf.squeeze(attentionA, 2)
        attentionA = tf.nn.softmax(attentionA, dim=1)
        print('### Attention model layer:', attentionA.get_shape().as_list())

        aDense = tf.matmul(tf.transpose(a, [0, 2, 1]), tf.expand_dims(attentionA, -1))
        aDense = tf.squeeze(aDense, -1)
        print('### Ad representation (after attention):', aDense.get_shape().as_list())

        qA = aDense
        if self.keep_prob < 1:
            # qA = tf.nn.dropout(qA, self.keep_prob)
            qA = tf.layers.dense(inputs=qA, units=(self.final_linear_projection_embedding * 4),
                                 activation=tf.nn.relu, trainable=True, use_bias=True)
            # qA = tf.nn.dropout(qA, self.keep_prob)
            print('### Dimensionality of the text after first dense layer:', qA.get_shape().as_list())
            qA = tf.layers.dense(inputs=qA, units=(self.final_linear_projection_embedding * 4),
                                 activation=tf.nn.relu, trainable=True, use_bias=True)
            qA = tf.nn.dropout(qA, self.keep_prob)
            print('### Dimensionality of the text after second layer:', qA.get_shape().as_list())
            qA = tf.layers.dense(inputs=qA, units=1, activation=None, trainable=True, use_bias=True)
            print('### Dimensionality of the text after dense layers:', qA.get_shape().as_list())
        elif self.keep_prob == 1:
            qA = tf.layers.dense(inputs=qA, units=(self.final_linear_projection_embedding * 4),
                                 activation=tf.nn.relu, trainable=True,
                                 use_bias=True, bias_regularizer=tf.nn.l2_loss, kernel_regularizer=tf.nn.l2_loss)
            print('### Dimensionality of the text after first dense layer:', qA.get_shape().as_list())
            qA = tf.layers.dense(inputs=qA, units=(self.final_linear_projection_embedding * 4),
                                 activation=tf.nn.relu, trainable=True,
                                 use_bias=True, bias_regularizer=tf.nn.l2_loss, kernel_regularizer=tf.nn.l2_loss)
            print('### Dimensionality of the text after second layer:', qA.get_shape().as_list())
            qA = tf.layers.dense(inputs=qA, units=self.number_of_classes, activation=None, trainable=True,
                                 use_bias=True, bias_regularizer=tf.nn.l2_loss, kernel_regularizer=tf.nn.l2_loss)
            print('### Dimensionality of the text after dense layers:', qA.get_shape().as_list())

        # --------------------------------------------------------------------------------
        y = tf.squeeze(qA)

        return y, attentionA

    def build_loss_ops(self):
        """
        Loss
        """
        if self.number_of_classes == 1:
            loss_op = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.inference_op, labels=self.labelPlaceHolder))
        else:
            loss_op = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=self.inference_op, labels=self.labelPlaceHolder))
        return loss_op

    def build_opt_ops(self):
        train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss_op)
        return train_op

    def build_predict_ops(self):
        if self.number_of_classes == 1:
            predict_op = tf.nn.sigmoid(self.inference_op)
        else:
            predict_op = tf.nn.softmax(self.inference_op)
        return predict_op

    def batch_fit(self, text_ids, labels):
        feed_dict = {self.textPlaceHolder: text_ids,
                     self.labelPlaceHolder: labels}

        loss, _ = self._sess.run([self.loss_op, self.train_op], feed_dict=feed_dict)

        return loss

    def predict(self, text_ids, labels):
        feed_dict = {self.textPlaceHolder: text_ids,
                     self.labelPlaceHolder: labels}

        loss, predict = self._sess.run([self.loss_op, self.predict_op], feed_dict=feed_dict)

        return loss, predict

    def retrieve_attentions(self, text_ids, labels):
        feed_dict = {self.textPlaceHolder: text_ids,
                     self.labelPlaceHolder: labels}
        attentions = self._sess.run(self.attentions, feed_dict=feed_dict)
        return attentions


class DeepAndWideTextAttentionModel(object):
    """ Deep Attention Model for Text Classification. """
    def __init__(self,
                 session,
                 batch_size=64,
                 embedding_size=256,
                 bi_rnn_text_embedding=128,
                 linear_projection_embedding=128,
                 final_linear_projection_embedding=128,
                 vocabulary_size=1000,
                 number_of_words=35,
                 total_number_of_features=50,
                 weights_init_strategy=tf.contrib.layers.variance_scaling_initializer(),
                 keep_probability=0.5,
                 learning_rate=0.001,
                 number_of_classes=1,
                 phase_train=True):

        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.bi_rnn_text_embedding = bi_rnn_text_embedding
        self.linear_projection_embedding = linear_projection_embedding
        self.final_linear_projection_embedding = final_linear_projection_embedding
        self.number_of_words = number_of_words
        self.vocabulary_size = vocabulary_size
        self.total_number_of_features = total_number_of_features
        self.weights_init_strategy = weights_init_strategy
        self.weights_init_strategy = tf.contrib.layers.variance_scaling_initializer()
        self.keep_prob = keep_probability
        self.learning_rate = learning_rate
        self.number_of_classes = number_of_classes
        self.phase_train = phase_train

        # self._initializer = tf.random_normal_initializer(stddev=0.1)
        self._initializer = tf.truncated_normal
        # if options['initializer'] == 0 else tf.random_uniform_initializer(minval=-0.01, maxval=0.01)

        self._build_params()
        self._build_inputs()

        self.inference_op, self.attentions = self.build_inference_ops()
        self.loss_op = self.build_loss_ops()
        self.train_op = self.build_opt_ops()
        self.predict_op = self.build_predict_ops()

        self._init_op = tf.global_variables_initializer()
        self._sess = session
        self._sess.run(self._init_op)

        self._saver = tf.train.Saver()

    def _build_inputs(self):
        self.textPlaceHolder = tf.placeholder(tf.int32, shape=[None, self.number_of_words])
        self.binaryFeaturesPlaceHolder = tf.placeholder(tf.float32, shape=[None, self.total_number_of_features])
        if self.number_of_classes > 1:
            self.labelPlaceHolder = tf.placeholder(tf.float32, shape=[None, self.number_of_classes])
        else:
            self.labelPlaceHolder = tf.placeholder(tf.float32, shape=[None, ])

    def _build_params(self):
        self.word_embeddings = tf.Variable(self._initializer([self.vocabulary_size, self.embedding_size]), name='W_emb')

    def build_inference_ops(self):
        """
        Forward pass of the model
        """
        print('###: textPlaceHolder:', self.textPlaceHolder.get_shape().as_list())
        a = tf.nn.embedding_lookup(self.word_embeddings, self.textPlaceHolder)
        print('###: textEmbedding1:', a.get_shape().as_list())

        # a = tf.nn.relu(a) # TODO: should we use a dense layer here or just use ReLU
        a = tf.layers.dense(inputs=a, units=self.linear_projection_embedding, activation=tf.nn.relu, trainable=True,
                            name='joint_linear_projection')
        print('###: textEmbedding2:', a.get_shape().as_list())

        with tf.variable_scope('birnn_text'):
            fw_lstm_cell_a = tf.contrib.rnn.LSTMCell(num_units=int(self.bi_rnn_text_embedding / 2), state_is_tuple=True)
            bw_lstm_cell_a = tf.contrib.rnn.LSTMCell(num_units=int(self.bi_rnn_text_embedding / 2), state_is_tuple=True)
            outputs, states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=fw_lstm_cell_a,
                cell_bw=bw_lstm_cell_a,
                dtype=tf.float32,
                inputs=a)

        a = tf.concat(outputs, 2)
        print('###: textEmbedding3:', a.get_shape().as_list())

        a = tf.layers.dense(inputs=a, units=self.final_linear_projection_embedding, activation=None, trainable=True,
                            name='second_ad_linear')
        print('###: textEmbedding4:', a.get_shape().as_list())

        print('### Learning attention for the text...')
        attentionA = tf.layers.dense(inputs=a, units=self.final_linear_projection_embedding, activation=None,
                                     trainable=True,
                                     name='attention_model_text_l1')
        attentionA = tf.layers.dense(inputs=attentionA, units=1, activation=None, trainable=True,
                                     name='attention_model_text_l2')
        attentionA = tf.squeeze(attentionA, 2)
        attentionA = tf.nn.softmax(attentionA, dim=1)
        print('### Attention model layer:', attentionA.get_shape().as_list())

        aDense = tf.matmul(tf.transpose(a, [0, 2, 1]), tf.expand_dims(attentionA, -1))
        aDense = tf.squeeze(aDense, -1)
        print('### Ad representation (after attention):', aDense.get_shape().as_list())

        qA = aDense
        if self.keep_prob < 1:
            # qA = tf.nn.dropout(qA, self.keep_prob)
            qA = tf.layers.dense(inputs=qA, units=(self.final_linear_projection_embedding * 4),
                                 activation=tf.nn.relu, trainable=True, use_bias=True)
            # qA = tf.nn.dropout(qA, self.keep_prob)
            print('### Dimensionality of the text after first dense layer:', qA.get_shape().as_list())

            qA = tf.concat([qA, self.binaryFeaturesPlaceHolder], 1)
            print('### Dimensionality of the text after join with binary features:', qA.get_shape().as_list())

            qA = tf.layers.dense(inputs=qA, units=(self.final_linear_projection_embedding * 4),
                                 activation=tf.nn.relu, trainable=True, use_bias=True)
            qA = tf.nn.dropout(qA, self.keep_prob)
            print('### Dimensionality of the text after second layer:', qA.get_shape().as_list())
            qA = tf.layers.dense(inputs=qA, units=self.number_of_classes, activation=None, trainable=True, use_bias=True)
            print('### Dimensionality of the text after dense layers:', qA.get_shape().as_list())
        elif self.keep_prob == 1:
            qA = tf.layers.dense(inputs=qA, units=(self.final_linear_projection_embedding * 4),
                                 activation=tf.nn.relu, trainable=True,
                                 use_bias=True, bias_regularizer=tf.nn.l2_loss, kernel_regularizer=tf.nn.l2_loss)
            print('### Dimensionality of the text after first dense layer:', qA.get_shape().as_list())

            qA = tf.concat([qA, self.binaryFeaturesPlaceHolder], 1)
            print('### Dimensionality of the text after join with binary features:', qA.get_shape().as_list())

            qA = tf.layers.dense(inputs=qA, units=(self.final_linear_projection_embedding * 4),
                                 activation=tf.nn.relu, trainable=True,
                                 use_bias=True, bias_regularizer=tf.nn.l2_loss, kernel_regularizer=tf.nn.l2_loss)
            print('### Dimensionality of the text after second layer:', qA.get_shape().as_list())
            qA = tf.layers.dense(inputs=qA, units=self.number_of_classes, activation=None, trainable=True,
                                 use_bias=True, bias_regularizer=tf.nn.l2_loss, kernel_regularizer=tf.nn.l2_loss)
            print('### Dimensionality of the text after dense layers:', qA.get_shape().as_list())

        # --------------------------------------------------------------------------------
        # y = tf.reshape(qA, [-1]) # TODO: check if this works for multiclass
        y = tf.squeeze(qA)

        return y, attentionA

    def build_loss_ops(self):
        """
        Loss
        """
        if self.number_of_classes == 1:
            loss_op = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.inference_op, labels=self.labelPlaceHolder))
        else:
            loss_op = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=self.inference_op, labels=self.labelPlaceHolder))
        return loss_op

    def build_opt_ops(self):
        train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss_op)
        return train_op

    def build_predict_ops(self):
        if self.number_of_classes == 1:
            predict_op = tf.nn.sigmoid(self.inference_op)
        else:
            predict_op = tf.nn.softmax(self.inference_op)
        return predict_op

    def batch_fit(self, text_ids, binary_features, labels):
        feed_dict = {self.textPlaceHolder: text_ids,
                     self.binaryFeaturesPlaceHolder: binary_features,
                     self.labelPlaceHolder: labels}

        loss, _ = self._sess.run([self.loss_op, self.train_op], feed_dict=feed_dict)

        return loss

    def predict(self, text_ids, binary_features, labels):
        feed_dict = {self.textPlaceHolder: text_ids,
                     self.binaryFeaturesPlaceHolder: binary_features,
                     self.labelPlaceHolder: labels}

        loss, predict = self._sess.run([self.loss_op, self.predict_op], feed_dict=feed_dict)

        return loss, predict

    def retrieve_attentions(self, text_ids, binary_features, labels):
        feed_dict = {self.textPlaceHolder: text_ids,
                     self.binaryFeaturesPlaceHolder: binary_features,
                     self.labelPlaceHolder: labels}
        attentions = self._sess.run(self.attentions, feed_dict=feed_dict)
        return attentions


class DeepAndWideMatch(object):
    """ Tensor Match model """

    def __init__(self,
                 session,
                 batch_size=6,
                 wide_part_input_size=26777,
                 img_width=82,
                 img_height=82,
                 wide_w1_size=100,
                 wide_b1_size=100,
                 wide_w2_size=100,
                 wide_b2_size=100,
                 cnn_base_factor=64,
                 vocabulary_size=10000,
                 query_embedding_size=256,
                 ad_embedding_size=256,
                 number_of_words_for_query=35,
                 number_of_words_for_ad=140,
                 query_and_ad_joint_linear_projection=40,
                 query_and_ad_final_linear_projection=50,
                 bi_rnn_query_embedding=30,
                 bi_rnn_ad_embedding=140,
                 weights_init_strategy=tf.contrib.layers.variance_scaling_initializer(),
                 weights_init_strategy_string='truncated_normal',
                 keep_probability=0.5,
                 learning_rate=0.001,
                 phase_train=True):

        self.batch_size = batch_size

        self.wide_part_input_size = wide_part_input_size
        self.img_width = img_width
        self.img_height = img_height

        # hyperparameters
        self.w1_size = wide_w1_size
        self.b1_size = wide_b1_size
        self.w2_size = wide_w2_size
        self.b2_size = wide_b2_size

        self.cnn_base_factor = cnn_base_factor

        self.vocabulary_size = vocabulary_size
        self.number_of_words_for_query = number_of_words_for_query
        self.number_of_words_for_ad = number_of_words_for_ad
        self.query_embedding_size = query_embedding_size
        self.ad_embedding_size = ad_embedding_size
        self.query_and_ad_joint_linear_projection = query_and_ad_joint_linear_projection
        self.query_and_ad_final_linear_projection = query_and_ad_final_linear_projection
        self.bi_rnn_query_embedding = bi_rnn_query_embedding
        self.bi_rnn_ad_embedding = bi_rnn_ad_embedding
        self.weights_init_strategy = weights_init_strategy
        self.weights_init_strategy = tf.contrib.layers.variance_scaling_initializer()
        self.weights_init_strategy_string = weights_init_strategy_string
        self.keep_prob = keep_probability
        self.learning_rate = learning_rate
        self.phase_train = phase_train

        self._initializer = tf.truncated_normal  # tf.random_normal_initializer(stddev=0.1)

        self._build_params()
        self._build_inputs()

        self.inference_op = self.build_inference_ops()
        self.loss_op = self.build_loss_ops()
        self.train_op = self.build_opt_ops()
        self.predict_op = self.build_predict_ops()

        init_op = tf.global_variables_initializer()
        self._sess = session
        self._sess.run(init_op)

        self._saver = tf.train.Saver()


    def _build_inputs(self):
        self.wideFeaturesPlaceholder = tf.sparse_placeholder(tf.float32)
        self.adTextPlaceHolder = tf.placeholder(tf.int32, shape=[None, self.number_of_words_for_ad])
        self.adImagePlaceHolder = tf.placeholder(tf.int32, shape=[None, self.img_width, self.img_height])
        self.labelPlaceHolder = tf.placeholder(tf.float32, shape=[None, ])

        self.keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)
        self.phase_train = tf.placeholder(tf.bool, name='phase_train')

    def _build_params(self):
        self.W_wide_1 = tf.Variable(self._initializer([self.wide_part_input_size, self.w1_size]), name='W_wide_1')
        self.b_wide_1 = tf.Variable(self._initializer([self.b1_size]), name='b_wide_1')
        self.W_wide_2 = tf.Variable(self._initializer([self.w1_size, self.w2_size]), name='W_wide_2')
        self.b_wide_2 = tf.Variable(self._initializer([self.b2_size]), name='b_wide_2')

        self.ad_embeddings = tf.Variable(
            self._initializer([self.vocabulary_size, self.ad_embedding_size]), name='ad_emb')
        self.query_embeddings = tf.Variable(
            self._initializer([self.vocabulary_size, self.query_embedding_size]), name='query_emb')
        self.alpha = tf.Variable(self._initializer([1]), name='alpha')

    def build_inference_ops(self):

        ################ wide part
        with tf.name_scope('ad_wide_features_embedding'):
            wide = tf.add(tf.sparse_tensor_dense_matmul(self.wideFeaturesPlaceholder, self.W_wide_1), self.b_wide_1)
            wide = tf.nn.relu(wide)
            wide = tf.add(tf.matmul(wide, self.W_wide_2), self.b_wide_2)
            wide = tf.nn.relu(wide)
            print('###: wideEmbedding:', wide.get_shape().as_list())

        ################ Ad text embedding
        with tf.name_scope('ad_text_embedding'):
            a_text = tf.nn.embedding_lookup(self.ad_embeddings, self.adTextPlaceHolder)
            print('###: adEmbedding1:', a_text.get_shape().as_list())

            a_text = tf.layers.dense(inputs=a_text, units=self.query_and_ad_joint_linear_projection, activation=None, trainable=True,
                                name='joint_linear_projection')
            print('###: adEmbedding2:', a_text.get_shape().as_list())

            with tf.variable_scope('birnn_ad'):
                fw_lstm_cell_a = tf.contrib.rnn.LSTMCell(num_units=self.bi_rnn_ad_embedding / 2, state_is_tuple=True)
                bw_lstm_cell_a = tf.contrib.rnn.LSTMCell(num_units=self.bi_rnn_ad_embedding / 2, state_is_tuple=True)
                outputs, states = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=fw_lstm_cell_a,
                    cell_bw=bw_lstm_cell_a,
                    dtype=tf.float32,
                    inputs=a_text)

            a_text = tf.concat(outputs, 2)
            print('###: adEmbedding3:', a_text.get_shape().as_list())

            a_text = tf.layers.dense(inputs=a_text, units=self.query_and_ad_final_linear_projection, activation=None, trainable=True,
                                name='second_ad_linear')
            print('###: adEmbedding4:', a_text.get_shape().as_list())

        ################ Ad image embedding
        with tf.name_scope('ad_image_embedding'):
            with tf.name_scope('conv_layer_1'):
                a_image = tf.layers.conv2d(inputs=self.adImagePlaceHolder, kernel_size=[3, 3], filters=self.cnn_base_factor,
                                           strides=1, padding='same', data_format='channels_last')
                # a_image = tf.nn.conv2d(input=self.adImagePlaceHolder, filter=self.conv1_weights, strides=[1, 1, 1, 1], padding='same', name='conv1')
                print('### First 2D convolution (3x3):', a_image.get_shape().as_list())
                
                a_image = tf.nn.relu(a_image)
                print('### After first ReLU:', a_image.get_shape().as_list())
                
                a_image = tf.layers.conv2d(inputs=a_image, kernel_size=[3, 3], filters=self.cnn_base_factor,
                                           strides=1, padding='same')
                # a_image = tf.nn.conv2d(input=a_image, filter=self.conv1_weights, strides=[1, 1, 1, 1], padding='same', name='conv1')
                print('### First 2D convolution (3x3):', a_image.get_shape().as_list())
                
                a_image = tf.nn.relu(a_image)
                print('### After first ReLU:', a_image.get_shape().as_list())
                
                a_image = tf.layers.max_pooling2d(inputs=a_image, pool_size=[2, 2], strides=2,
                                                  data_format='channels_last', padding='valid')

            with tf.name_scope('conv_layer_2'):
                a_image = tf.layers.conv2d(inputs=a_image, kernel_size=[3, 3], filters=self.cnn_base_factor,
                                           strides=1, padding='same')
                # a_image = tf.nn.conv2d(input=a_image, filter=self.conv1_weights, strides=[1, 1, 1, 1], padding='same', name='conv1')
                print('### First 2D convolution (3x3):', a_image.get_shape().as_list())

                a_image = tf.nn.relu(a_image)
                print('### After first ReLU:', a_image.get_shape().as_list())

                a_image = tf.layers.conv2d(inputs=a_image, kernel_size=[3, 3], filters=self.cnn_base_factor,
                                           strides=1, padding='same')
                # a_image = tf.nn.conv2d(input=a_image, filter=self.conv1_weights, strides=[1, 1, 1, 1], padding='same', name='conv1')
                print('### First 2D convolution (3x3):', a_image.get_shape().as_list())

                a_image = tf.nn.relu(a_image)
                print('### After first ReLU:', a_image.get_shape().as_list())

                a_image = tf.layers.max_pooling2d(inputs=a_image, pool_size=[2, 2], strides=2,
                                                  data_format='channels_last', padding='valid')

            with tf.name_scope('conv_layer_3'):
                a_image = tf.layers.conv2d(inputs=a_image, kernel_size=[3, 3], filters=self.cnn_base_factor,
                                           strides=1, padding='same')
                # a_image = tf.nn.conv2d(input=a_image, filter=self.conv1_weights, strides=[1, 1, 1, 1], padding='same', name='conv1')
                print('### First 2D convolution (3x3):', a_image.get_shape().as_list())

                a_image = tf.nn.relu(a_image)
                print('### After first ReLU:', a_image.get_shape().as_list())

                a_image = tf.layers.conv2d(inputs=a_image, kernel_size=[3, 3], filters=self.cnn_base_factor,
                                           strides=1, padding='same')
                # a_image = tf.nn.conv2d(input=a_image, filter=self.conv1_weights, strides=[1, 1, 1, 1], padding='same', name='conv1')
                print('### First 2D convolution (3x3):', a_image.get_shape().as_list())

                a_image = tf.nn.relu(a_image)
                print('### After first ReLU:', a_image.get_shape().as_list())

                a_image = tf.layers.max_pooling2d(inputs=a_image, pool_size=[2, 2], strides=2,
                                                  data_format='channels_last', padding='valid')


            with tf.name_scope('conv_scale_layer'):
                a_image = tf.layers.conv2d(inputs=a_image, kernel_size=[1, 1], filters=20, strides=1, padding='same')
                print('### Another 2D convolution (1x1):', a_image.get_shape().as_list())
                a_image = tf.nn.relu(a_image)
                print('### After second ReLU:', a_image.get_shape().as_list())

                a_image = tf.layers.max_pooling2d(inputs=a_image, pool_size=[3, 3], strides=1,
                                             data_format='channels_last', padding='valid')
                a_image = tf.squeeze(a_image, [1, 2])
                print('### Max-pool:', a_image.get_shape().as_list())

            with tf.name_scope('dense_layer'):
                # n_inputs = int(np.prod(qA.get_shape().as_list()[1:]))
                # qA = tf.reshape(qA, [-1, n_inputs])
                a_image = tf.layers.dense(inputs=a_image, units=1, activation=None, trainable=True,
                                     name='ad_image_final_linear_projection')


        ################ Ad text vs. image attention


        ################ Merging part


        ################ Ater merge part
        qA = [] # TODO: don't forget to remove this line later


        # TODO: there will probably be a need for more feature learning with CNN after the merge!


        n_inputs = int(np.prod(qA.get_shape().as_list()[1:]))
        qA = tf.reshape(qA, [-1, n_inputs])
        qA = tf.layers.dense(inputs=qA, units=(self.base_factor * 4), activation=tf.nn.relu, trainable=True)
        print('### Dimensionality of the pair after first dense layer:', qA.get_shape().as_list())
        n_inputs = int(np.prod(qA.get_shape().as_list()[1:]))
        qA = tf.reshape(qA, [-1, n_inputs])
        qA = tf.layers.dense(inputs=qA, units=(self.base_factor * 4), activation=tf.nn.relu, trainable=True)
        print('### Dimensionality of the pair after second layer:', qA.get_shape().as_list())
        n_inputs = int(np.prod(qA.get_shape().as_list()[1:]))
        qA = tf.reshape(qA, [-1, n_inputs])
        qA = tf.layers.dense(inputs=qA, units=1, activation=None, trainable=True)
        print('### Dimensionality of the pair after dense layers:', qA.get_shape().as_list())

        ################ Logits
        # logits = tf.reshape(qA, [-1])
        logits = tf.squeeze(qA)

        return logits

    def build_loss_ops(self):
        """
        Loss
        """
        loss_op = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.inference_op, labels=self.labelPlaceHolder))
        return loss_op

    def build_opt_ops(self):
        train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss_op)
        return train_op

    def build_predict_ops(self):
        predict_op = tf.nn.sigmoid(self.inference_op)
        return predict_op

    def batch_fit(self, wide_features, ads_text, ads_images, labels):
        feed_dict = {self.wideFeaturesPlaceholder: wide_features,
                     self.adTextPlaceHolder: ads_text,
                     self.adImagePlaceHolder: ads_images,
                     self.labelPlaceHolder: labels}

        loss, _ = self._sess.run([self.loss_op, self.train_op], feed_dict=feed_dict)

        return loss

    def predict(self, wide_features, ads_text, ads_images, labels):
        feed_dict = {self.wideFeaturesPlaceholder: wide_features,
                     self.adTextPlaceHolder: ads_text,
                     self.adImagePlaceHolder: ads_images,
                     self.labelPlaceHolder: labels}

        loss, predict = self._sess.run([self.loss_op, self.predict_op], feed_dict=feed_dict)

        return loss, predict

