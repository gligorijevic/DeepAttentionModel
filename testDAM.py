import argparse
import os

import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.python.client import device_lib
from sklearn.metrics import roc_auc_score


import dataProcess as dp
import resultsAnalysis as ra
from DeepTextModels import DeepTextAttentionModel, DeepAndWideTextAttentionModel

def predict_ER_patient_resources(testing_file_path="testing_file.txt",
                              pred_save_path="pred_file.txt",
                              vocabulary_local_path="vocab.txt",
                              gpu_to_use="cpu",
                              model_local_path="model_file.txt",
                              learning_rate=0.001,
                              number_of_epochs=100,
                              batchSize = 64,
                              numberOfWords = 40,
                              baseFactor = 64,
                              textEmbedding = 256,
                              bi_rnn_text_embedding=128,
                              linear_projection_embedding=128,
                              final_linear_projection_embedding=128,
                              mini_batch_display_freq=10,
                              model_save_frequency=10,
                              model_save_path="save_model.mdl",
                              n_classes=1,
                              n_features=0,
                              skip_stopwords=False,
                              verbose=False):

    print(device_lib.list_local_devices())
    if gpu_to_use == 'cpu':
        os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES = '100000'
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES = gpu_to_use

    weights_init_strategy = 'variance_scaling'

    if not os.path.exists(model_local_path):
        print('No model to resore. Ending..')
        return

    # load dictionary first
    dictationary_filepath = vocabulary_local_path + 'amazon_reviews_text_dictionary'
    dp.loadVocabulary(dictationary_filepath)

    skip_stopwords = True
    print('Total number of tokens is:', len(dp.word_vocabulary))

    print('Loading test data...')
    [testingTextTensor, testingLabelMulti] = dp.loadDEMdataWords(testing_file_path, numberOfWords)
    testingBinaryFeaturesTensor = []
    testingLabelBinary = []

    if n_classes == 1:
        testingLabel = testingLabelBinary
    else:
        testingLabel = testingLabelMulti

    testSetSize = len(testingTextTensor)

    keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)
    phase_train = tf.placeholder(tf.bool, name='phase_train')

    # this is a must for restoring graphs
    tf.reset_default_graph()

    config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
    sess = tf.Session(config=config)

    if n_features == 0:
        damModel = DeepTextAttentionModel(session=sess,
                        batch_size=batchSize,
                        vocabulary_size=len(dp.word_vocabulary) + 1,
                        number_of_words=numberOfWords,
                        embedding_size=textEmbedding,
                        bi_rnn_text_embedding=bi_rnn_text_embedding,
                        linear_projection_embedding=linear_projection_embedding,
                        final_linear_projection_embedding=final_linear_projection_embedding,
                        weights_init_strategy=weights_init_strategy,
                        keep_probability=1,
                        number_of_classes=n_classes,
                        learning_rate=learning_rate,
                        phase_train=True)
    else:
        damModel = DeepAndWideTextAttentionModel(session=sess,
                        batch_size=batchSize,
                        vocabulary_size=len(dp.word_vocabulary) + 1,
                        number_of_words=numberOfWords,
                        embedding_size=textEmbedding,
                        bi_rnn_text_embedding=bi_rnn_text_embedding,
                        linear_projection_embedding=linear_projection_embedding,
                        final_linear_projection_embedding=final_linear_projection_embedding,
                        total_number_of_features=n_features,
                        weights_init_strategy=weights_init_strategy,
                        keep_probability=1,
                        number_of_classes=n_classes,
                        learning_rate=learning_rate,
                        phase_train=True)

    damModel._saver.restore(sess, tf.train.latest_checkpoint(model_local_path))
    print('Model', model_local_path, 'restored.')

    numOfBatch = 0
    numOfEpoch = 0

    lossValAverage_test_list = []
    predicts_test_list = []
    lossValAverage_train_list = []
    predicts_train_list = []
    for start, end in dp.batchIndexesGenerator(batchSize, testSetSize):
        numOfBatch = numOfBatch + 1

        if n_classes == 1:
            empty_labels = np.zeros(end - start)
        else:
            empty_labels = np.zeros([end - start, n_classes])

        if n_features == 0:
            _, predicts_test = damModel.predict(dp.tensorizeText(testingTextTensor[start:end]),
                                                  empty_labels)
        else:
            _, predicts_test = damModel.predict(dp.tensorizeText(testingTextTensor[start:end]),
                                                  testingBinaryFeaturesTensor[start:end],
                                                  empty_labels)


        predicts_test_list = predicts_test_list + predicts_test.tolist()

    damModel._sess.close()
    if not os.path.exists(pred_save_path):
        os.makedirs(pred_save_path)

    if n_classes == 1:
        ra.getResultsBinary(testingLabel, predicts_test_list, pred_save_path)
        auc_test = roc_auc_score(testingLabel, predicts_test_list)
        print('Testing AUC score is:', auc_test)
    else:
        ra.getResultsMulti(testingLabel, np.asarray(predicts_test_list), pred_save_path)

    dataDf = {}
    dataDf['notes'] = testingTextTensor
    dataDf['labels'] = testingLabel
    dataDf['predictions'] = predicts_test_list
    dataDf = pd.DataFrame(data=dataDf)
    dataDf.columns = ['notes', 'labels', 'predictions']


    dataDf.to_csv(pred_save_path+'predictions.txt', header=None, index=False, sep='\t')
    print('Predictions are save to: ' + str(pred_save_path))


def parse_arguments(parser):
    parser.add_argument('--training_file_path', type=str,
                        help='The path to the training file')
    parser.add_argument('--validation_file_path', type=str,
                        help='The path to the validation file')
    parser.add_argument('--testing_file_path', type=str,
                        help='The path to the testing file')
    parser.add_argument('--model_save_path', type=str,
                        help='The path to the output models')
    parser.add_argument('--vocabulary_local_path', type=str, metavar='<out_file>',
                        help='The path to the vocabulary')
    parser.add_argument('--predictions_save_path', type=str, metavar='<out_file>',
                        help='The path to the predictions')
    parser.add_argument("--gpu_to_use", type=str,
                        help="ID of the GPU that will be used", default="cpu")
    parser.add_argument("--learning_rate", type=float, default=1e-8,
                        help="Learning rate for the optimization (default value: 1e-8)")
    parser.add_argument('--number_of_epochs', type=int, default=100,
                        help='The number of training epochs (default value: 100)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='The number of batch examples (default value: 64)')
    parser.add_argument("--metric_display_freq", type=int, default=100,
                        help="Metric display frequency (Default value: 100)")
    parser.add_argument("--model_save_frequency", type=int, default=250,
                        help="Model save frequency (Default value: 250)")
    parser.add_argument('--n_classes', type=int, default=1,
                        help='Number of classes to predict (Default value: 1)')
    parser.add_argument('--n_features', type=int, default=0,
                        help='Number of features to use for the wide part of the model (Default value: 0)')
    parser.add_argument('--n_chars', type=int, default=3000,
                        help='The number of words for each query and ad pair (Default value: 45)')
    parser.add_argument('--n_words', type=int, default=45,
                        help='The number of words for each query and ad pair (Default value: 45)')
    parser.add_argument('--n_conv_filters', type=int, default=64,
                        help='Number of convolutional filter for the middle layer of TensorMatch (Default value: 6)')
    parser.add_argument('--textEmbedding', type=int, default=256,
                        help='Embedding size of the ad words (Default value: 256)')
    parser.add_argument('--biRnnEmbedding', type=int, default=128,
                        help='Embedding size of the words for biRnn (Default value: 128)')
    parser.add_argument('--linearProjectionEmbedding', type=int, default=128,
                        help='Embedding size of the first linear layer (Default value: 128)')
    parser.add_argument('--finalLinearProjectionEmbedding', type=int, default=128,
                        help='Embedding size of the final word embedding (Default value: 128)')
    parser.add_argument("--skip_stopwords", help="Skip stopwords from queries and ads", action='store_true')
    parser.add_argument("--verbose", help="Print output after every 10 mini-batches", action='store_true')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test DeepMatch")
    args = parse_arguments(parser)

    print('Starting...')
    predict_ER_patient_resources(testing_file_path=args.testing_file_path,
                                 model_local_path=args.model_save_path,
                                 pred_save_path=args.predictions_save_path,
                                 vocabulary_local_path=args.vocabulary_local_path,
                                 gpu_to_use=args.gpu_to_use,
                                 learning_rate=args.learning_rate,
                                 number_of_epochs=args.number_of_epochs,
                                 batchSize=args.batch_size,
                                 mini_batch_display_freq=args.metric_display_freq,
                                 model_save_frequency=args.model_save_frequency,
                                 numberOfWords=args.n_words,
                                 textEmbedding=args.textEmbedding,
                                 bi_rnn_text_embedding=args.biRnnEmbedding,
                                 linear_projection_embedding=args.linearProjectionEmbedding,
                                 final_linear_projection_embedding=args.finalLinearProjectionEmbedding,
                                 n_classes=args.n_classes,
                                 n_features=args.n_features,
                                 skip_stopwords=args.skip_stopwords,
                                 verbose=args.verbose)
