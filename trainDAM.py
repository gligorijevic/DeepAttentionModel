import argparse
import os
import sys

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

import dataProcess as dp
from DeepTextModels import DeepTextAttentionModel, DeepAndWideTextAttentionModel


def predict_ER_patient_resources(training_file_path="training_file.txt",
                              test_file_path="test_file.txt",
                              vocabulary_local_path="",
                              gpu_to_use="cpu",
                              learning_rate=0.001,
                              number_of_epochs=100,
                              batchSize = 64,
                              numberOfWords = 40,
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

    if gpu_to_use == 'cpu':
        os.environ["CUDA_VISIBLE_DEVICES"]=CUDA_VISIBLE_DEVICES='100000'
    else:
        os.environ["CUDA_VISIBLE_DEVICES"]=CUDA_VISIBLE_DEVICES=gpu_to_use

    weights_init_strategy = 'variance_scaling'

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    if not os.path.exists(vocabulary_local_path):
        os.makedirs(vocabulary_local_path)

    if skip_stopwords:
        dp.loadStopwords()
    dp.learnVocabulary(training_file_path+'/amazon_reviews_text', skip_stopwords)
    dp.enrichVocabulary(test_file_path+'/amazon_reviews_text', skip_stopwords)
    dp.cleanAndBuildVocabulary(word_frequency_threshold=5)
    dictationary_filepath = vocabulary_local_path + 'amazon_reviews_text_dictionary'
    dp.saveVocabulary(dictationary_filepath)
    print('Total number of tokens is:', len(dp.word_vocabulary))

    print('Loading training data...')
    [trainingTextTensor, trainingLabelMulti] = \
        dp.loadDEMdataWords(training_file_path, numberOfWords)
    # to use additional features and include binary classification please use dp.loadDEMdataWordsAndNumerical() function
    trainingLabelBinary = []
    trainingBinaryFeaturesTensor = []
    print('Loading test data...')
    [testTextTensor, testLabelMulti] = \
        dp.loadDEMdataWords(test_file_path, numberOfWords)
    testLabelBinary = []
    testBinaryFeaturesTensor = []

    if n_classes == 1:
        trainingLabel = trainingLabelBinary
        testLabel = testLabelBinary
    else:
        trainingLabel = trainingLabelMulti
        testLabel = testLabelMulti

    trainingSetSize = len(trainingTextTensor)
    testSetSize = len(testTextTensor)

    print('Initializing Tensorflow session...')
    tf.reset_default_graph()
    sess = tf.Session()
    logList = []

    numOfBatch = 0
    numOfEpoch = 0

    print('Training Started')
    print('numOfEpoch,numOfBatch,loss_train,loss_test,auc_train,auc_test')

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

    for i in range(number_of_epochs):
        numOfEpoch = numOfEpoch + 1

        for start, end in dp.batchIndexesGenerator(batchSize, trainingSetSize):

            numOfBatch = numOfBatch + 1

            if verbose:
                print('Training: epoch={:d}, batch={:d}'.format(i, numOfBatch))

            # training
            if n_features == 0:
                damModel.batch_fit(dp.tensorizeText(trainingTextTensor[start:end]),
                                   trainingLabel[start:end])
            else:
                damModel.batch_fit(dp.tensorizeText(trainingTextTensor[start:end]),
                                   trainingBinaryFeaturesTensor[start:end],
                                   trainingLabel[start:end])

            # monitoring
            if numOfBatch % mini_batch_display_freq == 0:
                lossValAverage_test_list = []
                predicts_test_list = []
                lossValAverage_train_list = []
                predicts_train_list = []

                for start, end in dp.batchIndexesGenerator(batchSize, trainingSetSize):
                    if n_features == 0:
                        lossValAverage_train, predicts_train = damModel.predict(
                                                                        dp.tensorizeText(trainingTextTensor[start:end]),
                                                                        trainingLabel[start:end])
                    else:
                        lossValAverage_train, predicts_train = damModel.predict(
                                                                        dp.tensorizeText(trainingTextTensor[start:end]),
                                                                        trainingBinaryFeaturesTensor[start:end],
                                                                        trainingLabel[start:end])

                    lossValAverage_train_list.append(lossValAverage_train)
                    predicts_train_list = predicts_train_list + predicts_train.tolist()

                loss_train = np.array(lossValAverage_train_list).mean()
                if n_classes == 1:
                    auc_train = roc_auc_score(trainingLabel, predicts_train_list)


                for start, end in dp.batchIndexesGenerator(batchSize, testSetSize):
                    if n_features == 0:
                        lossValAverage_test, predicts_test = damModel.predict(
                                                                        dp.tensorizeText(testTextTensor[start:end]),
                                                                        testLabel[start:end])
                    else:
                        lossValAverage_test, predicts_test = damModel.predict(
                                                                        dp.tensorizeText(testTextTensor[start:end]),
                                                                        testBinaryFeaturesTensor[start:end],
                                                                        testLabel[start:end])

                    lossValAverage_test_list.append(lossValAverage_test)
                    predicts_test_list = predicts_test_list + predicts_test.tolist()

                loss_test = np.array(lossValAverage_test_list).mean()
                if n_classes == 1:
                    auc_test = roc_auc_score(testLabel, predicts_test_list)


                    toBeWritten = [numOfEpoch, numOfBatch, loss_train, auc_train, loss_test, auc_test]
                    print(numOfEpoch, numOfBatch, loss_train, loss_test, auc_train, auc_test)
                    sys.stdout.flush()
                    logList.append(toBeWritten)
                else:
                    print(numOfEpoch, numOfBatch, loss_train, loss_test)
                    sys.stdout.flush()

            if numOfBatch % model_save_frequency == 0:
                if n_classes == 1 and n_features == 0:
                    modelName = model_save_path + 'DAM_binary'
                    damModel._saver.save(sess, modelName, global_step=numOfBatch)
                elif n_classes == 1 and n_features > 0:
                    modelName = model_save_path + 'DWAM_binary'
                    damModel._saver.save(sess, modelName, global_step=numOfBatch)
                elif n_classes > 1 and n_features == 0:
                    modelName = model_save_path + 'DAM_multi'
                    damModel._saver.save(sess, modelName, global_step=numOfBatch)
                elif n_classes > 1 and n_features > 0:
                    modelName = model_save_path + 'DWAM_multi'
                    damModel._saver.save(sess, modelName, global_step=numOfBatch)

    sess.close()
    print('Script finished successfully.')

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

    pa = vars(args)
    print(pa)

    return args

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train DeepMatch")
    args = parse_arguments(parser)

    print('Starting...')
    predict_ER_patient_resources(training_file_path=args.training_file_path,
                                model_save_path=args.model_save_path,
                                test_file_path=args.validation_file_path,
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
