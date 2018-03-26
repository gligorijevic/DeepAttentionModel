import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics as mt


def getResultsBinary(labels, predictions, local_results_model):
    # get ROC curve vectors
    fpr, tpr, thresholds = mt.roc_curve(labels, predictions, pos_label=1)

    # get AUC of the curve
    roc_auc = mt.auc(fpr, tpr)

    # find the best threshold
    A = tpr - fpr
    ind = A.argmax(axis=0)
    threshold = thresholds[ind]

    # get PR curve vectors
    precision, recall, _ = mt.precision_recall_curve(labels, predictions)

    # get AUC of the curve
    prc_auc = mt.auc(recall, precision)

    # get quality indicators
    print('TESTING:', type(labels), type(predictions > threshold))
    print(labels[0:10], predictions[0:10])
    prec, reca, f1sc, supp = mt.precision_recall_fscore_support(labels, predictions > threshold, average='binary')
    acc = mt.accuracy_score(labels, predictions > threshold)

    # print into file
    f1 = open(local_results_model+'/stats.txt','a+')

    # print quality indicatorss
    f1.write('Stats :\n')
    f1.write('ROC AUC: {0} \nPRC AUC: {1}\n'.format(roc_auc, prc_auc))
    f1.write('Based on threshold: {0} the following performance metrics are:\n'.format(threshold))
    f1.write('Accuracy: {0} \nPrecision: {1} \nRecall: {2} \nF1-score: {3}\n\n'.format(acc, prec, reca, f1sc))
    f1.close()
    # plot ROC
    # plt.figure()
    # plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % (roc_auc))
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic '+key)
    # plt.legend(loc="lower right")
    # plt.savefig(local_results_model+'ROC_MODEL_'+key+'.pdf', bbox_inches='tight')
    #
    # plot PRC
    # plt.figure()
    # plt.plot(recall, precision, label='PR curve (area = %0.2f)' % (prc_auc))
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.title('Precision-Recall curve '+key)
    # plt.legend(loc="lower left")
    # plt.savefig(local_results_model+'PRC_MODEL_'+key+'.pdf', bbox_inches='tight')


def getResultsMulti(labels, predictions, local_results_model):
    labels_multi = np.asarray(labels)
    predictions_multi = np.zeros([len(labels), 6])
    print(type(predictions), predictions.shape)

    maxis_label = labels_multi.argmax(axis=1).tolist()

    maxis = predictions.argmax(axis=1).tolist()
    for i,m in enumerate(maxis):
        predictions_multi[i,m] = predictions[i,m] #1

    numerator = 0
    denominator = len(maxis_label)
    for i in range(len(maxis_label)):
        if maxis_label[i] == maxis[i]:
            numerator += 1

    confusion_matrix = mt.confusion_matrix(maxis_label, maxis)
    # print into file
    f1 = open(local_results_model + '/stats.txt', 'a+')
    f1.write('Overall Accuracy of the model is ' + str(float(numerator)/float(denominator)) + ' :\n')
    for i in range(6):
        print('Getting results for class:', i)

        labels_class = labels_multi[:,i]
        predictions_class = predictions_multi[:,i]
        print('Sum labels:', np.sum(labels_class))
        print('Sum preds:', np.sum(predictions_class))

        # get ROC curve vectors
        fpr, tpr, thresholds = mt.roc_curve(labels_class, predictions_class, pos_label=1)

        # get AUC of the curve
        roc_auc = mt.auc(fpr, tpr)

        # find the best threshold
        A = tpr - fpr
        ind = A.argmax(axis=0)
        threshold = thresholds[ind]

        # get PR curve vectors
        precision, recall, _ = mt.precision_recall_curve(labels_class, predictions_class)

        # get AUC of the curve
        prc_auc = mt.auc(recall, precision)

        # get quality indicators
        print('TESTING:', type(labels_class), type(predictions_class > threshold))
        print(labels_class[0:10], predictions_class[0:10])
        prec, reca, f1sc, supp = mt.precision_recall_fscore_support(labels_class.tolist(), predictions_class > threshold, average='binary')
        acc = mt.accuracy_score(labels_class, predictions_class > threshold)



        # print quality indicatorss
        f1.write('Stats for class '+ str(i) +' :\n')
        f1.write('ROC AUC: {0} \nPRC AUC: {1}\n'.format(roc_auc, prc_auc))
        f1.write('Based on threshold: {0} the following performance metrics are:\n'.format(threshold))
        f1.write('Accuracy: {0} \nPrecision: {1} \nRecall: {2} \nF1-score: {3}\n\n'.format(acc, prec, reca, f1sc))

    f1.write('\n')
    # f1.write(os.linesep.join(map(''.join, confusion_matrix)))
    f1.writelines('\t'.join(str(j) for j in i) + '\n' for i in confusion_matrix.tolist())
    f1.close()


