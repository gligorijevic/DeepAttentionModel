import numpy as np
import pandas as pd
import string, csv, re, sys
import pickle
from tensorflow.python.platform import gfile


word_vocabulary = {}
stopwords = []


def loadStopwords():
    global stopwords
    with open('../utils/stopwords.txt', 'r') as readfile:
        reader = readfile.readlines()
        for row in reader:
            if row.strip() not in stopwords:
                stopwords.append(row.strip())

def batchIndexesGenerator(batchSize, totalNumberOfLines):

    s = list(zip(range(0, totalNumberOfLines, batchSize), range(batchSize, totalNumberOfLines+ 1, batchSize)))
    lastIncludedIndex = s[-1][1] - 1
    if lastIncludedIndex + 1 != totalNumberOfLines:
        s.append([lastIncludedIndex + 1, totalNumberOfLines])

    return s


def fillString(s, lenght, char='\xff'):
    lenghtOfString = len(s)
    if lenghtOfString < lenght:
        return s + (lenght - lenghtOfString) * char
    else:
        return s[:lenght]


def tensorizeText(inList):
    return np.asarray(inList)


def loadPretrainedEmbeddingsAndVocabulary(wordDictPath='./data/glove.6B.300d.txt', wordVectorLenght=300):
    global word_vocabulary
    lines = [line.rstrip('\n') for line in open(wordDictPath)]
    wordDict = {}
    for line in lines:
        lineProcessed = line.split('\n')[0].split(' ')
        word = lineProcessed[0]
        vector = lineProcessed[1:]
        wordDict[word] = vector

    words_array = []
    for key, value in wordDict.items():
        words_array.append(key)

    word_vocabulary = {}
    words_array.sort()
    word_vocabulary = dict((c, i + 1) for i, c in enumerate(words_array))
    word_vocabulary['</s>'] = 0

    word_embeddings = np.zeros([len(word_vocabulary), wordVectorLenght])
    for word in word_vocabulary:
        word_index = word_vocabulary[word]
        if word in wordDict:
            word_embeddings[word_index, :] = wordDict[word]

    return word_embeddings


def learnVocabulary(dataset, skip_stopwords=True):
    global word_vocabulary
    word_vocabulary = {}
    with open(dataset, 'r') as readfile:
        reader = readfile.readlines()
        for row in reader:
            splitted_row = re.split('[ \t\n]', row)
            for w in splitted_row:
                w = w.lower()
                if skip_stopwords:
                    if w != '' and w not in word_vocabulary and w not in stopwords:
                        word_vocabulary[w] = 1
                    elif w != '' and w not in stopwords:
                        word_vocabulary[w] += 1
                else:
                    if w != '' and w not in word_vocabulary:
                        word_vocabulary[w] = 1
                    elif w != '':
                        word_vocabulary[w] += 1
    print('Vocabulary size:', len(word_vocabulary))
    return word_vocabulary

def loadVocabulary(vocab_file_path):
    global word_vocabulary
    with open(vocab_file_path+'.pkl', 'rb') as f:
        word_vocabulary = pickle.load(f)

def enrichVocabulary(dataset, skip_stopwords=True):
    global word_vocabulary
    with open(dataset, 'r') as readfile:
        reader = readfile.readlines()
        for row in reader:
            splitted_row = re.split('[ \t\n]', row)
            for w in splitted_row:
                w = w.lower()
                if skip_stopwords:
                    if w != '' and w not in word_vocabulary and w not in stopwords:
                        word_vocabulary[w] = 1
                    elif w != '' and w not in stopwords:
                        word_vocabulary[w] += 1
                else:
                    if w != '' and w not in word_vocabulary:
                        word_vocabulary[w] = 1
                    elif w != '':
                        word_vocabulary[w] += 1
    print('Vocabulary size:', len(word_vocabulary))

def cleanAndBuildVocabulary(word_frequency_threshold):
    global word_vocabulary
    words_array = []
    for key, value in word_vocabulary.items():
        if value >= word_frequency_threshold:
            words_array.append(key)
    word_vocabulary = {}

    words_array.sort()
    word_vocabulary = dict((c, i + 1) for i, c in enumerate(words_array))
    word_vocabulary['</s>'] = 0

def saveVocabulary(dictationary_file):
    with open(dictationary_file+'.pkl', 'wb') as f:
        pickle.dump(word_vocabulary, f, pickle.HIGHEST_PROTOCOL)


def indexideText(s, length):
    s_ids = [0]*length
    import math
    if type(s) == float and math.isnan(s):
        print('Something is wrong with the example:', s)
        return s_ids
    if type(s) == str:
        s_array = s.split(' ')
        lengthOfString = len(s_array)
        j = 0
        for i in range(lengthOfString):
            if i == length:
                break
            if s_array[i].lower() in word_vocabulary:
                s_ids[j] = word_vocabulary[s_array[i].lower()]
                j += 1
        return s_ids
    else:
        lengthOfString = len(s)
        j = 0
        for i in range(lengthOfString):
            if i == length:
                break
            if s[i].lower() in word_vocabulary:
                s_ids[j] = word_vocabulary[s[i].lower()]
                j += 1
        return s_ids


def loadDEMdataWords(dataset, number_of_words=1000):
    """
    Name of the files for training, validation and test should be named the same and put into different folders
    """
    # read nurses notes data
    with open(dataset+'/amazon_reviews_text', 'r') as csv_read_file:
        notes = []
        spamreader = csv.reader(csv_read_file, delimiter=' ', quotechar='|')
        for row in spamreader:
            document = indexideText(row, length=number_of_words)
            notes.append(document)

    # read multiclass classes
    with open(dataset + '/amazon_reviews_scores', 'r') as csv_read_file:
        labels_multi = []
        spamreader = csv.reader(csv_read_file, delimiter=',', quotechar='|')
        for row in spamreader:
            labels_multi.append([int(i) for i in row])

    return [notes, labels_multi]


def loadDEMdataWordsAndNumerical(dataset, number_of_words=1000):
    """
        Name of the files for training, validation and test should be named the same and put into different folders
    """
    # read nurses notes data
    with open(dataset+'/all_txt_in_for_phrases.csv', 'r') as csv_read_file:
        notes = []
        spamreader = csv.reader(csv_read_file, delimiter=' ', quotechar='|')
        for row in spamreader:
            document = indexideText(row, length=number_of_words)
            notes.append(document)

    # read numerical features data
    with open(dataset + '/all_binary_features.csv', 'r') as csv_read_file:
        binary_features = []
        spamreader = csv.reader(csv_read_file, delimiter=',', quotechar='|')
        for row in spamreader:
            binary_features.append([float(i) for i in row])

    # read binary classes
    with open(dataset + '/all_classes_binary.csv', 'r') as csv_read_file:
        labels_binary = []
        spamreader = csv.reader(csv_read_file, delimiter=' ', quotechar='|')
        for row in spamreader:
            labels_binary.append(int(row[0]))

    # read multiclass classes
    with open(dataset + '/all_classes_multi.csv', 'r') as csv_read_file:
        labels_multi = []
        spamreader = csv.reader(csv_read_file, delimiter=',', quotechar='|')
        for row in spamreader:
            labels_multi.append([int(i) for i in row])

    return [notes, binary_features, labels_binary, labels_multi]


def loadDEMdataNumerical(dataset, number_of_words=1000):
    """
        Name of the files for training, validation and test should be named the same and put into different folders
    """
    # read numerical features data
    with open(dataset + '/all_binary_features.csv', 'r') as csv_read_file:
        binary_features = []
        spamreader = csv.reader(csv_read_file, delimiter=',', quotechar='|')
        for row in spamreader:
            binary_features.append([float(i) for i in row])

    # read binary classes
    with open(dataset + '/all_classes_binary.csv', 'r') as csv_read_file:
        labels_binary = []
        spamreader = csv.reader(csv_read_file, delimiter=' ', quotechar='|')
        for row in spamreader:
            labels_binary.append(int(row[0]))

    # read multiclass classes
    with open(dataset + '/all_classes_multi.csv', 'r') as csv_read_file:
        labels_multi = []
        spamreader = csv.reader(csv_read_file, delimiter=',', quotechar='|')
        for row in spamreader:
            labels_multi.append([int(i) for i in row])

    return [binary_features, labels_binary, labels_multi]



def loadDEMdataChars(dataset, number_of_chars=10000):
    with open(dataset, 'r') as csv_read_file:
        notes = []
        labels = []
        spamreader = csv.reader(csv_read_file, delimiter=' ', quotechar='|')
        for row in spamreader:
            target = 1.0 if int(row[-1][10:]) > 3 else 0.0
            nurses_note = ' '.join(map(str, row[1:-2]))
            document = fillString(nurses_note[0:number_of_chars], number_of_chars)
            notes.append(document)
            labels.append(target)

    return [notes, labels]


def loadDEMdataCharsAndNumerical(dataset, number_of_chars=10000):
    # read nurses notes data
    with open(dataset + '/all_txt_in_for_phrases.csv', 'r') as csv_read_file:
        notes = []
        spamreader = csv.reader(csv_read_file, delimiter=' ', quotechar='|')
        for row in spamreader:
            nurses_note = ' '.join(map(str, row))
            document = fillString(nurses_note[0:number_of_chars], number_of_chars)
            notes.append(document)

    # read numerical features data
    with open(dataset + '/all_binary_features.csv', 'r') as csv_read_file:
        binary_features = []
        spamreader = csv.reader(csv_read_file, delimiter=',', quotechar='|')
        for row in spamreader:
            binary_features.append([float(i) for i in row])

    # read binary classes
    with open(dataset + '/all_classes_binary.csv', 'r') as csv_read_file:
        labels_binary = []
        spamreader = csv.reader(csv_read_file, delimiter=' ', quotechar='|')
        for row in spamreader:
            labels_binary.append(int(row[0]))

    # read multiclass classes
    with open(dataset + '/all_classes_multi.csv', 'r') as csv_read_file:
        labels_multi = []
        spamreader = csv.reader(csv_read_file, delimiter=',', quotechar='|')
        for row in spamreader:
            labels_multi.append([int(i) for i in row])

    return [notes, binary_features, labels_binary, labels_multi]
