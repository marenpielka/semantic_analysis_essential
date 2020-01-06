import json
import os
import random
import pickle
import sys
sys.path.append("../")
import os
import re
import scipy.sparse
import string
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import itertools

def vectorize_data(preprocessed_data):
    vectorized_data = []
    y = []
    i = 0
    for sentence_pair in preprocessed_data:
        feat1 = vectorizer.transform([sentence_pair[0]])[0]
        feat2 = vectorizer.transform([sentence_pair[1]])[0]
        if sentence_pair[2] == "entailment":
            y.append(1)
        elif sentence_pair[2] == "neutral":
            y.append(0)
        elif sentence_pair[2] == "contradiction":
            y.append(-1)
        else:
            print ("unknown label {0}".format(sentence_pair[2]))
            break
        i +=1
        #vectorized_data_separated.append([feat1, feat2])
        vectorized_data.append(np.concatenate((np.asarray(feat1.todense())[0], np.asarray(feat2.todense())[0])))
        if i % 1000 == 0:
            print("{0} sentence pairs successfully processed".format(i))
    vectorized_data = scipy.sparse.csr_matrix(vectorized_data)
    return vectorized_data, y

def vectorize_data_tfidf(train_data_preprocessed, test_data_preprocessed):
    train_data_vectorized, y_train = vectorize_data(train_data_preprocessed)
    test_data_vectorized, y_test = vectorize_data(test_data_preprocessed)
    return train_data_vectorized, test_data_vectorized, y_train, y_test

def vectorize_data_mf(train_data_preprocessed, test_data_preprocessed):
    mfModel1 = TruncatedSVD(n_components=500, algorithm="arpack")
    #mfModel2 = TruncatedSVD(n_components=500, algorithm="arpack")
    train_data_vectorized, test_data_vectorized, y_train, y_test = vectorize_data_tfidf(train_data_preprocessed,\
                                                                                        test_data_preprocessed)
    vectorized_data_concatenated = scipy.sparse.csr_matrix(np.concatenate((train_data_vectorized.todense(),\
                                                                    test_data_vectorized.todense()), axis=0))
    # fit the two MF models on the first / second slice of data each
    mfModel1.fit(vectorized_data_concatenated.asfptype())
    #mfModel2.fit(vectorized_data_concatenated[:,int(0.5*vectorized_data_concatenated.shape[1]):].asfptype())
    # transform the data separately for train and test
    train_data_vectorized = mfModel1.transform(train_data_vectorized.asfptype())
    #mf2 = mfModel1.transform(train_data_vectorized[:,:int(0.5*train_data_vectorized.shape[1])].asfptype())
    #train_data_vectorized = np.concatenate((mf1, mf2), axis=1)
    test_data_vectorized = mfModel1.transform(test_data_vectorized.asfptype())
    #mf2 = mfModel1.transform(test_data_vectorized[:,:int(0.5*test_data_vectorized.shape[1])].asfptype())
    #test_data_vectorized = np.concatenate((mf1, mf2), axis=1)
    return train_data_vectorized, test_data_vectorized, y_train, y_test

# get a D2V representation from two (DM and DBOW) vectorizers
def vectorize_data_d2v(train_data_preprocessed, test_data_preprocessed, pathsToD2VModels, method="sum"):
    #with open(pathsToD2VModels[0], "rb") as fp:
        #DBOWVectorizer = pickle.load(fp)
    #with open(pathsToD2VModels[1], "rb") as fp:
        #DMVectorizer = pickle.load(fp)
    DBOWVectorizer = Doc2Vec.load(pathsToD2VModels[0])
    DMVectorizer = Doc2Vec.load(pathsToD2VModels[1])
    train_data_vectorized = vectorize_one_part_d2v(train_data_preprocessed, DBOWVectorizer, DMVectorizer,\
                                                           method=method)
    test_data_vectorized = vectorize_one_part_d2v(test_data_preprocessed, DBOWVectorizer, DMVectorizer,\
                                                         method=method)
    return train_data_vectorized, test_data_vectorized

def vectorize_one_part_d2v(data_preprocessed, DBOWVectorizer, DMVectorizer, method):
    data_vectorized = []
    y = []
    for sentence_pair in data_preprocessed:
        word_tokens_1 = word_tokenize(sentence_pair[0])
        word_tokens_2 = word_tokenize(sentence_pair[1])
        feat1_1 = DMVectorizer.infer_vector(word_tokens_1)
        feat2_1 = DBOWVectorizer.infer_vector(word_tokens_1)
        feat1_2 = DMVectorizer.infer_vector(word_tokens_2)
        feat2_2 = DBOWVectorizer.infer_vector(word_tokens_2)
        if method=="concat":
            feat1 = np.concatenate([feat1_1, feat2_1])
            feat2 = np.concatenate([feat1_2, feat2_2])
        elif method=="sum":
            feat1 = np.sum([feat1_1, feat2_1], axis=0)
            feat2 = np.sum([feat1_2, feat2_2], axis=0)
        elif method=="mean":
            feat1 = np.mean([feat1_1, feat2_1], axis=0)
            feat2 = np.mean([feat1_1, feat2_1], axis=0)
        data_vectorized.append(np.asarray(np.concatenate((feat1, feat2), axis=0)))

    data_vectorized = np.array(data_vectorized)
    return data_vectorized

def extract_custom_features(data_labeled):
    custom_features = []
    for sentence_pair in data_labeled:
        num_synonyms = 0
        num_antonyms = 0
        num_hypernyms = 0
        num_hyponyms = 0
        num_entailments = 0
        for pair in itertools.product(sentence_pair[0].split(" "), sentence_pair[1].split(" ")):
            if len(wn.synsets(pair[0])) > 0 and len(wn.synsets(pair[1])) > 0:
                syn1 = wn.synsets(pair[0])[0]
                syn2 = wn.synsets(pair[1])[0]
                if syn1 == syn2:
                    num_synonyms += 1
                if syn1.entailments():
                    for ent in syn1.entailments():
                        if ent == syn2:
                            num_entailments += 1
                for lemma1 in syn1.lemmas():
                    for lemma2 in syn2.lemmas():
                        if lemma1.antonyms():
                            if lemma2 in lemma1.antonyms():
                                num_antonyms += 1
                for hyp in syn1.hypernyms():
                    if hyp == syn2:
                        num_hypernyms += 1
                for hypo in syn1.hyponyms():
                    if hypo == syn2:
                        num_hyponyms += 1
        lenFraction = len(sentence_pair[0].split(" ")) / (len(sentence_pair[0].split(" ")) + len(sentence_pair[1]))
        custom_features.append([num_synonyms, num_entailments, num_antonyms, num_hypernyms, num_hyponyms, lenFraction])
    custom_features = np.array(custom_features)
    return custom_features

"""
#load the already preprocessed data
with open("../../data/snli_data_translated/train_data_preprocessed_rmDigit_rmPunct.p", "rb") as fp:
    train_data_preprocessed = pickle.load(fp)

with open("../../data/snli_data_translated/test_data_preprocessed_rmDigit_rmPunct.p", "rb") as fp:
    test_data_preprocessed = pickle.load(fp)


# vectorize data
print("Vectorizing data ...\n")

vectorizer = TfidfVectorizer(min_df=10)

corpus = []

for sentence_pair in train_data_preprocessed + test_data_preprocessed:
    corpus.append(sentence_pair[0])
    corpus.append(sentence_pair[1])

vectorizer.fit(corpus)



pathsToD2VModels = ["../../data/models/D2V/DBOW_pretrained_german.doc2vec", "../../data/models/D2V/DM_pretrained_german.doc2vec"]
train_data_vectorized, test_data_vectorized = vectorize_data_d2v(train_data_preprocessed,\
    test_data_preprocessed, pathsToD2VModels)

with open("/data/maren_semantic_analysis/ESN/esn_embeddings_train.p", "rb") as fp:
    train_data_vectorized = pickle.load(fp)

with open("/data/maren_semantic_analysis/ESN/esn_embeddings_test.p", "rb") as fp:
    test_data_vectorized = pickle.load(fp)


custom_features_test = extract_custom_features(test_data_preprocessed)
custom_features_train = extract_custom_features(train_data_preprocessed)

train_data_vectorized = scipy.sparse.csr_matrix(np.concatenate((train_data_vectorized.todense(),\
                                                            custom_features_train), axis=1))
test_data_vectorized = scipy.sparse.csr_matrix(np.concatenate((test_data_vectorized.todense(),\
                                                            custom_features_test), axis=1))

"""
labels = {"entailment": 0, "neutral": 1, "contradiction": 2}

#with open("../../data/vectorized_data/labels.p", "rb") as fp:
   #y_train, y_test = pickle.load(fp)

with open("/data/maren_semantic_analysis/S2S/s2s_flair_embeddings_G2G.p", "rb") as fp:
    data = pickle.load(fp)

#print(data)


train_data_vectorized = np.array(data[0])
test_data_vectorized = np.array(data[1])


train_labels = data[2]
test_labels = data[3]

y_train = []
for label in train_labels:
    y_train.append(labels[label[2]])

y_test = []
for label in test_labels:
    y_test.append(labels[label[2]])


# train and test classifier
print("Training the classifier ...\n")
clf = LogisticRegression(solver="newton-cg",multi_class="multinomial")
clf.fit(train_data_vectorized, y_train)

pred = clf.predict(test_data_vectorized)

print("Accuracy: " + str(accuracy_score(y_test, pred)))
