import json
import os
import random
import pickle
import sys

sys.path.append("../")
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import time
import requests
import spacy

from analytics.neuralNetworks import FeedForwardNN, FeedForwardNN_multipleLayers
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader

from analytics.utils import plot_confusion_matrix


train = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("/data/maren_semantic_analysis/E2E/att_one_way_all/e2e_att_one_way_all_embeddings.p", "rb") as fp:
    data = pickle.load(fp)

#with open("../../data/vectorized_data/vectorized_data_german_tfidf.p", "rb") as fp:
    #train_data_vectorized, test_data_vectorized = pickle.load(fp)

#with open("../../data/vectorized_data/labels_binary.p", "rb") as fp:
    #y_train, y_test = pickle.load(fp)

labels = {"entailment": 0, "neutral": 1, "contradiction": 2}



train_labels = data[1]
test_labels = data[3]

y_train = []
for label in train_labels:
    y_train.append(labels[label[2]])

y_test = []
for label in test_labels:
    y_test.append(labels[label[2]])

train_data_vectorized = data[0]
test_data_vectorized = data[2]

#train_data_vectorized = np.array(train_data_vectorized)
#test_data_vectorized = np.array(test_data_vectorized)


class_names = np.array(list(set(y_train)))

NUM_LABELS = len(set(labels.values()))
VOCAB_SIZE = train_data_vectorized.shape[1]
LAYER_CONFIG = [100]
BATCH_SIZE = 64
NUM_EPOCHS = 100


save_name = "/data/maren_semantic_analysis/E2E/{0}_{1}_hidden_model.pt".format(model_save_name, len(LAYER_CONFIG))
# save_name = "/data/maren_semantic_analysis/E2E/glove_freeze_wv/glove_freeze_wv_1_hidden_model.pt"

model = FeedForwardNN_multipleLayers(NUM_LABELS, VOCAB_SIZE, LAYER_CONFIG).to(device)  # dropout=0.0

if train:
    #dataset = TensorDataset(torch.from_numpy((train_data_vectorized.todense())).type(torch.float32), \
                                             #torch.from_numpy(np.array(y_train)).type(torch.long))
    #dataset = TensorDataset(torch.from_numpy(train_data_vectorized).type(torch.float32),\
                            #torch.from_numpy(np.array(y_train)).type(torch.long))
    dataset = TensorDataset(train_data_vectorized,\
                            torch.from_numpy(np.array(y_train)).type(torch.long))
    dataloader = DataLoader(dataset, batch_size= BATCH_SIZE)

    loss_function = nn.NLLLoss()
    optimizer = optim.Adadelta(model.parameters(), weight_decay=1e-4)

    print("Training ...")

    for epoch in range(NUM_EPOCHS):
        total_loss = 0.0
        for i, data in enumerate(dataloader):
            model.zero_grad()

            inputs, labels_train = data

            log_probs = model.forward(inputs.to(device))

            loss = loss_function(log_probs, labels_train.to(device))
            total_loss += loss
            loss.backward()
            optimizer.step()
        print("Loss in epoch {0}: {1}".format(epoch + 1, total_loss))

    #torch.save(model.state_dict(), save_name)

else:
    model.load_state_dict(torch.load(save_name))
    model.eval()

print("Testing ...")

with torch.no_grad():
    pred = []
    for vector, label in zip(test_data_vectorized, y_test):
        #log_probs = model(torch.from_numpy(vector).type(torch.float32).to(device))
        log_probs = model(vector)
        pred.append(int(np.argmax(log_probs.cpu())))

acc = accuracy_score(y_test, pred)
print("Accuracy: {0}".format(acc))

f1 = f1_score(y_test, pred)
print("F1-Score: {0}".format(f1))

plot_confusion_matrix(y_test, pred, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')


#print(class_names)
#with open("../../data/results/NN_{0}Layers_s2s_glove_D2E_binary.p".format(len(LAYER_CONFIG)), "wb") as fp:
    #pickle.dump((pred, y_test), fp)
