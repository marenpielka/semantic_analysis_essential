from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import numpy as np
import copy
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import time
import math
from flair.embeddings import WordEmbeddings, Sentence, FlairEmbeddings, StackedEmbeddings
from gensim.models.doc2vec import Doc2Vec
import gc
import os

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

import sys
sys.path.append("../")

from analytics.utils import plot_confusion_matrix, print_confusion_matrix
from analytics.neuralNetworks import FeedForwardNN_multipleLayers
from analytics.SequenceToSequenceModel import EncoderRNN as EncoderRNN
from analytics.SequenceToSequenceModel_withFlair import EncoderRNN as EncoderRNN_Flair

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#torch.cuda.set_device(1)
import pickle

np.random.seed(27)

SOS_token = 1
EOS_token = 2

#### ADJUSTABLE PARAMETERS ####

model_save_name = "no_att_english"

# vectorization method for the pre-trained word vectors
word_vecs = "glove" # "glove", "flair", "w2v

if word_vecs == "glove":
    hidden_size = 300
    embedding_size = 300
elif word_vecs == "flair":
    hidden_size = 2048
    embedding_size = 4096
elif word_vecs == "w2v":
    hidden_size = 400
    embedding_size = 400
else:
    print("Undefined vectorization method {0}!".format(word_vecs))
    sys.exit(1)

MAX_LENGTH = 25

# model parameters
learning_rate=0.01
weight_decay = 0.0

NUM_LABELS = 2
SENT_EMBED_SIZE = 2*hidden_size
LAYER_CONFIG = [1000] if word_vecs == "flair" else [100]
NUM_EPOCHS = 100
BATCH_SIZE=64
DROPOUT=0.0

# label encoding
labels = {"entailment": 0, "neutral": 0, "contradiction": 1}

# whether to freeze the word vectors or train them
freeze_wv = True

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {"OOV": 0}
        self.word2count = {}
        self.index2word = {0: "OOV", 1: "SOS", 2: "EOS"}
        self.n_words = 3  # Count SOS, EOS and OOV

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def normalizeString(s):
    if word_vecs == "glove":
        s = s.lower()
    s = s.strip()
    s = re.sub(r"([,.!?])", r" \1", s)
    return s


def filterSent(p):
    return len(p.split(' ')) < MAX_LENGTH


def filterLang(lang):
    return [sent for sent in lang if filterSent(sent)]

def transform_data(data):
    data_new = []
    for sentence_pair in data:
        data_new.append(sentence_pair[0])
        data_new.append(sentence_pair[1])
    return data_new

def readSentences(lang1):
    print("Reading lines...")
    # Split every line into pairs and normalize
    lang1 = [normalizeString(s) for s in lang1]
    return lang1

def prepareData(sents):
    input_sents = readSentences(sents)
    input_sents = filterLang(input_sents)
    lang = Lang(input_sents)
    # pairs = pairs[:1000]
    print("Counting words...")
    for sent in input_sents:
        lang.addSentence(sent)
    return lang

def indexesFromSentence(lang, sentence):
    # append token for OOV words
    out = []
    for word in sentence.split(' '):
        if word in lang.word2index.keys():
            out.append(lang.word2index[word])
        else:
            out.append(lang.word2index["OOV"])
    return out


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def get_embeddings(encoder, sentence, input_lang):
    with torch.no_grad():
        if word_vecs == "flair":
            flair_embedding = StackedEmbeddings([
                FlairEmbeddings('de-forward'),
                FlairEmbeddings('de-backward'),
            ])

            sent = Sentence(sentence + " <EOS>")
            flair_embedding.embed(sent)
            input_tensor = [token.embedding for token in sent.tokens]
            input_length = len(input_tensor)
        else:
            input_tensor = tensorFromSentence(input_lang, sentence)
            input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        for ei in range(input_length):
            _, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        return encoder_hidden

def get_embeddings_for_dataset(data, encoder, input_lang, max_length=MAX_LENGTH, bidirectional=False, pooling="max"):
    embeddings = []
    y = []
    original_data = []
    for sentence_pair in data:
        normalizedval1 = sentence_pair[0].lower().strip()
        normalizedval1 = re.sub(r"([,.!?])", r" \1", normalizedval1)
        normalizedval2 = sentence_pair[1].lower().strip()
        normalizedval2 = re.sub(r"([,.!?])", r" \1", normalizedval2)
        if len(normalizedval1.split(' ')) < max_length and len(normalizedval2.split(' ')) < max_length \
                and sentence_pair[2] != "-":

            if bidirectional:
                emb_1_bid = get_embeddings(encoder, normalizedval1, input_lang)
                emb_2_bid = get_embeddings(encoder, normalizedval2, input_lang)
                emb1_1 = emb_1_bid[0][0]
                emb1_2 = emb_1_bid[1][0]
                emb2_1 = emb_2_bid[0][0]
                emb2_2 = emb_2_bid[1][0]

                if pooling == "sum":
                    emb1 = torch.sum(torch.stack((emb1_1, emb1_2)), 0)
                    emb2 = torch.sum(torch.stack((emb2_1, emb2_2)), 0)
                elif pooling == "max":
                    emb1 = torch.max(emb1_1, emb1_2)
                    emb2 = torch.max(emb2_1, emb2_2)
                elif pooling == "concat":
                    emb1 = torch.cat((emb1_1, emb1_2), -1)
                    emb2 = torch.cat((emb2_1, emb2_2), -1)

            else:
                emb1 = get_embeddings(encoder, normalizedval1,  input_lang)[0][0]
                emb2 = get_embeddings(encoder, normalizedval2,  input_lang)[0][0]

            embeddings.append(torch.cat((emb1, emb2), -1))
            y.append(labels[sentence_pair[2]])
            original_data.append(sentence_pair)

    embeddings = torch.stack(tuple(embeddings))
    y = np.array(y)

    return embeddings, y, original_data

def savePlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.savefig("/data/maren_semantic_analysis/E2E/trainingLoss_E2E_German.png")

#@profile
def trainIters_supervised(encoder, model, train_data, val_data, batch_size, n_iters, input_lang, print_every=100,
                          plot_every=10, learning_rate=0.1, weight_decay=1e-4, loss_threshold=0.01, save_every=10, evaluate_every=10):
    labels = {"contradiction": 1, "neutral": 0, "entailment": 0}
    class_names = copy.deepcopy(np.array(list(labels.keys())))

    data = []
    for sentence_pair in train_data:
        normalizedval1 = sentence_pair[0].lower().strip()
        normalizedval1 = re.sub(r"([.!?])", r" \1", normalizedval1)
        normalizedval2 = sentence_pair[1].lower().strip()
        normalizedval2 = re.sub(r"([.!?])", r" \1", normalizedval2)
        if len(normalizedval1.split(' ')) < MAX_LENGTH and len(normalizedval2.split(' ')) < MAX_LENGTH \
                and sentence_pair[2] != "-":
            data.append([normalizedval1, normalizedval2, labels[sentence_pair[2]]])

    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model_optimizer = optim.SGD(model.parameters(), weight_decay=weight_decay, lr=learning_rate)
    #encoder_optimizer = optim.Adam(encoder.parameters(), weight_decay=weight_decay, lr=learning_rate)  # Adam
    #model_optimizer = optim.Adam(model.parameters(), weight_decay=weight_decay, lr=learning_rate)  # Adam

    if word_vecs=="flair":
        flair_embedding = StackedEmbeddings([
            FlairEmbeddings('de-forward'),
            FlairEmbeddings('de-backward'),
        ])
        input_tensors_1 = []
        input_tensors_2 = []
        for training_pair in data:
            sentence1 = Sentence(training_pair[0] + " <EOS>")
            flair_embedding.embed(sentence1)
            input_tensors_1.append(torch.stack([token.embedding for token in sentence1.tokens]))

            sentence2 = Sentence(training_pair[1] + " <EOS>")
            flair_embedding.embed(sentence2)
            input_tensors_2.append(torch.stack([token.embedding for token in sentence2.tokens]))

    else:
        input_tensors_1 = [tensorFromSentence(input_lang, training_pair[0]) for training_pair in data]
        input_tensors_2 = [tensorFromSentence(input_lang, training_pair[1]) for training_pair in data]

    targets = [training_pair[2] for training_pair in data]

    scheduler1 = ReduceLROnPlateau(encoder_optimizer, 'min', verbose=True)
    scheduler2 = ReduceLROnPlateau(model_optimizer, 'min', verbose=True)

    # compensate for unbalanced classes
    targets2 = torch.from_numpy(np.array(targets)).to(device)
    positive_weight = float(targets2.shape[0]) / torch.sum(targets2).type(torch.float32)
    negative_weight = float(targets2.shape[0]) / (targets2.shape[0] - torch.sum(targets2)).type(torch.float32)
    #criterion = nn.NLLLoss(weight=torch.tensor([negative_weight, positive_weight]).to(device))  # [0.5,1.0]
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([negative_weight, positive_weight]).to(device))  # [0.5,1.0]

    print_loss_avg_last = 0.0
    all_data = list(zip(input_tensors_1, input_tensors_2, targets))
    for iter in range(1, n_iters + 1):
        # create new random batches in each iteration
        batches = []
        current_batch_inputs1 = []
        current_batch_inputs2 = []
        current_batch_targets = []

        random.shuffle(all_data)
        for input_tensor_1, input_tensor_2, target in all_data:
            if len(current_batch_targets) < batch_size:
                current_batch_inputs1.append(input_tensor_1)
                current_batch_inputs2.append(input_tensor_2)
                current_batch_targets.append(target)
            else:
                batches.append((copy.deepcopy(current_batch_inputs1), copy.deepcopy(current_batch_inputs2),
                                torch.from_numpy(np.array(current_batch_targets)).to(device)))
                current_batch_inputs1 = []
                current_batch_inputs2 = []
                current_batch_targets = []

        # for i, batch in enumerate(dataloader):
        for batch in batches:
            encoder_optimizer.zero_grad()
            model_optimizer.zero_grad()
            concat_tensors = []

            input_tensors_1 = batch[0]
            input_tensors_2 = batch[1]
            targets = batch[2]

            for input_tensor_1, input_tensor_2 in zip(input_tensors_1, input_tensors_2):
                # get the embedding for the first sentence
                encoder_hidden_1 = encoder.initHidden()

                input_length_1 = input_tensor_1.size(0)
                input_length_2 = input_tensor_2.size(0)

                for ei in range(input_length_1):
                    _, encoder_hidden_1 = encoder(
                        input_tensor_1[ei], encoder_hidden_1)

                emb1 = torch.max(encoder_hidden_1[0][0], encoder_hidden_1[1][0])

                # re-initialize encoder hidden state and get the embedding for the second sentence
                encoder_hidden_2 = encoder.initHidden()

                for ei in range(input_length_2):
                    _, encoder_hidden_2 = encoder(
                        input_tensor_2[ei], encoder_hidden_2)
                emb2 = torch.max(encoder_hidden_2[0][0], encoder_hidden_2[1][0])

                # concatenate the embeddings
                concat_tensors.append(torch.cat((emb1, emb2), -1).to(device))

                #gc.collect()

            concat_tensors = torch.stack(tuple(concat_tensors))

            log_probs = model.forward(concat_tensors)

            loss = criterion(log_probs, targets)
            loss.backward()

            encoder_optimizer.step()
            model_optimizer.step()

            print_loss_total += loss.item()
            plot_loss_total += loss.item()

            gc.collect()

        scheduler1.step(print_loss_total)
        scheduler2.step(print_loss_total)
        del batches

        if iter % save_every == 0:
            torch.save(encoder.state_dict(),
                       "/data/maren_semantic_analysis/E2E/{0}/encoder_epoch_{1}.pt".format(model_save_name, iter))
            torch.save(model.state_dict(),
                       "/data/maren_semantic_analysis/E2E/{0}/model_epoch_{1}.pt".format(model_save_name, iter))

        if iter % evaluate_every == 0:
            print("Performance in epoch {0}:".format(iter))
            train_embeddings, train_y, _ = get_embeddings_for_dataset(train_data, encoder, input_lang,
                                                                      bidirectional=True, pooling="max")
            val_embeddings, val_y, _ = get_embeddings_for_dataset(val_data, encoder, input_lang,
                                                                    bidirectional=True, pooling="max")

            with torch.no_grad():
                pred = []
                for vector, label in zip(val_embeddings, val_y):
                    # log_probs = model(torch.from_numpy(vector.todense()).type(torch.float32))
                    log_probs = model(vector)
                    log_probs_cpu = log_probs.cpu()
                    pred.append(int(np.argmax(log_probs_cpu)))
                    del log_probs_cpu

            acc = accuracy_score(val_y, pred)
            print("Accuracy of the model: {0}".format(acc))

            print_confusion_matrix(val_y, pred, classes=class_names, normalize=False)

            train_embeddings_cpu = train_embeddings.cpu()
            val_embeddings_cpu = val_embeddings.cpu()

            clf = LogisticRegression(solver="newton-cg", multi_class="multinomial", class_weight="balanced")  # class_weight="balanced"
            clf.fit(train_embeddings_cpu, train_y)
            pred = clf.predict(val_embeddings_cpu)
            acc = accuracy_score(val_y, pred)
            print("Accuracy of Logistic regression: {0}".format(acc))

            del clf
            del train_embeddings_cpu
            del val_embeddings_cpu

            print_confusion_matrix(val_y, pred, classes=class_names, normalize=False)

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_total))
            print_loss_total = 0

            if loss_threshold is not None and np.abs(print_loss_avg - print_loss_avg_last) < loss_threshold:
                print("Loss has converged! Stopping training")
                print("Epoch {0}".format(iter))
                break
            # print_loss_avg_last = copy.deepcopy(print_loss_avg)

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
        gc.collect()
    #savePlot(plot_losses)


with open("../../data/snli_data_processed/train_data_downsampled.p", "rb") as fp:
    train_data = pickle.load(fp)

with open("../../data/snli_data_processed/val_data.p", "rb") as fp:
    val_data = pickle.load(fp)

train_data_concatenated = transform_data(train_data)

input_lang = prepareData(train_data_concatenated)

if word_vecs=="glove":
    with open("/data/maren_semantic_analysis/GloVe/glove_english_pretrained_300d.p", "rb") as fp:
        glove_word_vectors, dictionary = pickle.load(fp)

    word_vectors = []
    count_found = 0
    count_oov = 0
    for ind in input_lang.index2word.keys():
        if input_lang.index2word[ind] in dictionary.keys():
            word_vectors.append(glove_word_vectors[dictionary[input_lang.index2word[ind]]])
            count_found += 1
        else:
            # create new random vectors in the interval [-1,1]
            word_vectors.append(np.random.rand(len(glove_word_vectors[0])) * 2 - 1)

elif word_vecs=="w2v":
    DMVectorizer = Doc2Vec.load("/data/maren_semantic_analysis/D2V/DM_SNLI_german.doc2vec")

    word_vectors = []
    count_found = 0
    count_oov = 0
    for ind in input_lang.index2word.keys():
        if input_lang.index2word[ind] in DMVectorizer.wv.index2word:
            word_vectors.append(DMVectorizer.wv.get_vector(input_lang.index2word[ind]))
            count_found += 1
        else:
            word_vectors.append(np.random.rand(DMVectorizer.wv.vectors[0].shape[0]))

if word_vecs=="flair":
    encoder1 = EncoderRNN_Flair(embedding_size, hidden_size, bidirectional=True).to(device)

else:
    word_vectors = np.array(word_vectors).astype(np.float64)

    encoder1 = EncoderRNN(input_lang.n_words, hidden_size, embedding_size=embedding_size, pretrained=word_vectors,
                          bidirectional=True, freeze_wv=freeze_wv).to(device)

if not os.path.exists("/data/maren_semantic_analysis/E2E/{0}/".format(model_save_name)):
    os.mkdir("/data/maren_semantic_analysis/E2E/{0}/".format(model_save_name))

model = FeedForwardNN_multipleLayers(NUM_LABELS, SENT_EMBED_SIZE, LAYER_CONFIG, dropout=DROPOUT).to(device)

trainIters_supervised(encoder1, model, train_data, val_data, BATCH_SIZE, NUM_EPOCHS, input_lang, loss_threshold=None,
                      learning_rate=learning_rate, weight_decay=weight_decay,print_every=1, plot_every=1, save_every=10, evaluate_every=5)