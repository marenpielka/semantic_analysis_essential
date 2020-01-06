from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from gensim.models.doc2vec import Doc2Vec
import gc

import sys
sys.path.append("../")
from analytics.SequenceToSequenceModel import EncoderRNN, AttnDecoderRNN, DecoderRNN
from analytics.neuralNetworks import FeedForwardNN_multipleLayers
from analytics.utils import plot_confusion_matrix, print_confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#torch.cuda.set_device(0)
#device = "cpu"

import pickle
import time
import math
import copy

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

np.random.seed(27)

SOS_token = 0
EOS_token = 1

#### ADJUSTABLE PARAMETERS ####

model_save_name = "att_one_way_all_rerun"

# vectorization method for the pre-trained word vectors
word_vecs = "glove" # "glove", "flair"

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

class_labels = {"entailment": 0, "neutral": 0, "contradiction": 1}

# train 2-way (swap prem and hyp and concatenate embeddings)
two_way = False

# only learn attention for the last output
last_only = False

# model parameters
learning_rate=0.01
weight_decay = 0.0

NUM_LABELS = 2
SENT_EMBED_SIZE = 2*hidden_size if two_way else hidden_size
LAYER_CONFIG = [100]
NUM_EPOCHS = 100
BATCH_SIZE=64
DROPOUT=0.0


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {"OOV": 2}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: "OOV"}
        self.n_words = 3  # Count SOS, EOS, and OOV

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

def normalizeString(s):
    if word_vecs == "glove":
        s = s.lower()
    s = s.strip()
    s = re.sub(r"([,.!?])", r" \1", s)
    return s

def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
           len(p[1].split(' ')) < MAX_LENGTH

def filterPairs(pairs, labels):
    pairs_new = [pair for pair in pairs if filterPair(pair)]
    labels_new = [label for pair, label in zip(pairs, labels) if filterPair(pair)]
    return pairs_new, labels_new

def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Split every line into pairs and normalize
    lang1 = [normalizeString(s) for s in lang1]
    lang2 = [normalizeString(s) for s in lang2]
    pairs = list(zip(lang1, lang2))

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs, labels = filterPairs(pairs, all_labels)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    return input_lang, output_lang, pairs, labels

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

def tensorsFromPair(pair):
    prem_tensor = tensorFromSentence(input_lang, pair[0])
    hyp_tensor = tensorFromSentence(output_lang, pair[1])
    return (prem_tensor, hyp_tensor)

def savePlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.savefig("/data/maren_semantic_analysis/E2E/trainingLoss_E2EAttention_German.png")

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

def get_embeddings(encoder, decoder, sentence1, sentence2, max_length=MAX_LENGTH):
    with torch.no_grad():
        prem_tensor = tensorFromSentence(input_lang, sentence1)
        prem_length = prem_tensor.size()[0]
        hyp_tensor = tensorFromSentence(output_lang, sentence2)
        hyp_length = hyp_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(prem_length):
            encoder_output, encoder_hidden = encoder(
                prem_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        #decoder_input = torch.tensor([[SOS_token]], device=device)

        decoder_hidden = encoder_hidden

        # always use teacher forcing
        for di in range(hyp_length):
            if last_only and di != hyp_length - 1:
                decoder_input = hyp_tensor[di]  # Teacher forcing
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs, apply_attn=False)
            else:
                decoder_input = hyp_tensor[di]  # Teacher forcing
                decoder_output, decoder_hidden, decoder_attention = decoder(#
                    decoder_input, decoder_hidden, encoder_outputs)

        # use last decoder hidden state as class. model input
        return decoder_hidden[0][0]

def get_embeddings_for_dataset(data, encoder, decoder):
    embeddings = []
    y = []
    for sentence_pair in data:
        normalizedval1 = sentence_pair[0].lower().strip()
        normalizedval1 = re.sub(r"([,.!?])", r" \1", normalizedval1)
        normalizedval2 = sentence_pair[1].lower().strip()
        normalizedval2 = re.sub(r"([,.!?])", r" \1", normalizedval2)
        if len(normalizedval1.split(' ')) < MAX_LENGTH and len(normalizedval2.split(' ')) < MAX_LENGTH\
            and sentence_pair[2] != "-":

            emb1 = get_embeddings(encoder, decoder, normalizedval1, normalizedval2)
            if two_way:
                emb2 = get_embeddings(encoder, decoder, normalizedval2, normalizedval1)
                embeddings.append(torch.cat((emb1, emb2), -1))
            else:
                embeddings.append(emb1)
            y.append(class_labels[sentence_pair[2]])

    embeddings = torch.stack(tuple(embeddings))
    y = np.array(y)
    return embeddings, y


def train_supervised(prem_tensor, hyp_tensor, encoder, decoder, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    prem_length = prem_tensor.size(0)
    hyp_length = hyp_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    for ei in range(prem_length):
        encoder_output, encoder_hidden = encoder(
            prem_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    #decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    # always use teacher forcing
    for di in range(hyp_length):
        if last_only and di != hyp_length-1:
            decoder_input = hyp_tensor[di]  # Teacher forcing
            decoder_output, decoder_hidden, decoder_attention = decoder(#
                decoder_input, decoder_hidden, encoder_outputs, apply_attn=False)
        else:
            decoder_input = hyp_tensor[di]
            decoder_output, decoder_hidden, decoder_attention = decoder(#
                decoder_input, decoder_hidden, encoder_outputs)

    #gc.collect()

    return decoder_hidden[0][0]

def trainIters_supervised(encoder, decoder, class_model, targets, n_iters, print_every=1000, plot_every=100,
                          evaluate_every=5, save_every=10, learning_rate=0.05, loss_threshold=None, weight_decay=1e-4,
                          batch_size=BATCH_SIZE, max_length=MAX_LENGTH):
    start = time.time()
    plot_losses = []
    validation_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    class_model_optimizer = optim.SGD(class_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    #encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    #decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    #class_model_optimizer = optim.Adam(class_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # rand_indices = [random.choice(range(len(labels))) for i in range(n_iters)]
    training_pairs = [tensorsFromPair(pair) for pair in pairs]

    scheduler1 = ReduceLROnPlateau(encoder_optimizer, 'min', verbose=True)
    scheduler2 = ReduceLROnPlateau(decoder_optimizer, 'min', verbose=True)
    scheduler3 = ReduceLROnPlateau(class_model_optimizer, 'min', verbose=True)

    # compensate for unbalanced classes
    targets2 = torch.from_numpy(np.array(targets)).to(device)
    positive_weight = float(targets2.shape[0]) / torch.sum(targets2).type(torch.float32)
    negative_weight = float(targets2.shape[0]) / (targets2.shape[0] - torch.sum(targets2)).type(torch.float32)
    #criterion = nn.NLLLoss(weight=torch.tensor([negative_weight, positive_weight]).to(device))  # [0.5,1.0]
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([negative_weight, positive_weight]).to(device))

    all_data = list(zip(training_pairs, targets))
    print_loss_avg_last = 0.0
    for iter in range(1, n_iters + 1):
        # create new random batches in each iteration
        batches = []
        current_batch_inputs = []
        current_batch_targets = []

        random.shuffle(all_data)
        for training_pair, target in all_data:
            if len(current_batch_targets) < batch_size:
                current_batch_inputs.append(training_pair)
                current_batch_targets.append(target)
            else:
                batches.append((copy.deepcopy(current_batch_inputs),
                                torch.from_numpy(np.array(current_batch_targets)).to(device)))
                current_batch_inputs = []
                current_batch_targets = []

        for batch in batches:
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            class_model_optimizer.zero_grad()

            training_pairs = batch[0]
            targets = batch[1]
            tensors = []

            for training_pair in training_pairs:
                emb1 = train_supervised(training_pair[0], training_pair[1], encoder, decoder, max_length=max_length)

                if two_way:
                    emb2 = train_supervised(training_pair[1], training_pair[0], encoder, decoder, max_length=max_length)
                    tensors.append(torch.cat((emb1, emb2), -1))
                else:
                    tensors.append(emb1)

            tensors = torch.stack(tuple(tensors))

            # use last decoder hidden state as class. model input
            log_probs = class_model.forward(tensors)
            loss = criterion(log_probs, targets)

            loss.backward()

            encoder_optimizer.step()
            decoder_optimizer.step()
            class_model_optimizer.step()

            print_loss_total += loss.item()
            plot_loss_total += loss.item()

            #gc.collect()

        del batches

        if iter % print_every == 0:

            print_loss_avg = print_loss_total / print_every
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_total))
            print_loss_total = 0

            if loss_threshold is not None and np.abs(print_loss_avg - print_loss_avg_last) < loss_threshold:
                print("Loss has converged! Stopping training")
                print("Epoch {0}".format(iter))
                break
            #print_loss_avg_last = copy.deepcopy(print_loss_avg)

        if iter % plot_every == 0:
            scheduler1.step(plot_loss_total)
            scheduler2.step(plot_loss_total)
            scheduler3.step(plot_loss_total)

            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

            val_embeddings, val_y = get_embeddings_for_dataset(val_data, encoder, decoder)
            log_probs = class_model(val_embeddings)
            val_loss = criterion(log_probs, torch.Tensor(val_y).type(torch.long).to(device))
            validation_losses.append(val_loss.item())

        if iter % evaluate_every == 0:
            print("Performance in epoch {0}:".format(iter))
            train_embeddings, train_y = get_embeddings_for_dataset(train_data, encoder, decoder,
                                                                   bidirectional=True, pooling="max")
            val_embeddings, val_y = get_embeddings_for_dataset(val_data, encoder, decoder,
                                                                 bidirectional=True, pooling="max")

            with torch.no_grad():
                pred = []
                for vector, label in zip(val_embeddings, val_y):
                    # log_probs = model(torch.from_numpy(vector.todense()).type(torch.float32))
                    log_probs = class_model(vector)
                    log_probs_cpu = log_probs.cpu()
                    pred.append(int(np.argmax(log_probs_cpu)))
                    del log_probs_cpu

            acc = accuracy_score(val_y, pred)
            print("Accuracy of the model: {0}".format(acc))

            print_confusion_matrix(val_y, pred, classes=np.array(["no contradiction", "contradiction"]), normalize=False)

            train_embeddings_cpu = train_embeddings.cpu()
            val_embeddings_cpu = val_embeddings.cpu()

            clf = LogisticRegression(solver="newton-cg", multi_class="multinomial",
                                     class_weight="balanced")  # class_weight="balanced"
            clf.fit(train_embeddings_cpu, train_y)
            pred = clf.predict(val_embeddings_cpu)
            acc = accuracy_score(val_y, pred)
            print("Accuracy of Logistic regression: {0}".format(acc))

            del train_embeddings_cpu
            del val_embeddings_cpu

            print_confusion_matrix(val_y, pred, classes=np.array(["no contradiction", "contradiction"]), normalize=False)

            if iter % save_every == 0:
                torch.save(encoder.state_dict(),
                           "/data/maren_semantic_analysis/E2E/{0}/encoder_epoch_{1}.pt".format(model_save_name, iter))

                torch.save(decoder.state_dict(),
                           "/data/maren_semantic_analysis/E2E/{0}/decoder_epoch_{1}.pt".format(model_save_name, iter))

                torch.save(class_model.state_dict(),
                           "/data/maren_semantic_analysis/E2E/{0}/model_epoch_{1}.pt".format(model_save_name, iter))
        gc.collect()
    #savePlot(plot_losses)

    with open("/data/maren_semantic_analysis/E2E/{0}/train_losses.p".format(model_save_name), "wb") as fp:
        pickle.dump(plot_losses, fp)

    with open("/data/maren_semantic_analysis/E2E/{0}/val_losses.p".format(model_save_name), "wb") as fp:
        pickle.dump(validation_losses, fp)


with open("../../data/snli_data_translated/train_data.p", "rb") as fp:
    train_data = pickle.load(fp)

with open("../../data/snli_data_translated/val_data.p", "rb") as fp:
    val_data = pickle.load(fp)

premises_concatenated = []
hypotheses_concatenated = []
all_labels = []
for pair in train_data:
    if pair[2] != "-":
        premises_concatenated.append(pair[0])
        hypotheses_concatenated.append(pair[1])
        all_labels.append(class_labels[pair[2]])

if two_way:
    input_lang, output_lang, pairs, labels = prepareData(premises_concatenated + hypotheses_concatenated,
                                                         premises_concatenated + hypotheses_concatenated, False)
else:
    input_lang, output_lang, pairs, labels = prepareData(premises_concatenated, hypotheses_concatenated, False)

if word_vecs == "glove":
    with open("/data/maren_semantic_analysis/GloVe/glove_german_pretrained.p", "rb") as fp:
        glove_word_vectors, dictionary = pickle.load(fp)

    # add all word vectors from input and output language, to get common vector representations
    word_vectors = []
    count_found = 0
    count_oov = 0
    for word in list(input_lang.index2word.values()) + list(output_lang.index2word.values()):
        if word in dictionary.keys():
            word_vectors.append(glove_word_vectors[dictionary[word]])
            count_found += 1
        else:
            # create new random vectors in the interval [-1,1]
            word_vectors.append(np.random.rand(len(glove_word_vectors[0])) * 2 - 1)
            count_oov +=1
            print("Word OOV: " + word)

elif word_vecs=="w2v":
    DMVectorizer = Doc2Vec.load("/data/maren_semantic_analysis/D2V/DM_SNLI_german.doc2vec")

    word_vectors = []
    count_found = 0
    count_oov = 0
    for word in list(input_lang.index2word.values()) + list(output_lang.index2word.values()):
        if word in DMVectorizer.wv.index2word:
            word_vectors.append(DMVectorizer.wv.get_vector(word))
            count_found += 1
        else:
            word_vectors.append(np.random.rand(DMVectorizer.wv.vectors[0].shape[0]))
            count_oov +=1
            print("Word OOV: " + word)

if not os.path.exists("/data/maren_semantic_analysis/E2E/{0}/".format(model_save_name)):
    os.mkdir("/data/maren_semantic_analysis/E2E/{0}/".format(model_save_name))

word_vectors = np.array(word_vectors).astype(np.float64)

encoder1 = EncoderRNN(input_lang.n_words, hidden_size, embedding_size=embedding_size, pretrained=word_vectors).to(
    device)
attn_decoder1 = AttnDecoderRNN(hidden_size, embedding_size, output_lang.n_words, dropout_p=DROPOUT,
                               pretrained=word_vectors, max_length=MAX_LENGTH).to(device)
#attn_decoder1 = DecoderRNN(hidden_size, embedding_size, output_lang.n_words,
                               #pretrained=word_vectors).to(device)

model = FeedForwardNN_multipleLayers(NUM_LABELS, SENT_EMBED_SIZE, LAYER_CONFIG, dropout=DROPOUT).to(device)

trainIters_supervised(encoder1, attn_decoder1, model, labels, NUM_EPOCHS, loss_threshold=None, learning_rate=learning_rate,
                      weight_decay=weight_decay, print_every=1, plot_every=1, evaluate_every=1000, batch_size=BATCH_SIZE,
                      save_every=10)