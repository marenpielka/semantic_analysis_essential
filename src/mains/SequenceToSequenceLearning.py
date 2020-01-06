from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import matplotlib

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import pickle
import copy

import sys
import time
import math
import numpy as np

sys.path.append("../")
from analytics.SequenceToSequenceModel import EncoderRNN, AttnDecoderRNN, DecoderRNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#torch.cuda.set_device(1)
#device="cpu"


# Parameters
setting = "G2E"

SOS_token = 0
EOS_token = 1

max_epochs = 75000
MAX_LENGTH = 25

teacher_forcing_ratio = 0.8
dropout = 0.8
learning_rate = 0.05
loss_threshold = 0.1

attention = False
bidirectional = True

hidden_size = 300
embedding_size = 300

pooling = "max" #"sum", "concat"

labels = {"entailment": 0, "neutral": 0, "contradiction": 1}



def savePlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.savefig("/data/maren_semantic_analysis/S2S/trainingLoss_G2E_GloVe.png")


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


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH,\
          teacher_forcing_ratio=teacher_forcing_ratio, attention=False):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    if attention:
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        if attention:
            encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            if attention:
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
            else:
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            if attention:
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
            else:
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def trainIters(encoder, decoder, n_iters, pairs, input_lang, output_lang, print_every=1000, plot_every=100, learning_rate=0.01,\
               teacher_forcing_ratio=teacher_forcing_ratio, loss_threshold=None, max_length=MAX_LENGTH, attention=False):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(pairs), input_lang, output_lang)
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    print_loss_avg_last = 0.0
    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion, teacher_forcing_ratio=teacher_forcing_ratio,\
                     max_length=max_length, attention=attention)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

            if loss_threshold is not None and np.abs(print_loss_avg - print_loss_avg_last) < loss_threshold:
                print("Loss has converged! Stopping training")
                print("Epoch {0}".format(iter))
                break

            print_loss_avg_last = copy.deepcopy(print_loss_avg)

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    savePlot(plot_losses)


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {"OOV": 2}
        self.word2count = {}
        self.index2word = {2: "OOV", 0: "SOS", 1: "EOS"}
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

def normalizeString(s):
    s = s.lower().strip()
    s = re.sub(r"([,.!?])", r" \1", s)
    #s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def filterPair(p, max_length=MAX_LENGTH):
    return len(p[0].split(' ')) < max_length and \
           len(p[1].split(' ')) < max_length

def filterPairs(pairs, max_length=MAX_LENGTH):
    return [pair for pair in pairs if filterPair(pair, max_length)]

def transform_data(data):
    data_new = []
    for sentence_pair in data:
        data_new.append(sentence_pair[0])
        data_new.append(sentence_pair[1])
    return data_new

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

def prepareData(lang1, lang2, reverse=False, max_length=MAX_LENGTH):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs, max_length=max_length)
    #pairs = pairs[:1000]
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    #print("Counted words:")
    #print(input_lang.name, input_lang.n_words)
    #print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs

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


def tensorsFromPair(pair, input_lang, output_lang):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

def get_embeddings(encoder, sentence, input_lang):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        for ei in range(input_length):
            _, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
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
                emb_1_bid = get_embeddings(encoder1, normalizedval1, input_lang)
                emb_2_bid = get_embeddings(encoder1, normalizedval2, input_lang)
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
                emb1 = get_embeddings(encoder, normalizedval1, input_lang)[0][0]
                emb2 = get_embeddings(encoder, normalizedval2, input_lang)[0][0]

            embeddings.append(torch.cat((emb1, emb2), -1))
            y.append(labels[sentence_pair[2]])
            original_data.append(sentence_pair)

    embeddings = torch.stack(embeddings)
    y = np.array(y)

    return embeddings, y, original_data

def get_word_vectors(lang, glove_word_vectors, dictionary):
    word_vectors = []
    count_found = 0
    count_oov = 0
    for ind in lang.index2word.keys():
        if lang.index2word[ind] in dictionary.keys():
            word_vectors.append(glove_word_vectors[dictionary[lang.index2word[ind]]])
            count_found += 1
        else:
            word_vectors.append(np.random.rand(len(glove_word_vectors[0])))
            count_oov += 1
            # print("Word OOV: " + lang.index2word[ind])

    word_vectors = np.array(word_vectors).astype(np.float64)
    return word_vectors

if __name__=="__main__":

    print("Reading data ...")

    with open("../../data/snli_data_translated/train_data.p", "rb") as fp:
        train_data_german = pickle.load(fp)
    with open("../../data/snli_data_processed/train_data_downsampled.p", "rb") as fp:
        train_data_english = pickle.load(fp)

    print("Preparing data ...")

    train_data_english_concatenated = transform_data(train_data_english)
    train_data_german_concatenated = transform_data(train_data_german)

    input_lang, output_lang, pairs = prepareData(train_data_german_concatenated, train_data_english_concatenated, False)
    #print(input_lang.n_words)

    print("Reading pre-trained word embeddings ...")

    with open("/data/maren_semantic_analysis/GloVe/glove_german_pretrained.p", "rb") as fp:
        glove_word_vectors_german, dictionary_german = pickle.load(fp)

    with open("/data/maren_semantic_analysis/GloVe/glove_english_pretrained.p", "rb") as fp:
        glove_word_vectors_english, dictionary_english = pickle.load(fp)

    word_vectors_input = get_word_vectors(input_lang, glove_word_vectors_german, dictionary_german)
    word_vectors_output = get_word_vectors(output_lang, glove_word_vectors_english, dictionary_english)

    print("Training the model ...")

    encoder1 = EncoderRNN(input_lang.n_words, hidden_size, embedding_size, pretrained=word_vectors_input,
                          bidirectional=True).to(device)

    """
        if attention:
        decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=dropout).to(device)
    else:
        decoder1 = DecoderRNN(hidden_size, output_lang.n_words, embedding_size, pretrained=word_vectors_output,
                              bidirectional=bidirectional).to(device)

    trainIters(encoder1, decoder1, max_epochs, pairs, input_lang, output_lang, print_every=1000, learning_rate=learning_rate,
               attention=attention, loss_threshold=loss_threshold)
               
    torch.save(encoder1.state_dict(), "/data/maren_semantic_analysis/S2S/encoder_{0}_gloVe.pt".format(setting))
    torch.save(encoder1.state_dict(), "/data/maren_semantic_analysis/S2S/decoder_{0}_gloVe.pt".format(setting))
    """

    with open("../../data/snli_data_translated/test_data.p", "rb") as fp:
        test_data_german = pickle.load(fp)

    #with open("../../data/snli_data_processed/test_data.p", "rb") as fp:
        #test_data_english = pickle.load(fp)

    encoder1.load_state_dict(torch.load("/data/maren_semantic_analysis/S2S/encoder_{0}_gloVe.pt".format(setting)))

    print("Inferring the embeddings ...")

    embeddings_train, y_train, original_data_train = get_embeddings_for_dataset(train_data_german, encoder1, input_lang,
                                                                    bidirectional=bidirectional, pooling=pooling)
    embeddings_test, y_test, original_data_test = get_embeddings_for_dataset(test_data_german, encoder1, input_lang,
                                                                    bidirectional=bidirectional, pooling=pooling)

    with open("/data/maren_semantic_analysis/S2S/s2s_gloVe_embeddings_{0}.p".format(setting), "wb") as fp:
        pickle.dump((embeddings_train, embeddings_test, original_data_train, original_data_test), fp)

    print("Training and testing the prediction model ...")

    clf = LogisticRegression(solver="newton-cg",multi_class="multinomial")
    clf.fit(embeddings_train, y_train)
    pred = clf.predict(embeddings_test)
    acc = accuracy_score(y_test, pred)
    print(acc)

    #with open("../../data/results/predictions_{0}_gloVe.p".format(setting), "wb") as fp:
        #pickle.dump((y_test, pred), fp)

