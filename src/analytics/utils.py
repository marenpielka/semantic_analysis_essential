import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import numpy as np

def preprocess_data(data, listOfPreprocessingSteps):
    preprocessed_data = []
    original_data = []
    i = 0
    for sentence_pair in data:
        if sentence_pair[2] != "-":
            preprocessed_text_1 = sentence_pair[0].strip()
            preprocessed_text_2 = sentence_pair[1].strip()
            if "rmDigit" in listOfPreprocessingSteps:
                preprocessed_text_1 = re.sub(r'[0-9]+', '', preprocessed_text_1)
                preprocessed_text_2 = re.sub(r'[0-9]+', '', preprocessed_text_2)
            if "rmPunct" in listOfPreprocessingSteps:
                preprocessed_text_1 = re.sub(r'[^\w\s]',' ',preprocessed_text_1)
                preprocessed_text_2 = re.sub(r'[^\w\s]',' ',preprocessed_text_2)
            if "lowCase" in listOfPreprocessingSteps:
                preprocessed_text_1 = preprocessed_text_1.lower()
                preprocessed_text_2 = preprocessed_text_2.lower()
            if "rmStop" in listOfPreprocessingSteps:
                stop_words = set(stopwords.words('english'))
                word_tokens_1 = word_tokenize(preprocessed_text_1)
                preprocessed_text_1 = (" ").join([w for w in word_tokens_1 if not w in stop_words])
                word_tokens_2 = word_tokenize(preprocessed_text_2)
                preprocessed_text_2 = (" ").join([w for w in word_tokens_2 if not w in stop_words])
                '''
            if "lemma" in listOfPreprocessingSteps:
                lemmatizer = WordNetLemmatizer()
                word_tokens_1 = word_tokenize(preprocessed_text_1)
                tagged_tokens_1 = [(token[0], get_wordnet_pos(token[1])) for token in (nltk.pos_tag(word_tokens_1))]
                preprocessed_text_1 = (" ").join([lemmatizer.lemmatize(w[0], w[1]) for w in tagged_tokens_1])
                word_tokens_2 = word_tokenize(preprocessed_text_2)
                tagged_tokens_2 = [(token[0], get_wordnet_pos(token[1])) for token in (nltk.pos_tag(word_tokens_2))]
                preprocessed_text_2 = (" ").join([lemmatizer.lemmatize(w[0], w[1]) for w in tagged_tokens_2])    
                '''

            # remove multiple whitespaces
            preprocessed_text_1 = " ".join(preprocessed_text_1.split())
            preprocessed_text_2 = " ".join(preprocessed_text_2.split())
            data_to_append = [preprocessed_text_1, preprocessed_text_2, sentence_pair[2]]
            preprocessed_data.append(data_to_append)
            original_data.append([sentence_pair[0], sentence_pair[1], sentence_pair[2]])
            i += 1
            if i % 1000 == 0:
                print("{0} sentence pairs successfully processed".format(i))
    return preprocessed_data, original_data

def print_confusion_matrix(y_true, y_pred, classes,
                          normalize=False):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax