import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
# Tokenizing a string that represents a sentence splits the sentence into a list of words.
from collections import Counter
# NB : les imports sont généralement tous au début, par convention
from nltk.stem import WordNetLemmatizer as wnl
from nltk.corpus import stopwords
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', None)  # or 199
import string
import matplotlib.pyplot as plt
from PIL import Image
import random
from wordcloud import WordCloud
import ast


def sentence_tokenization(df):
    """ Tokenize sentences from our raw data """
    sentence = df['Sentence'].lower()  # convertion to lower case
    tokens = word_tokenize(sentence)  # tokenization
    token_words = [word for word in tokens if word.isalpha()]  # only words taken (no punctuation)
    return token_words

def sentence_lemmatization(df):
    """ Lemmatize previous tokenized sentences """
    sentence_tokenized = df['Sentence_tokenized']
    lemmatized_sentence = [wnl().lemmatize(word) for word in sentence_tokenized]
    return (lemmatized_sentence)

stop_words = stopwords.words('english')
stop_words.extend(['wa','go','know','see','got','come','yes','ha','get','ca'])

def remove_stop_words(df):
    """ Remove stop_words from lemmatized sentences """
    sentence = df['Sentence_lemmatized']
    new_sentence = [word for word in sentence if not word in stop_words]
    return (new_sentence)

def common_words(df):
    """ Return most common words by character """
    text = df['speech']
    text = text.replace(", ", " ")
    text = text.replace("'", "")
    text = ''.join(text)
    word_list = Counter(text.split()).most_common(100)
    return (word_list)

def grey_color_func(word, font_size, position, orientation, random_state=None,
                    **kwargs):
    """ Set the range of color for the wordcloud """
    return "hsl(0, 0%%, %d%%)" % random.randint(0, 10)

def word_count(df):
    """ Count number of words per sentence """
    sentence = df['Sentence_tokenized']
    return len(sentence)