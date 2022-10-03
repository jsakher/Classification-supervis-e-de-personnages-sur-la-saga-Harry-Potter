import pandas as pd
import numpy as np  # imported but unused
import nltk
import pprint as p  # imported but unused
# from nltk.stem import WordNetLemmatizer as wnl
# Tiffany : lemmetize doesnt work for me with this line
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize  # Tokenizing a string that represents a sentence splits the sentence into a list of words.
from collections import Counter
wnl = WordNetLemmatizer()
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# df1 = pd.read_csv('Classification_Personnage_2021/Data/Harry Potter 1.csv', sep=';', encoding= 'unicode_escape')
# df2 = pd.read_csv('Classification_Personnage_2021/Data/Harry Potter 2.csv', sep=';', encoding= 'unicode_escape')
# df3 = pd.read_csv('Classification_Personnage_2021/Data/Harry Potter 3.csv', sep=';', encoding= 'unicode_escape')

# Local Path
df1 = pd.read_csv('../Data/Harry Potter 1.csv', sep=r'\;', encoding='unicode_escape')
df2 = pd.read_csv('../Data/Harry Potter 2.csv', sep=r'\;', encoding='unicode_escape')
df3 = pd.read_csv('../Data/Harry Potter 3.csv', sep=r'\;', encoding='unicode_escape')


def sentence_tokenization(df):
    sentence = df['Sentence'].lower()  # convertion to lower case
    tokens = word_tokenize(sentence)  # tokenization
    token_words = [word for word in tokens if word.isalpha()]  # only words taken (not punctuation)
    return token_words


df1['Sentence_tokenized'] = df1.apply(sentence_tokenization, axis=1)
df2['Sentence_tokenized'] = df2.apply(sentence_tokenization, axis=1)
df3['Sentence_tokenized'] = df3.apply(sentence_tokenization, axis=1)


def sentence_lemmatization(df):
    sentence_tokenized = df['Sentence_tokenized']
    lemmatized_sentence = [wnl.lemmatize(word) for word in sentence_tokenized]
    return (lemmatized_sentence)


df1['Sentence_lemmatized'] = df1.apply(sentence_lemmatization, axis=1)
df2['Sentence_lemmatized'] = df2.apply(sentence_lemmatization, axis=1)
df3['Sentence_lemmatized'] = df3.apply(sentence_lemmatization, axis=1)

stop_words = set(stopwords.words('english'))


def remove_stop_words(df):
    sentence = df['Sentence_lemmatized']
    new_sentence = [word for word in sentence if not word in stop_words]
    return (new_sentence)


df1['Sentence_cleared'] = df1.apply(remove_stop_words, axis=1)
df2['Sentence_cleared'] = df2.apply(remove_stop_words, axis=1)
df3['Sentence_cleared'] = df3.apply(remove_stop_words, axis=1)


# Tiffany : Il faudrait enregistrer les df post-traitement
# car ce n'est pas utile de refaire le traitement plusieurs fois.
# Rq : je conseille d'installer un 'linter' car votre code n'était pas PEP8 ;)


def word_count(column):
    df_list = [item for sublist in column.tolist() for item in sublist]
    return Counter(df_list)


def most_freq_word(counted: dict, n: int):
    df1_counter = pd.DataFrame(counted.items())
    df1_counter.columns = ["word", "freq"]
    df_tmp = df1_counter[df1_counter.freq >= n]
    return df_tmp


df1_counter = dict(word_count(df1.Sentence_cleared))
print(most_freq_word(df1_counter, 33))
