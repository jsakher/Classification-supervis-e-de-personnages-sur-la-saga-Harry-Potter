import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# Tokenizing a string that represents a sentence splits the sentence
# into a list of words.
from collections import Counter  # imported but unused
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
# NB : les imports sont généralement tous au début, par convention

windows = True

if windows:
    df_char = pd.read_csv('..\\Classification_Personnage_2021\\Data\\Characters.csv', sep=r'\;')
    df1 = pd.read_csv('..\\Classification_Personnage_2021\\Data\\Harry Potter 1.csv', sep=r'\;')
    df2 = pd.read_csv('..\\Classification_Personnage_2021\\Data\\Harry Potter 2.csv', sep=r'\;')
    df3 = pd.read_csv('..\\Classification_Personnage_2021\\Data\\Harry Potter 3.csv', sep=r'\;')
    df_potions = pd.read_csv('..\\Classification_Personnage_2021\\Data\\Potions.csv', sep=r'\;')
    df_spells = pd.read_csv('..\\Classification_Personnage_2021\\Data\\Spells.csv', sep=r'\;')
else:
    df1 = pd.read_csv('../Data/Harry Potter 1.csv', sep=r'\;', encoding='unicode_escape')
    df2 = pd.read_csv('../Data/Harry Potter 2.csv', sep=r'\;', encoding='unicode_escape')
    df3 = pd.read_csv('../Data/Harry Potter 3.csv', sep=r'\;', encoding='unicode_escape')
    df_char = pd.read_csv('../Data/Characters.csv', sep=r'\;', encoding='unicode_escape')

def sentence_tokenization(df):
    sentence = df['Sentence'].lower()  # convertion to lower case
    tokens = word_tokenize(sentence)  # tokenization
    token_words = [word for word in tokens if word.isalpha()]  # only words taken (not punctuation)
    return token_words


df1['Sentence_tokenized'] = df1.apply(sentence_tokenization, axis=1)
df2['Sentence_tokenized'] = df2.apply(sentence_tokenization, axis=1)
df3['Sentence_tokenized'] = df3.apply(sentence_tokenization, axis=1)

# Stemming not done cause it alters characters' names: Harry->Harri
# from nltk.stem import PorterStemmer
#
# def sentence_stemming(df):
#    sentence_tokenized = df['Sentence_tokenized']
#    stemmed_sentence = [PorterStemmer().stem(word) for word in sentence_tokenized]
#    return (stemmed_sentence)
#
# df1['Sentence_stemmed'] = df1.apply(sentence_stemming, axis=1)
# df2['Sentence_stemmed'] = df2.apply(sentence_stemming, axis=1)
# df3['Sentence_stemmed'] = df3.apply(sentence_stemming, axis=1)

from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()


def sentence_lemmatization(df):
    sentence_tokenized = df['Sentence_tokenized']
    lemmatized_sentence = [wnl.lemmatize(word) for word in sentence_tokenized]
    return (lemmatized_sentence)


df1['Sentence_lemmatized'] = df1.apply(sentence_lemmatization, axis=1)
df2['Sentence_lemmatized'] = df2.apply(sentence_lemmatization, axis=1)
df3['Sentence_lemmatized'] = df3.apply(sentence_lemmatization, axis=1)


from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
stop_words.add('wa')
stop_words.add('go')
stop_words.add('know')
stop_words.add('see')
stop_words.add('got')
stop_words.add('come')
stop_words.add('yes')
stop_words.add('ha')
stop_words.add('get')


def remove_stop_words(df):
    sentence = df['Sentence_lemmatized']
    new_sentence = [word for word in sentence if not word in stop_words]
    return (new_sentence)


df1['Sentence_cleared'] = df1.apply(remove_stop_words, axis=1)
df2['Sentence_cleared'] = df2.apply(remove_stop_words, axis=1)
df3['Sentence_cleared'] = df3.apply(remove_stop_words, axis=1)


def word_count(df):
    # count number of words per sentence
    sentence = df['Sentence_tokenized']
    return len(sentence)


df1['Word_count'] = df1.apply(word_count, axis=1)
df2['Word_count'] = df2.apply(word_count, axis=1)
df3['Word_count'] = df3.apply(word_count, axis=1)


def remove_ext_wp(df):
    # remove left and right whitespaces in character's column
    char = df['Character']
    return char.strip()


df1['Character'] = df1.apply(remove_ext_wp, axis=1)
df2['Character'] = df2.apply(remove_ext_wp, axis=1)
df3['Character'] = df3.apply(remove_ext_wp, axis=1)

total_words_1 = df1.groupby('Character').Word_count.sum()  # summing all words pronounced by each character
total_words_1 = pd.DataFrame(total_words_1)

total_words_2 = df2.groupby('Character').Word_count.sum()  # summing all words pronounced by each character
total_words_2 = pd.DataFrame(total_words_2)

total_words_3 = df3.groupby('Character').Word_count.sum()  # summing all words pronounced by each character
total_words_3 = pd.DataFrame(total_words_3)


#ajout variable freq(nombre d' interventions de chaque personnage)
df1['freq'] = df1.groupby(by='Character')['Character'].transform('count')
df2['freq'] = df1.groupby(by='Character')['Character'].transform('count')
df3['freq'] = df1.groupby(by='Character')['Character'].transform('count')



import seaborn as sns
import matplotlib.pyplot as plt

# Plots

# 1st film
# sns.set_style("dark")
total_words_1.plot(kind="bar")
plt.title("Nombre de mots prononcés par personnage dans le premier film ")
plt.xlabel("Personnages")
plt.ylabel("Nombre de mots")
# ax = sns.countplot(x = "Character", data = total_words_1)
# ax.set_xticklabels(ax.get_xticklabels(), rotation = 110, ha = "right")
# ax.set_xticklabels(ax.get_xticklabels(), fontsize = 7)
plt.tight_layout()
plt.show()

# 2nd film
# sns.set_style("dark")
total_words_2.plot(kind="bar")
plt.title("Nombre de mots prononcés par personnage dans la second film")
plt.xlabel("Personnages")
plt.ylabel("Nombre de mots")
# ax = sns.countplot(x = "Character", data = total_words_2)
# ax.set_xticklabels(ax.get_xticklabels(), rotation = 110, ha = "right")
# ax.set_xticklabels(ax.get_xticklabels(), fontsize = 7)
plt.tight_layout()
plt.show()

# 3rd film
# sns.set_style("dark")
total_words_3.plot(kind="bar")
plt.title("Nombre de mots prononcés par personnage dans le troisième film")
plt.xlabel("Personnages")
plt.ylabel("Nombre de mots")
# ax = sns.countplot(x = "Character", data = total_words_3)
# ax.set_xticklabels(ax.get_xticklabels(), rotation = 110, ha = "right")
# ax.set_xticklabels(ax.get_xticklabels(), fontsize = 7)

#1st film
#sns.set_style("dark")
total_words_1.plot(kind = "bar")
plt.title("Nombre de mots prononcés par chaque personnage dans le premier film ")
plt.xlabel("Personnages")
plt.ylabel("Nombre de mots")
#ax = sns.countplot(x = "Character", data = total_words_1)
#ax.set_xticklabels(ax.get_xticklabels(), rotation = 110, ha = "right")
#ax.set_xticklabels(ax.get_xticklabels(), fontsize = 7)
plt.tight_layout()
plt.show()

#2nd film
#sns.set_style("dark")
total_words_2.plot(kind = "bar")
plt.title("Nombre de mots prononcés par chaque personnage dans la second film")
plt.xlabel("Personnages")
plt.ylabel("Nombre de mots")
#ax = sns.countplot(x = "Character", data = total_words_2)
#ax.set_xticklabels(ax.get_xticklabels(), rotation = 110, ha = "right")
#ax.set_xticklabels(ax.get_xticklabels(), fontsize = 7)
plt.tight_layout()
plt.show()

#3rd film
#sns.set_style("dark")
total_words_3.plot(kind = "bar")
plt.title("Nombre de mots prononcés par chaque personnage dans le troisième film")
plt.xlabel("Personnages")
plt.ylabel("Nombre de mots")
#ax = sns.countplot(x = "Character", data = total_words_3)
#ax.set_xticklabels(ax.get_xticklabels(), rotation = 110, ha = "right")
#ax.set_xticklabels(ax.get_xticklabels(), fontsize = 7)

plt.tight_layout()
plt.show()

# Word cloud
from wordcloud import WordCloud
from PIL import Image
import random



def grey_color_func(word, font_size, position, orientation, random_state=None,
                    **kwargs):
    return "hsl(0, 0%%, %d%%)" % random.randint(0, 10)


text_1 = df1['Sentence_cleared'].sum()
text_1 = ' '.join(text_1)
text_2 = df2['Sentence_cleared'].sum()
text_2 = ' '.join(text_2)
text_3 = df3['Sentence_cleared'].sum()
text_3 = ' '.join(text_3)

text_wc = text_1 + text_2 + text_3

# custom_mask = np.array(Image.open("..\\Classification_Personnage_2021\\Wordcloud\\mask5.jpg"))

# Local Path :
custom_mask = np.array(Image.open("../Wordcloud/mask5.jpg"))
wc = WordCloud(width=1600, height=800, background_color="white",
               mask=custom_mask, color_func=grey_color_func, max_words=500)
wc.generate(text_wc)
plt.figure(figsize=(20, 10))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()
# It's a nice wordcloud, congrats !

wc = WordCloud(width=1600, height=800, background_color="white", mask=custom_mask,
                color_func=grey_color_func, max_words=500)
wc.generate(text_wc)
plt.figure(figsize = (20,10))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()




if windows:
    df_char = pd.read_csv('../Data/Characters_filtered.csv', sep=r'\;')
else:
    df_char = pd.read_csv('..\\Classification_Personnage_2021\\Data\\Characters_filtered.csv', sep=r'\;')

# Ajout des variables de comptage
import string

def name_capitalizer(df):
    name = df.Character
    return string.capwords(name, sep = None)

## Film 1
total_words_1 = total_words_1.rename_axis('Character').reset_index()
total_words_1['Character'] = total_words_1.apply(name_capitalizer, axis = 1)
df_char = df_char.assign(Word_count_1=0)
for i in range(len(total_words_1['Character'])):
    for j in range(len(df_char['Name'])):
        if total_words_1['Character'][i] in df_char['Name'][j]:
            df_char['Word_count_1'][j] = total_words_1['Word_count'][i]

## Film 2
total_words_2 = total_words_2.rename_axis('Character').reset_index()
total_words_2['Character'] = total_words_2.apply(name_capitalizer, axis = 1)
df_char = df_char.assign(Word_count_2=0)
for i in range(len(total_words_2['Character'])):
    for j in range(len(df_char['Name'])):
        if total_words_2['Character'][i] in df_char['Name'][j]:
            df_char['Word_count_2'][j] = total_words_2['Word_count'][i]

## Film 3
total_words_3 = total_words_3.rename_axis('Character').reset_index()
total_words_3['Character'] = total_words_3.apply(name_capitalizer, axis = 1)
df_char = df_char.assign(Word_count_3=0)
for i in range(len(total_words_3['Character'])):
    for j in range(len(df_char['Name'])):
        if total_words_3['Character'][i] in df_char['Name'][j]:
            df_char['Word_count_3'][j] = total_words_3['Word_count'][i]


# Exportation des différents dataframe

df1['Character'] = df1.apply(name_capitalizer, axis = 1)
df2['Character'] = df2.apply(name_capitalizer, axis = 1)
df3['Character'] = df3.apply(name_capitalizer, axis = 1)
df_char['Name'] = df_char.apply(name_capitalizer, axis = 1)


df1.to_csv(r'..\\Classification_Personnage_2021\\Data\\HP1_preprocessed.csv', sep=';')
df2.to_csv(r'..\\Classification_Personnage_2021\\Data\\HP2_preprocessed.csv', sep=';')

df_HP1 = pd.read_csv('..\\Classification_Personnage_2021\\Data\\HP1_preprocessed.csv', sep=r'\;')
df_HP2 = pd.read_csv('..\\Classification_Personnage_2021\\Data\\HP2_preprocessed.csv', sep=r'\;')
## Local path
df_HP1 = pd.read_csv('../Data/HP1_preprocessed.csv', sep=r'\;', encoding='unicode_escape')
df_HP2 = pd.read_csv('../Data/HP2_preprocessed.csv', sep=r'\;', encoding='unicode_escape')
### Attention, en important ces nouveaux csv cela change les types des données dans les dataframes exportés
### et ça pose problème dans la suite (autant garder ceux avant exportation)

# Speech splitting
df1_data = df1[['Character', 'Sentence_cleared']]
df2_data = df2[['Character', 'Sentence_cleared']]
df_data = pd.concat([df1_data, df2_data], ignore_index = True, sort = False)

persos = set(df_data.Character.values)
print(persos)
list_speech = []
for perso in persos :
    t  = list(df_data[df_data.Character == perso].Sentence_cleared.values) #array of perso's speech
    #film_data[film_data.Character.str.contains("perso")]
    list_speech.append([item for sublist in t for item in sublist])
df_words = pd.DataFrame(list(zip(persos, list_speech)), columns = ['Character', 'Words'])


# Work on characters' df
## Nouvelle catégorie pour ceux qui n'ont pas de maison ou qui sont d'une autre école
houses = ['Gryffindor', 'Ravenclaw', 'Slytherin', 'Hufflepuff']
for i in range(len(df_char['House'])):
    if df_char['House'][i] not in houses:
        df_char['House'][i] = 'Other'

## Suppresion des colonnes qui ne nous intéressent pas
df_char = df_char.drop(["Job", "Wand", "Patronus", "Species", "Hair colour", "Eye colour", "Birth", "Death"], axis = 1)
## df exported as Character_preprocessed


data_char = pd.read_csv('../Data/Character_preprocessed.csv', sep=r'\;', encoding='unicode_escape')
data_char['Character'] = data_char.apply(name_capitalizer, axis = 1)

df_words_single = pd.read_csv('../Data/Words_by_character_modified.csv', sep=r'\;', encoding='unicode_escape')
df_words_single = df_words_single.groupby(['Character']).sum()
df_words_single = df_words_single.rename_axis('Character').reset_index()

## Most common words by character
def common_words(df):
    # return most common words by character
    text = df['speech']
    text = text.replace(", ", " ")
    text = text.replace("'", "")
    text = ''.join(text)
    word_list = Counter(text.split()).most_common(100)
    return (word_list)
df_words_single['Most_common'] = df_words_single.apply(common_words, axis = 1)
df_words_single['Character'] = df_words_single.apply(name_capitalizer, axis = 1)


character_all = data_char.Character.values
common_all = []
for char in character_all :
    t  = list(df_words_single[df_words_single.Character == char].Most_common.values)
    common_all.append([item for sublist in t for item in sublist])
df_common_all = pd.DataFrame(list(zip(character_all, common_all)), columns = ['Character', 'Most_common'])


# Mots les plus communs
## Overall
np.unique(text_wc).shape
common_words = Counter(text_wc.split()).most_common(20)
## HP 1
common_words_1 = Counter(text_1.split()).most_common(20)
## HP 2
common_words_2 = Counter(text_2.split()).most_common(20)
## HP 3
common_words_3 = Counter(text_3.split()).most_common(20)
## HP 1+2
most_common = Counter((text_1+text_2).split()).most_common(20)
word_common = [most_common[i][0] for i in range(len(most_common))]

# characters most common words in the two first films ones
tteesstt = []
for i in range(len(df_common_all)):
    list_tuple = df_common_all.Most_common[i]
    test = []
    for j in range(len(list_tuple)):
        tuple_word = list_tuple[j]
        word = tuple_word[0]
        if word in word_common:
            test.append(tuple_word)
    tteesstt.append(test)
df_common_all['In film_common'] = tteesstt
df_film_common = df_common_all.drop(['Most_common'], axis = 1)


data_char = pd.merge(data_char, df_film_common, on = ['Character'])
