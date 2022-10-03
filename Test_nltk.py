import nltk
nltk.download()

from nltk.corpus import stopwords
print(set(stopwords.words('English'))) # stop words listing

# Example :
from nltk.tokenize import word_tokenize # Tokenizing a string that represents a sentence splits the sentence into a list of words.

text = 'In this tutorial, I\'m learning NLTK. It is an interesting platform.'
stop_words = set(stopwords.words('english'))
words = word_tokenize(text)

new_sentence = []

for word in words:
    if word not in stop_words:
        new_sentence.append(word)

print(new_sentence)

