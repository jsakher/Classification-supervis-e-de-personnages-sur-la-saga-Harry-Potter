import pandas as pd
import matplotlib.pyplot as plt 
pd.options.display.max_rows
# pip install funpymodeling
from funpymodeling.exploratory import freq_tbl


#df_char = pd.read_csv('/Users/hayou/Desktop/S2/projet/Classification_Personnage_2021/Data/Characters.csv', sep = ';', encoding= 'unicode_escape')

df1=pd.read_csv('Classification_Personnage_2021/Data/Harry Potter 1.csv', sep = ';', encoding= 'unicode_escape')
# df1 = pd.read_csv('../Data/Harry Potter 1.csv', sep = ';', encoding= 'unicode_escape')
# Tiffany : c'est la 2ème commande qui marche pour mon pc, la 1ère non.

fq=freq_tbl(df1["Character"])
# Cette fonction est interessante
# Mais certains personnages apparaissent plusieurs fois
# Avez-vous remarqué ? Qu'en pensez-vous ?
# NB : ce code n'est pas PEP8, pensez au linter ;)

df1.Character=df1.Character.str.strip()
print(df1.Character.value_counts())
ax =df1.Character.value_counts().plot.bar()
ax.set_ylabel('nombre fois')
ax.set_xlabel('Character')
plt.show()