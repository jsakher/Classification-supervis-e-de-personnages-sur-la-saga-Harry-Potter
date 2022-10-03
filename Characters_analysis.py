import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#target variable "House" 
df_char = pd.read_csv('Data\Characters.csv', sep = r'\;')
df_char = df_char.drop(df_char.columns[[5, 6, 13, 14]], axis=1) 
df6=df_char.dropna(subset=['House'])
df6.head(n=10)
df2=df6[((df_char.House != 'Beauxbatons Academy of Magic') &( df_char.House != 'Beauxbatons Academy of Magic') & (df_char.House != 'Durmstrang Institute') & (df_char.House != 'House elf'))]



# cleaning data could be done with NLTK but for the sake of learning it's interesting to fix it without it for this data
df2.isnull().sum()
df2[["Job", "Blood status", "Species"]] = df6[["Job", "Blood status", "Species"]].replace(["\xa0", "\x96", "\n", "\["], " ", regex=True)
df1 = df2.apply(lambda x: x.str.strip() if x.dtypes == "O" else x) # removing white spaces
df1["Blood status"].unique()

df1["Blood status"] = df1["Blood status"].str.title()# Pure-blood or half-blood = Pure-blood or Half-blood

df1["Blood status"].unique()# there are two spellings for this one, so I decided to title case them

df1.to_csv('df_char1.csv', index=False)
print(df1)

# overall gender percentage
size = df1.groupby("Gender").size()
labels = size.index

# pie chart
fig, ax = plt.subplots(figsize=(10, 5))

patches, texts, autotexts = ax.pie(size, colors=["#f96d80", "#1b6ca8"], autopct='%1.1f%%', startangle=90)

for auto in autotexts:
    auto.set_color("white")

ax.legend(patches, labels,
          title="Gender",
          loc="best")

plt.setp(autotexts, size=12, weight="bold")

ax.axis('equal')
plt.tight_layout()
plt.show()

# gender based on houses

plt.figure(figsize=(15, 6))
a = sns.countplot(x="House", hue="Gender", palette=["#1b6ca8", "#f96d80"], data=df1)
a.legend(loc="upper right")
plt.title("Houses per Gender")
plt.xlabel("House")
plt.show()

#Blood status
pd.crosstab(df6["Blood status"], df1["House"]).plot(stacked=True, figsize=(10, 6), kind = 'barh', color= ["#f96d80", "#1b6ca8", "#AE0001", "#FFDB00", "#222F5B", "#06b300"])
plt.title("Blood Status per House")
plt.show()

