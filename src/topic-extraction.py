# %%
# DATA IMPORT
import pandas as pd
import numpy as np

df = pd.read_csv("../data/Womens Clothing E-Commerce Reviews.csv")
df.columns = ["clothing_id","age","title","review","rating", \
            "recomend","pos_fb_ct","division","department","class"]

df = df[[i is not np.nan for i in df.review]]

#%%
# TERM LEMMATIZATION/STEMMING
# NOTE: DOWNLOAD corpora/wordnet AND models/punkt
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
import re 

def lemmatizeSentence(sentence):
    token_words=word_tokenize(sentence)
    porter=PorterStemmer()
    stem_sentence=[]
    for word in token_words:
        word2 = re.sub('[^A-Za-z0-9]+',"",word)
        if len(word2)>0 and not any(i.isnumeric() for i in word2):
            stem_sentence.append(porter.stem(word2))
            stem_sentence.append(" ")
    return "".join(stem_sentence)


df['review_lemma'] = df.review.apply(lambda x: lemmatizeSentence(x))

# %%
X = df.drop("recomend",axis=1)
y = df["recomend"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42,stratify=list(y))

# %%
# VECTORIZED VOCABULARY
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train.review_lemma)
vocab = pd.DataFrame(X_train_counts.toarray(), \
    columns=count_vect.get_feature_names(), \
        index=X_train.index)

# %%
# TERM FREQUENCY
from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# %%
tf = pd.DataFrame(X_train_tf.toarray(), \
    columns=count_vect.get_feature_names(), \
        index=X_train.index)
tfidf = pd.DataFrame(X_train_tfidf.toarray(), \
    columns=count_vect.get_feature_names(), \
        index=X_train.index)

# %%
