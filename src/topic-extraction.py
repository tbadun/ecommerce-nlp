# %%
# DATA IMPORT
import pandas as pd
import numpy as np
import os

os.chdir("/Users/tess/Desktop/Desktop/SCSML/assignments/final project/ecommerce-nlp/src")
df = pd.read_csv("../data/Womens Clothing E-Commerce Reviews.csv")
df.columns = ["clothing_id","age","title","review","rating", \
            "recomend","pos_fb_ct","division","department","class"]

df = df[[i is not np.nan for i in df.review]]

# %%
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
        word2 = re.sub('[^A-Za-z0-9$]+',"",word)
        if len(word2)>0 and not any(i.isnumeric() for i in word2):
            stem_sentence.append(porter.stem(word2))
            stem_sentence.append(" ")
    return "".join(stem_sentence)


df['review_lemma'] = df.review.apply(lambda x: lemmatizeSentence(x))

# %%
# VECTORIZED VOCABULARY
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
df_counts = count_vect.fit_transform(df.review_lemma)
vocab = pd.DataFrame(df_counts.toarray(), \
    columns=count_vect.get_feature_names(), \
        index=df.index)

# %%
# TERM FREQUENCY
from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer(use_idf=False).fit(df_counts)
df_tf = tf_transformer.transform(df_counts)
tf = pd.DataFrame(df_tf.toarray(), \
    columns=count_vect.get_feature_names(), \
        index=df.index)

tfidf_transformer = TfidfTransformer()
df_tfidf = tfidf_transformer.fit_transform(df_counts)
tfidf = pd.DataFrame(df_tfidf.toarray(), \
    columns=count_vect.get_feature_names(), \
        index=df.index)

# %%
# TEST/TRAIN SPLIT
X = tfidf.copy()
y = df["recomend"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42,stratify=list(y))


# "PREDICT" RECOMMENDATION TO GATHER IMPORTANT TERMS
# TRY MODELS:
# - GradientBoostingClassifier
# - RandomForestClassifier
# - DecisionTreeClassifier
# - DecisionTreeClassifier w/ Bagging
# Grid search each
# For best try StatifiedKFold # from sklearn.model_selection import StratifiedKFold

# %%
# GRID SEARCH
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from IPython.display import clear_output
from scipy.stats import randint, rv_continuous

np.random.seed(38)

def performGridSearch(model,X,y):
    if isinstance(model,DecisionTreeClassifier):
        print("DecisionTreeClassifier")
        param_grid = {'criterion':['gini','entropy'],
            'max_depth':randint(5,200)}
    elif isinstance(model,BaggingClassifier):
        print("BaggingClassifier")
        param_grid = {'base_estimator__criterion':['gini','entropy'],
            'base_estimator__max_depth':randint(5,200), 
            'max_samples' : rv_continuous(a=0.01,b=1),
            'max_features': randint(5,500),
            'n_estimators': randint(5,200)}
    elif isinstance(model,RandomForestClassifier):
        print("RandomForestClassifier")
        param_grid = {'criterion':['gini','entropy'],
            'n_estimators': randint(5,200), 
            'max_features': randint(5,500)}
    elif isinstance(model,GradientBoostingClassifier):
        print("GradientBoostingClassifier")
        param_grid = {'loss':['deviance', 'exponential'],
            'learning_rate':rv_continuous(a=1e-5,b=1),
            'n_estimators': randint(5,200), 
            'max_features': randint(5,500)}
    grid_search = RandomizedSearchCV(model, param_grid, cv=5,
                            scoring='neg_mean_squared_error', 
                            random_state=55,verbose=10,
                            n_jobs=2,n_iter=10)
    a = grid_search.fit(X, y)
    return grid_search.best_params_, grid_search.best_estimator_

models = {
    "rfc":RandomForestClassifier(random_state=42),
    "gbc":GradientBoostingClassifier(random_state=42),
    "dtc":DecisionTreeClassifier(random_state=42),
    "bag":BaggingClassifier(DecisionTreeClassifier(random_state=42), n_jobs=2, random_state=42)
}
best_params = dict()

for k,v in models.items():
    clear_output()
    print(k)
    if k == "bag":
        models[k] = BaggingClassifier(DecisionTreeClassifier(random_state=42,**best_params["dtc"]["param"]), n_jobs=2, random_state=42)
    param,est = performGridSearch(v,X_train,y_train)
    best_params[k] = dict(param=param,est=est)

#%%
# BEST ACCURACY
# for each model check test/train accuracy
from sklearn.metrics import precision_score, recall_score

accuracy = []
for k,v in best_params:
    row = []
    print("%s: %s".format(k,str(v['param'])))
    train_pred = v['est'].predict(X_train)
    test_pred = v['est'].predict(X_test)
    row.append(precision_score(y_train,train_pred))
    row.append(recall_score(y_train,train_pred))
    row.append(precision_score(y_test,test_pred))
    row.append(recall_score(y_test,test_pred))
    accuracy.append(row)


# %%
acc = pd.DataFrame(accuracy,
                columns=["train_precision",
                        "train_recall",
                        "test_precision",
                        "test_recall"],
                index=list(models.keys())) # double check this is the rigt orientation
acc

#%%
# SELECT BEST
from sklearn.metrics import precision_recall_curve

model = best_params[BEST]["est"]
test_pred = model.predict(X_test)
precisions, recalls, thresholds = precision_recall_curve(y_train, test_pred)

# %%
# PRECISION/RECALL PLOT OF BEST
import matplotlib.pyplot as plt

def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])

plt.figure(figsize=(8, 6))
plot_precision_vs_recall(precisions, recalls)
plt.show()

