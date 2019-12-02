# %%
# DATA IMPORT
import pandas as pd
import numpy as np

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
# TEST/TRAIN SPLIT
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
tf = pd.DataFrame(X_train_tf.toarray(), \
    columns=count_vect.get_feature_names(), \
        index=X_train.index)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
tfidf = pd.DataFrame(X_train_tfidf.toarray(), \
    columns=count_vect.get_feature_names(), \
        index=X_train.index)


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
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from IPython.display import clear_output

def performGridSearch(model,X,y):
    if isinstance(model,DecisionTreeClassifier): # 50 = 2 x 5 x 5
        print("DecisionTreeClassifier")
        param_grid = [
            {'criterion':['gini','entropy'],
            'max_depth':[5,10,50,100,150]}
        ]
    elif isinstance(model,BaggingClassifier): # 800 = 2 x 5 x 4 x 4 x 5
        print("BaggingClassifier")
        param_grid = [
            {'base_estimator__criterion':['gini','entropy'],
            'base_estimator__max_depth':[5,10,50,100,150], 
            'max_samples' : [0.05, 0.1, 0.2, 0.5],
            'n_estimators': [10, 50, 100, 200]}
        ]
    elif isinstance(model,RandomForestClassifier): # 160 = 2 x 4 x 4 x 5
        print("RandomForestClassifier")
        param_grid = [
            {'criterion':['gini','entropy'],
            'n_estimators': [10, 50, 100, 200], 
            'max_features': [10,100,1000,"auto"]}# ,
        ]
    elif isinstance(model,GradientBoostingClassifier): # 640 = 2 x 4 x 4 x 4 x 5
        print("GradientBoostingClassifier")
        param_grid = [
            {'loss':['deviance', 'exponential'],
            'learning_rate':[0.1,1e-2,1e-3,1e-4],
            'n_estimators': [10, 50, 100, 200], 
            'max_features': [10,100,1000,"auto"]}
        ]
    grid_search = GridSearchCV(model, param_grid, cv=5,
                            scoring='neg_mean_squared_error', 
                            return_train_score=True,verbose=10,
                            n_jobs=-1)
    a = grid_search.fit(X, y)
    return grid_search.best_params_, grid_search.best_estimator_

models = {
    "bag":BaggingClassifier(DecisionTreeClassifier(random_state=42), n_jobs=-1, random_state=42),
    "rfc":RandomForestClassifier(random_state=42),
    "gbc":GradientBoostingClassifier(random_state=42),
    "dtc":DecisionTreeClassifier(random_state=42)
}
best_params = dict()

for k,v in models.items():
    clear_output()
    print(k)
    param,est = performGridSearch(v,tfidf,y_train)
    best_params[k] = dict(param=param,est=est)

#%%
# BEST ACCURACY
# for each model check test/train accuracy
for k,v in best_params:


# %%
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

# PRECISION/RECALL PLOT OF BEST
def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])

plt.figure(figsize=(8, 6))
plot_precision_vs_recall(precisions, recalls)
save_fig("precision_vs_recall_plot")
plt.show()