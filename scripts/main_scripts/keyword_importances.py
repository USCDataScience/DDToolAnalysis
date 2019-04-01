from openpyxl.chart import marker
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
from sklearn.neural_network import MLPClassifier
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import w2v
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
import re
from cosine_classifier import CosineClassifier
# from voting_classifier import VotingClassifier
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
import dill

from rfpimp import *


def decimalToBinary(n):
    return list(map(float,format(n, '04b')))

# https://stats.stackexchange.com/questions/221715/apply-word-embeddings-to-entire-document-to-get-a-feature-vector
# https://stackoverflow.com/questions/45074579/votingclassifier-different-feature-sets
# https://www.ritchieng.com/machinelearning-learning-curve/
# https://www.dataquest.io/blog/learning-curves-machine-learning/
# https://www.kaggle.com/den3b81/better-predictions-stacking-with-votingclassifier
# https://mlwave.com/kaggle-ensembling-guide/
# http://blog.kaggle.com/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/
# https://markhneedham.com/blog/2017/12/10/scikit-learn-using-gridsearch-tune-hyper-parameters-votingclassifier/

# wordlist from: http://www-personal.umich.edu/~jlawler/wordlist

# https://explained.ai/rf-importance/index.html

def savemodel(model,outfile):
    with open(outfile, 'wb') as output:
        dill.dump(model, output)
    return ''

def loadmodel(infile):
    model=''
    with open(infile, 'rb') as inp:
        model = dill.load(inp)
    return model


def clean(arr):
    arr=arr.replace('. ',' . ')
    arr = arr.replace(', ', ' , ')

    arr=arr.split('\n')
    arr_new=[]
    for s in arr:
        if len(s.split())<3:
            s=''
        if len(s)>=3 and s[0]=='<':
            s=''
        if 'login' in s or 'cookie' in s:
            s=''

        s_new=[]
        for w in s.split():
            if re.match("^[a-zA-Z]*$", w):
                s_new.append(w)
        s=' '.join(s_new)

        if s=='':
            continue
        else:
            arr_new.append(s)
    return '\n'.join(arr_new)

def create_single_data(X, y,splits,estimator, title, color, marker, load=False):
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    # cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    if load:
        model=loadmodel(title)
        train_sizes,train_scores,valid_scores=model[0],model[1],model[2]
    else:
        train_sizes, train_scores, valid_scores = learning_curve(estimator, X, y, train_sizes=range(10,407,20), cv=splits, scoring='accuracy')
        # train_sizes, train_scores, valid_scores = learning_curve(estimator, X, y, train_sizes=range(10, 407,200), cv=splits,
        #                                                          scoring='accuracy')
        savemodel((train_sizes, train_scores, valid_scores),title)

    trace = go.Scatter(
        x=train_sizes,
        y=list(map(lambda x:x*100, np.mean(valid_scores, axis=1))),
        legendgroup=title,
        # mode='lines+markers',
        # marker=dict(
        #     color=color,
        #     ),

        line = dict(
        # color = color,
        dash = marker),
        name=title
    )
    # if marker!='':
    #     trace['marker']['symbol']=marker
    return trace


def find_importances(X,y,columns,splits):
    estimator = RandomForestClassifier(n_estimators=100)
    dfs = []
    for train_index, test_index in splits.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        estimator.fit(X_train, y_train)
        X_test = pd.DataFrame(X_test)
        X_test.columns = columns
        imp = importances(estimator, X_test, y_test)
        dfs.append(imp)

    df = pd.concat(dfs).groupby('Feature', as_index=True).mean().sort_values(by='Importance', ascending=False)
    return df


# read the json
final_json=json.loads(open('final_json.json','r').read())
more_final_json=json.loads(open('400_final_json.json','r').read())
remaining_final_json=json.loads(open('remaining_400_final_json.json','r').read())
still_final_json=json.loads(open('still_remaining_400_final_json.json','r').read())
still_remaining_final_json=json.loads(open('still_remaining_200_final_json.json','r').read())

X_text=[]
y=[]
urls=[]
for jsn in final_json:
    content=clean(jsn['content'])
    if content == '':
        continue
    X_text.append(content)
    y.append(float(jsn['class']))
    urls.append(jsn['url'])
# print(len(final_json))

for jsn in more_final_json:
    content = clean(jsn['content'])
    if content == '':
        continue
    X_text.append(content)
    y.append(float(jsn['class']))
    urls.append(jsn['url'])
# print(len(more_final_json))

for jsn in remaining_final_json:
    content = clean(jsn['content'])
    if content == '':
        continue
    X_text.append(content)
    y.append(float(jsn['class']))
    urls.append(jsn['url'])
# print(len(remaining_final_json))

for jsn in still_final_json:
    content = clean(jsn['content'])
    if content == '':
        continue
    X_text.append(content)
    y.append(float(jsn['class']))
    urls.append(jsn['url'])

for jsn in still_remaining_final_json:
    content = clean(jsn['content'])
    # print(content.replace('\n',' '))
    if content == '':
        continue
    X_text.append(content)
    y.append(float(jsn['class']))
    urls.append(jsn['url'])


# squish some labels together:
y_new=[]
for label in y:
    if label in [2,3,4]:
        label=3
    y_new.append(label)
y=y_new
y=np.array(y)


data={}


# for a specific train-test split fit the model using test, then find the feature importance using the library


splits = StratifiedShuffleSplit(n_splits=50, test_size=0.3, random_state=0)


# keywords model
X_keywords=[]
keywords=[keyword.strip() for keyword in open('keywords.txt','r').readlines()]
for x in X_text:
    vector=[]
    for i, keyword in enumerate(keywords):
        vector.append(float(len([m.start() for m in re.finditer(keyword.lower(),x.lower())])))
    X_keywords.append(vector)
X_keywords=np.array(X_keywords)
X=X_keywords
print(np.array(X).shape)
importances_keywords=find_importances(X,y,keywords,splits)

# topical keywords model
X_keywords=[]
keywords=[keyword.strip() for keyword in open('topic_keywords.txt','r').readlines()]
for x in X_text:
    vector=[]
    for i, keyword in enumerate(keywords):
        vector.append(float(len([m.start() for m in re.finditer(keyword.lower(),x.lower())])))
    X_keywords.append(vector)
X_keywords=np.array(X_keywords)
X=X_keywords
print(np.array(X).shape)
importances_topical_keywords=find_importances(X,y,keywords,splits)

# n-gram models
count_vect = TfidfVectorizer(ngram_range=(1,2), stop_words='english',max_df=0.8, min_df=3, max_features=5000)
X_gram = count_vect.fit_transform(X_text)
X=X_gram.todense()
print(X.shape)
importances_n_gram=find_importances(X,y,count_vect.get_feature_names(),splits)

# save stuff

savemodel(importances_n_gram,'importances_n_gram')
savemodel(importances_keywords, 'importances_keywords')
savemodel(importances_topical_keywords, 'importances_topical_keywords')

# load stuff

importances_n_gram=loadmodel('importances_n_gram')
importances_keywords=loadmodel('importances_keywords')
importances_topical_keywords=loadmodel('importances_topical_keywords')

# plot stuff:
def plot_plot(x, y, color, filename):

    data = [go.Bar(
            x=x,
            y=y,
        marker=dict(
            color=color
        )
        )]
    plot(data, filename=filename)

usual_color='rgba(204,204,204,1)'
spl_color_pos='rgb(46,139,87)'
spl_color_neg='rgba(222,45,38,0.8)'

keywords=list(map(lambda x:x.lower(),list(importances_keywords.index)))
keyword_importances=list(importances_keywords['Importance'])
keyword_importance_lookup={k:v for k,v in zip(keywords,keyword_importances)}

topical_keywords=list(map(lambda x:x.lower(),list(importances_topical_keywords.index)))
topical_keyword_importances=list(importances_topical_keywords['Importance'])
topical_keyword_importance_lookup={k:v for k,v in zip(topical_keywords, topical_keyword_importances)}


n_grams=list(map(lambda x:x.lower(),list(importances_n_gram.index)))
n_gram_importances=list(importances_n_gram['Importance'])
n_gram_importance_lookup={k:v for k,v in zip(n_grams,n_gram_importances)}


color=[]
for keyword in keywords:
    importance=keyword_importance_lookup[keyword]
    if importance > 0:
        color.append(spl_color_pos)
    else:
        color.append(spl_color_neg)

plot_plot(keywords,keyword_importances,color,'importances_keyword')



color=[usual_color]*len(topical_keywords)
plot_plot(topical_keywords,topical_keyword_importances,color,'importances_topical_keyword')


color=[]
color_filtered=[]
x=[]
y=[]
for n_gram in n_grams:
    # if n_gram=='marine':
    #     print(n_gram_importance_lookup['ocean'])

    if n_gram in keywords:
        keyword_importance=keyword_importance_lookup[n_gram]
        if keyword_importance>0:
            color.append(spl_color_pos)
            color_filtered.append(spl_color_pos)
        else:
            color.append(spl_color_neg)
            color_filtered.append(spl_color_neg)
        x.append(n_gram)
        y.append(n_gram_importance_lookup[n_gram])
    else:
        color.append(usual_color)
plot_plot(x,y,color_filtered,'importances_n_gram_filtered')
plot_plot(n_grams,n_gram_importances,color,'importances_n_gram')


# from the above plots you can see what things the models agree on!!
# shows the relative predictability of features in a model
# when two features are collinear; when one of them is dropped, the other fulfills the duty of the dropped one,
#  hence seeming that the dropped on had no importance! In terms of n_grams the co-ocurance is something that may do this??!!
