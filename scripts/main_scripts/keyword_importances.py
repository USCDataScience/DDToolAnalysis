import json

import plotly.graph_objs as go
from plotly.offline import plot
# from voting_classifier import VotingClassifier
from rfpimp import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedShuffleSplit
from utils.utils import *

from utils.feature_importance import find_importances

output_dir='output/'
temp_dir='temp/'
data_dir='../../data/'

# ======= read the files
final_json=open(data_dir+'400_final_json.json')
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



# ======= squish some labels together:
y_new=[]
for label in y:
    if label in [2,3,4]:
        label=3
    y_new.append(label)
y=y_new
y=np.array(y)



# ======= create splits
splits = StratifiedShuffleSplit(n_splits=50, test_size=0.3, random_state=0)


# ======= keywords model
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

# ======= topical keywords model
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

# ======= n-gram models
count_vect = TfidfVectorizer(ngram_range=(1,2), stop_words='english',max_df=0.8, min_df=3, max_features=5000)
X_gram = count_vect.fit_transform(X_text)
X=X_gram.todense()
print(X.shape)
importances_n_gram=find_importances(X,y,count_vect.get_feature_names(),splits)

# ======= save stuff

savemodel(importances_n_gram,temp_dir+'importances_n_gram')
savemodel(importances_keywords, temp_dir+'importances_keywords')
savemodel(importances_topical_keywords, temp_dir+'importances_topical_keywords')

# ======= load stuff

importances_n_gram=loadmodel(temp_dir+'importances_n_gram')
importances_keywords=loadmodel(temp_dir+'importances_keywords')
importances_topical_keywords=loadmodel(temp_dir+'importances_topical_keywords')

#  ======= plot stuff:
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

plot_plot(keywords,keyword_importances,color,data_dir+'importances_keyword')



color=[usual_color]*len(topical_keywords)
plot_plot(topical_keywords,topical_keyword_importances,color,data_dir+'importances_topical_keyword')


color=[]
color_filtered=[]
x=[]
y=[]
for n_gram in n_grams:
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
plot_plot(x,y,color_filtered,data_dir+'importances_n_gram_filtered')
plot_plot(n_grams,n_gram_importances,color,data_dir+'importances_n_gram')

