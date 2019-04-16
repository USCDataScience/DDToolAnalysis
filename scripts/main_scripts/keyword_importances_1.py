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



# find importances for keywords from each topic
topics=open(data_dir+'topic_keywords.txt','r').read().split('====\n')
for topic_num, topic in enumerate(topics):
    keywords=topic.strip().split('\n')
    # topical keywords model
    X_keywords=[]
    for x in X_text:
        vector=[]
        for i, keyword in enumerate(keywords):
            vector.append(float(len([m.start() for m in re.finditer(keyword.lower(),x.lower())])))
        X_keywords.append(vector)
    X_keywords=np.array(X_keywords)
    X=X_keywords
    print(np.array(X).shape)
    importances_topical_keywords=find_importances(X,y,keywords,splits)

    # save stuff

    savemodel(importances_topical_keywords, temp_dir+'importances_topical_keywords')

    # load stuff

    importances_topical_keywords=loadmodel(temp_dir+'importances_topical_keywords')

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


    topical_keywords=list(map(lambda x:x.lower(),list(importances_topical_keywords.index)))
    topical_keyword_importances=list(importances_topical_keywords['Importance'])
    topical_keyword_importance_lookup={k:v for k,v in zip(topical_keywords, topical_keyword_importances)}




    color=[usual_color]*len(topical_keywords)
    plot_plot(topical_keywords,topical_keyword_importances, color, output_dir+'importances_topical_keyword_'+str(topic_num))

