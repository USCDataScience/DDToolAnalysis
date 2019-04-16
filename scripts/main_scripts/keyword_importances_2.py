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

# ======= read data
mod=json.loads(open(data_dir+'ruth_model','r').read())
y=np.array(list(map(int, mod['labeled'])))
X_text=mod['url_text']



# ======= for a specific train-test split fit the model using test, then find the feature importance using the library
splits = StratifiedShuffleSplit(n_splits=50, test_size=0.3, random_state=0)



# ======= n-gram models
count_vect = TfidfVectorizer(ngram_range=(1,2), stop_words='english',max_df=0.8, min_df=3, max_features=5000)
X_gram = count_vect.fit_transform(X_text)
X=X_gram.todense()
print(X.shape)
importances_n_gram=find_importances(X,y,count_vect.get_feature_names(),splits)

# ======= save

savemodel(importances_n_gram, temp_dir+'importances_n_gram_ruth')

# # ======= load
#
# importances_n_gram=loadmodel(temp_dir+'importances_n_gram_ruth')

# ======= plot :
def plot_plot(x, y, color, filename):

    data = [go.Bar(
            x=x,
            y=y,
        marker=dict(
            color=color
        )
        )]
    plot(data, filename=filename)

# ======= select colors to use
usual_color='rgba(204,204,204,1)'
spl_color_pos='rgb(46,139,87)'
spl_color_neg='rgba(222,45,38,0.8)'


n_grams=list(map(lambda x:x.lower(),list(importances_n_gram.index)))
n_gram_importances=list(importances_n_gram['Importance'])
n_gram_importance_lookup={k:v for k,v in zip(n_grams,n_gram_importances)}


color=len(n_gram_importances)*[usual_color]
plot_plot(n_grams,n_gram_importances,color,output_dir+'importances_n_gram_ruth')

