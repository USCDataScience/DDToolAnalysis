import plotly.graph_objs as go
from plotly.offline import plot
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import learning_curve

from utils.utils import *


# from voting_classifier import VotingClassifier


def create_single_data(X, y, splits, estimator, title, color, marker, load=False):

    # to prepare for plotting a single model

    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    # cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    if load:
        model=loadmodel(title)
        train_sizes,train_scores,valid_scores=model[0],model[1],model[2]
    else:
        train_sizes, train_scores, valid_scores = learning_curve(estimator, X, y, train_sizes=range(10,407,20), cv=splits, scoring='accuracy')
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
        color = color,
        dash = marker),
        name=title
    )
    # if marker!='':
    #     trace['marker']['symbol']=marker
    return trace


# ======= read the files
data_dir='../../data/'
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


# collect all the plotting traces here:
data={}


# ======= create splits
splits = StratifiedShuffleSplit(n_splits=50, test_size=0.3, random_state=0)
splits=list(splits.split(np.zeros(len(y)),y))




# ======= n-gram models
count_vect = TfidfVectorizer(ngram_range=(1,2), stop_words='english',max_df=0.8, min_df=3, max_features=5000)
X_gram = count_vect.fit_transform(X_text)
X=X_gram
print(X.todense().shape)


count_vect_inv_vocab={l:k for k,l in count_vect.vocabulary_.items()}

estimators=[RandomForestClassifier(n_estimators=100),]
titles=['Random Forest']
colors=['rgb(44, 160, 44)' ]
legend_group=[3]

line_type='solid'
algorithm_variation=' (N-gram)'


for estimator,title, color, group in zip(estimators,titles, colors,legend_group):
    print(title+ algorithm_variation)
    title=title+ algorithm_variation
    if group not in data.keys():
        data[group]=[]
    data[group].append(create_single_data(X,y, splits,estimator,title,color,line_type))


# ======= keywords input

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


estimators=[RandomForestClassifier(n_estimators=100)]
titles=['Random Forest']
colors=['rgb(44, 160, 44)' ]
legend_group=[3]
line_type='dot'
algorithm_variation=' (Keywords)'


for estimator,title, color, group in zip(estimators,titles, colors,legend_group):
    print(title+ algorithm_variation)
    title=title+ algorithm_variation
    if group not in data.keys():
        data[group]=[]
    data[group].append(create_single_data(X,y, splits,estimator,title,color,line_type))


# ======= topic modelling keyword input


topics=open('topic_keywords.txt','r').read().split('====\n')
for topic_num,topic in enumerate(topics):
    keywords=topic.strip().split('\n')

    X_keywords = []
    for x in X_text:
        vector=[]
        for i, keyword in enumerate(keywords):
            vector.append(float(len([m.start() for m in re.finditer(keyword.lower(),x.lower())])))
        X_keywords.append(vector)
    X_keywords=np.array(X_keywords)
    X=X_keywords
    print(np.array(X).shape)


    estimators=[RandomForestClassifier(n_estimators=100)]
    titles=['Random Forest']
    colors=['rgb(44, 160, 44)']
    legend_group=[3]



    line_type='dot'
    algorithm_variation=' (topical Keywords) - topic'+ str(topic_num)


    for estimator,title, color, group in zip(estimators,titles, colors,legend_group):
        print(title+ algorithm_variation)
        title=title+ algorithm_variation
        if group not in data.keys():
            data[group]=[]
        data[group].append(create_single_data(X,y, splits,estimator,title,color,line_type))



# ======= rearrange the plots

data=[arr for arr in data.values()]
data=[item for sublist in data for item in sublist]

# ======= plot

layout = go.Layout(
    title='Learning Curve',
    xaxis=dict(
        title='Number of Training URLs',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
    yaxis=dict(
        title='Accuracy (%)',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
legend=dict(

        font=dict(
            size=10
        )
)
)

fig = go.Figure(data=data, layout=layout)
plot(fig)