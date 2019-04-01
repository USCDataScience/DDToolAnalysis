import plotly.graph_objs as go
from plotly.offline import plot
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import learning_curve
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
# from voting_classifier import VotingClassifier
from sklearn.pipeline import Pipeline

from utils import w2v
from utils.cosine_classifier import CosineClassifier
from utils.utils import *


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
count_vect = TfidfVectorizer(ngram_range=(1,2), stop_words='english',max_df=0.6, min_df=3, max_features=1000)
X_gram = count_vect.fit_transform(X_text)
X=X_gram
print(X.todense().shape)


count_vect_inv_vocab={l:k for k,l in count_vect.vocabulary_.items()}
ALPHA = 0.05

estimators=[MultinomialNB(alpha=ALPHA), MLPClassifier(max_iter=700, learning_rate='adaptive')
            , RandomForestClassifier(n_estimators=100), svm.SVC(gamma='auto'), CosineClassifier(operation='max'), CosineClassifier(operation='avg')]
titles=['Naive Bayes', 'Neural Network', 'Random Forest', 'Support Vector Machines', 'Cosine Similarity - Max', 'Cosine Similarity - Avg']
colors=['rgb(31, 119, 180)','rgb(128, 0, 255)','rgb(44, 160, 44)','rgb(214, 39, 40)','rgb(255, 127, 14)','rgb(122, 77, 31)' ]
legend_group=[1,2,3,4,5,6]
line_type='solid'
algorithm_variation=' (N-gram)'


for estimator,title, color, group in zip(estimators,titles, colors,legend_group):
    print(title+ algorithm_variation)
    title=title+ algorithm_variation
    if group not in data.keys():
        data[group]=[]
    data[group].append(create_single_data(X,y, splits,estimator,title,color,line_type))






# ======= word vector models

X_text_new, model=w2v.create_model(X_text) # create a wv model
X_ww=w2v.create_doc_vec(X_text_new, model) # create document vectors
X=X_ww
print(np.array(X).shape)

estimators=[MLPClassifier(max_iter=700, learning_rate='adaptive')
            , RandomForestClassifier(n_estimators=100), svm.SVC(gamma='auto'), CosineClassifier(operation='max'), CosineClassifier(operation='avg')]
titles=['Neural Network', 'Random Forest', 'Support Vector Machines', 'Cosine Similarity - Max', 'Cosine Similarity - Avg']
colors=['rgb(210,117,255)','rgb(44, 255, 44)','rgb(235, 71, 112)','rgb(255, 167, 45)', 'rgb(162, 102, 42)' ]
legend_group=[2,3,4,5,6]
line_type='solid'
algorithm_variation=' (Word Vec)'

for estimator,title, color, group in zip(estimators,titles, colors,legend_group):
    print(title+ algorithm_variation)
    title=title+ algorithm_variation
    if group not in data.keys():
        data[group]=[]
    data[group].append(create_single_data(X,y, splits,estimator,title,color,line_type))







# ======= urls models (words picked from the urls themselves)
X_urls=[]

word_dict=set([word.strip().lower() for word in open('word_dict.txt','r').readlines() if len(word.strip())>3])

for url in urls:
    url='.'.join(url.split('.')[0:-1])
    if 'www.' in url:
        url=url.split('www.')[1]
    url_broken=[]
    for word in break_natural_boundaries(url):
        url_broken.extend(process_substrings(list(substrings(word, word_dict))))
    X_urls.append(' '.join(url_broken))


cv = CountVectorizer()
X_urls = cv.fit_transform(X_urls)
X=X_urls
print(X.todense().shape)


estimators=[MultinomialNB(alpha=ALPHA), MLPClassifier(max_iter=700, learning_rate='adaptive')
            , RandomForestClassifier(n_estimators=100), svm.SVC(gamma='auto'), CosineClassifier(operation='max'), CosineClassifier(operation='avg')]
titles=['Naive Bayes', 'Neural Network', 'Random Forest', 'Support Vector Machines', 'Cosine Similarity - Max', 'Cosine Similarity - Avg']
colors=['rgb(31, 119, 180)','rgb(128, 0, 255)','rgb(44, 160, 44)','rgb(214, 39, 40)','rgb(255, 127, 14)','rgb(122, 77, 31)'  ]
legend_group=[1,2,3,4,5,6]
line_type='dash'
algorithm_variation=' (Url)'


for estimator,title, color, group in zip(estimators,titles, colors,legend_group):
    print(title+ algorithm_variation)
    title=title+ algorithm_variation
    if group not in data.keys():
        data[group]=[]
    data[group].append(create_single_data(X,y, splits,estimator,title,color,line_type))






# ======= keywords input (only consider the keywords from the SMEs)

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

estimators=[MultinomialNB(alpha=ALPHA), MLPClassifier(max_iter=700, learning_rate='adaptive')
            , RandomForestClassifier(n_estimators=100), svm.SVC(gamma='auto'), CosineClassifier(operation='max'), CosineClassifier(operation='avg')]
titles=['Naive Bayes', 'Neural Network', 'Random Forest', 'Support Vector Machines', 'Cosine Similarity - Max', 'Cosine Similarity - Avg']
colors=['rgb(31, 119, 180)','rgb(128, 0, 255)','rgb(44, 160, 44)','rgb(214, 39, 40)','rgb(255, 127, 14)', 'rgb(122, 77, 31)']
legend_group=[1,2,3,4,5,6]
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
for topic_num, topic in enumerate(topics):
    topic_keywords=topic.strip().split('\n')

    X_topic_keywords = []
    for x in X_text:
        vector=[]
        for i, keyword in enumerate(topic_keywords):
            vector.append(float(len([m.start() for m in re.finditer(keyword.lower(),x.lower())])))
        X_topic_keywords.append(vector)
    X_topic_keywords=np.array(X_topic_keywords)
    X=X_topic_keywords
    print(np.array(X).shape)

    estimators = [MultinomialNB(alpha=ALPHA), MLPClassifier(max_iter=700, learning_rate='adaptive')
        , RandomForestClassifier(n_estimators=100), svm.SVC(gamma='auto'), CosineClassifier(operation='max'),
                  CosineClassifier(operation='avg')]
    titles = ['Naive Bayes', 'Neural Network', 'Random Forest', 'Support Vector Machines', 'Cosine Similarity - Max',
              'Cosine Similarity - Avg']
    colors = ['rgb(31, 119, 180)', 'rgb(128, 0, 255)', 'rgb(44, 160, 44)', 'rgb(214, 39, 40)', 'rgb(255, 127, 14)',
              'rgb(122, 77, 31)']
    # todo: the colors can be commnon to the SME keyword colors for this one, was added later
    legend_group = [1, 2, 3, 4, 5, 6]


    line_type='dot'
    algorithm_variation=' (topical Keywords) - topic'+ str(topic_num)


    for estimator,title, color, group in zip(estimators,titles, colors,legend_group):
        print(title+ algorithm_variation)
        title=title+ algorithm_variation
        if group not in data.keys():
            data[group]=[]
        data[group].append(create_single_data(X,y, splits,estimator,title,color,line_type))







# #    ensamble models ============

# prepare input
X_en=np.concatenate((X_gram.todense(),np.array(X_ww),np.array(X_keywords)),axis=1)
X=X_en
print(X.shape)


# todo: to use voting on different combination of feature set and model, and being able to add it to the VotingClassifier object we use pipeline below.
# todo: Warning: the ranges are hard coded!!
pipe1 = Pipeline([
    ('col_extract', ColumnExtractor( cols=range(0,1000) )),
    ('clf', RandomForestClassifier(n_estimators=100))
    ])

pipe2 = Pipeline([
    ('col_extract', ColumnExtractor( cols=range(1000,1300) )),
    ('clf', RandomForestClassifier(n_estimators=100))
    ])

pipe3 = Pipeline([
    ('col_extract', ColumnExtractor( cols=range(1300,1346))),
    ('clf', RandomForestClassifier(n_estimators=100))
    ])

pipe4 = Pipeline([
    ('col_extract', ColumnExtractor( cols=range(0,1000) )),
    ('clf', CosineClassifier(operation='max'))
    ])

pipe5 = Pipeline([
    ('col_extract', ColumnExtractor( cols=range(0,1000) )),
    ('clf', MultinomialNB(alpha=ALPHA))
    ])




# ======= hard voting:
estimator=VotingClassifier([('rf1',pipe1),('rf2',pipe2),('rf3',pipe3),('rf4',pipe4),('rf5',pipe5)], voting='hard')
group=0
line_type='solid'
algorithm_variation='Ensemble (Random Forest (Ngram, Word Vec, Keywords), Naive Bayes (Ngram), Cosine (Ngram))'
color='rgb(64,64,64)'
title=algorithm_variation

if group not in data.keys():
    data[group] = []
data[group].append(create_single_data(X, y, splits, estimator, title, color, line_type))




# ======= soft voting: (does not word for every algorithm)
estimator=VotingClassifier([('rf1',pipe1),('rf2',pipe2),('rf3',pipe3)],voting='soft')
group=0
line_type='solid'
algorithm_variation='Ensemble (Random Forest (Ngram, Word Vec, Keywords))'
color='rgb(128,128,128)'
title=algorithm_variation

if group not in data.keys():
    data[group] = []
data[group].append(create_single_data(X, y, splits, estimator, title, color, line_type))




# ======= a simple grid search, by creating different combinations of weights for soft voting
# create lines with permutations of weights
cnt=0
for w1 in range(1,4):
    for w2 in range(1,4):
        for w3 in range(1,4):

                weights = [w1,w2,w3]
                if len(set(weights)) == 1: # skip if all weights are equal
                    continue

                print(cnt, weights)
                # convert to softmax float:
                weights = list(map(lambda x: float(x) / (sum(weights)), weights))

                print(cnt,weights)
                cnt+=1
                pipe1 = Pipeline([
                    ('col_extract', ColumnExtractor( cols=range(0,1000) )),
                    ('clf', RandomForestClassifier(n_estimators=100))
                    ])

                pipe2 = Pipeline([
                    ('col_extract', ColumnExtractor( cols=range(1000,1300) )),
                    ('clf', RandomForestClassifier(n_estimators=100))
                    ])

                pipe3 = Pipeline([
                    ('col_extract', ColumnExtractor( cols=range(1300,1346))),
                    ('clf', RandomForestClassifier(n_estimators=100))
                    ])

                estimator = VotingClassifier([('rf1', pipe1), ('rf2', pipe2), ('rf3', pipe3)],
                                             voting='soft',weights=weights)
                group = 0
                line_type = 'solid'
                algorithm_variation = str(', '.join(list(map(lambda x:str(round(x,2)*100)+'%',weights))))
                print(algorithm_variation)
                title = algorithm_variation
                if group not in data.keys():
                    data[group] = []
                data[group].append(create_single_data(X, y, splits, estimator, title, color, line_type))



# ======= rearrange the plots based on data group provided (to have a controlled sequence in the legend)

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