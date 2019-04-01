import itertools
import re
from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np
import dill


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

def savemodel(model,outfile):
    with open(outfile, 'wb') as output:
        dill.dump(model, output)
    return ''

def loadmodel(infile):
    model=''
    with open(infile, 'rb') as inp:
        model = dill.load(inp)
    return model

class ColumnExtractor(TransformerMixin, BaseEstimator):
    def __init__(self, cols):
        self.cols = cols

    def transform(self, X):
        col_list = []
        for c in self.cols:
            col_list.append(X[:, c:c+1])
        return np.concatenate(col_list, axis=1)

    def fit(self, X, y=None):
        return self

def break_natural_boundaries(line):
    line = line.split()
    tokens = []
    for string in line:
        stringbreak = []
        if len(string.split(' ')) > 1:
            stringbreak = string.split(' ')
        else:
            # spl = '[\.|\%|\$|\^|\*|\@|\!|\_|\-|\(|\)|\:|\;|\'|\"|\{|\}|\[|\]|]'
            alpha = '[A-z]'
            num = '\d'
            spl = '[^A-z\d]'

            matchindex = set()
            matchindex.update(set(m.start() for m in re.finditer(num + alpha, string)))
            matchindex.update(set(m.start() for m in re.finditer(alpha + num, string)))

            matchindex.update(set(m.start() for m in re.finditer(spl + alpha, string)))
            matchindex.update(set(m.start() for m in re.finditer(alpha + spl, string)))


            matchindex.add(len(string) - 1)
            matchindex = sorted(matchindex)
            start = 0

            for i in matchindex:
                end = i
                if string[start:end+1].isalpha():
                    stringbreak.append(string[start:end + 1])
                start = i + 1
            tokens.extend(stringbreak)

    return tokens

def findsubsets(S,m):
    return set(itertools.combinations(S, m))



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


def find_all_words(s, dictionary):
    all_words_here = {}
    for word in dictionary:
        ix=s.find(word)
        if ix !=-1:
            all_words_here[word]=(ix, ix+len(word)-1)
    return all_words_here


def substrings(s, dictionary):
    # print('working on', s)
    all_words_here=find_all_words(s,dictionary)
    # print('found words',all_words_here.keys())
    summ=[]
    for word in all_words_here:
        multip=[]
        left=[]
        right=[]
        ix=s.find(word)
        jx=ix+len(word)
        part_left=s[0:ix]
        part_right=s[jx:len(s)]
        # print(part_left,'||',word,'||',part_right)
        if len(part_left)!=0:
            left=substrings(part_left,dictionary)
        if len(part_right)!=0:
            right=substrings(part_right,dictionary)
        # print('lr',left,right)

        if len(left) == 0 and len(right) == 0:
            multip.append([word])
        elif len(left)!=0 and len(right)!=0:
            for l in left:
                for r in right:
                    multip.append(l+[word]+r)
        elif len(right)!=0:
            for r in right:
                multip.append([word] + r)
        elif len(left)!=0:
            for l in left:
                multip.append(l+ [word])
        # print('multi',multip)

        summ.extend(multip)
    # print('sum',summ)
    return summ


def process_substrings(list_of_list):
    if len(list_of_list)==0:
        return []
    minimum_size=min([len(lst) for lst in list_of_list])
    list_of_list=list(filter(lambda x:True if len(x)<=minimum_size else False,list_of_list))
    lens=[len(''.join(lst)) for lst in list_of_list]
    max_ix=np.argmax(lens)
    return list_of_list[max_ix]
