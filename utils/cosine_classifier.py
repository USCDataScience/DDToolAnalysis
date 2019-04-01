from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import operator
from sklearn.base import BaseEstimator

class CosineClassifier(BaseEstimator):
    def __init__(self, operation='avg'):
        self._estimator_type='classifier'
        self.operation=operation

    def fit(self, X, y):
        self.X=X
        self.y=y

    def predict(self, X):
        result=[]
        sim=cosine_similarity(X,self.X,dense_output=True)


        for i in range(X.shape[0]):
            average={}
            for j in range(self.X.shape[0]):
                similarity=sim[i][j]
                label=self.y[j]
                if label not in average.keys():
                    average[label]=[]
                average[label].append(similarity)
            if self.operation=='max':
                fuse={label:np.max(lst) for label,lst in average.items()}
            else:
                fuse={label:np.mean(lst) for label,lst in average.items()}
            result.append(max(fuse.items(), key=operator.itemgetter(1))[0])
        return np.array(result)


# estimator=CosineClassifier(operation='max')
#
# X=[[2,3,4],[3,4,5],[5,6,7],[7,8,9]]
# y=['a','b','a','b']
#
# estimator.fit(X,y)
# print(estimator.predict([[3,4,5],[2,3,4]]))
