from sklearn.ensemble import RandomForestClassifier
from rfpimp import *

def find_importances(X,y,columns,splits, estimator=RandomForestClassifier(n_estimators=100)):
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
