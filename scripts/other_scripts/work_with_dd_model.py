from sklearn.externals import joblib
import json

def explore_model(filepath):

    model = joblib.load(filepath)

    for detail in model['url_details']:
        print(detail['url'], detail['title'], detail['label'])
    print(model['labeled'])
    print(model.keys())
    print(model['url_details'][0].keys())
    # for k,v in model.items():
    #     print(k,v)


# explore_model('ModelAfter260URLs.dms')


# read the model from dd tool and save it as a json file

data_dir='../../data/'
js = {}
model = joblib.load(data_dir+'ModelAfter260URLs.dms')
js['url_text'] = list(model['url_text'])
js['labeled'] = list(model['labeled'])
mod = open(data_dir+'ruth_model', 'w')
mod.write(json.dumps(js))
