import os

import pandas as pd
from pandas.core.frame import DataFrame 

FILE_PATH = os.path.dirname(__file__)
SRC_PATH = os.path.dirname(FILE_PATH)
BASE_PATH = os.path.dirname(SRC_PATH)
DATA_PATH = os.path.join(BASE_PATH, 'data')
MODEL_PATH = os.path.join(BASE_PATH, 'model')
RAW_PATH = os.path.join(DATA_PATH, 'raw')
STRUCTURED_PATH = os.path.join(DATA_PATH, 'structured')
RESULTS_PATH = os.path.join(DATA_PATH, 'results')

print('Importing model... ',end='')
model = pd.read_pickle(os.path.join(MODEL_PATH,"example_model.pkl"))
print('Ok!')

print('Importing test dataset... ',end='')
X_test = pd.read_csv(os.path.join(STRUCTURED_PATH,"test.csv"))
print('Ok!')

print("Predicting... ",end="")
y_pred = model['model'].predict(X_test[model['features']])
print('Ok!')

print("Exporting results...",end="")
pd.DataFrame(y_pred).to_csv(os.path.join(RESULTS_PATH,'results.csv'))
print('Ok!')