import os
import argparse

import pandas as pd

FILE_PATH = os.path.dirname(__file__)
TRAIN_PATH = os.path.dirname(FILE_PATH)
SRC_PATH = os.path.dirname(TRAIN_PATH)
BASE_PATH = os.path.dirname(SRC_PATH)
DATA_PATH = os.path.join(BASE_PATH, 'data')
RAW_PATH = os.path.join(DATA_PATH, 'raw')
STRUCTURED_PATH = os.path.join(DATA_PATH, 'structured')

def col_cabin(x):
    try:
        return x[0]
    except:
        return 'Z'

def col_name(x):
    if "mr" in x.lower(): 
        return 'Mr'
    if "mrs" in x.lower():
        return 'Mrs'
    if "miss" in x.lower():
        return 'Miss'
    else:
        return None

def etl_df(dataframe):
    dataframe['Cabin'] = dataframe['Cabin'].apply(lambda x: col_cabin(x))
    dataframe['Name'] = dataframe['Name'].apply(lambda x: col_name(x))
    dataframe.drop(columns=['PassengerId','Ticket'],inplace=True)
    return dataframe

parser = argparse.ArgumentParser(description="ETL for model")
parser.add_argument("--dataset", help='Dataset Titanic', default='train.csv', type=str)
args = parser.parse_args()

df = pd.read_csv(os.path.join(RAW_PATH,args.dataset))
df_transformed = etl_df(df)
df_transformed.to_csv(os.path.join(STRUCTURED_PATH,args.dataset))