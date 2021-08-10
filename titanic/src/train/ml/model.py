import os

import pandas as pd 
import numpy as np 
from feature_engine.imputation import CategoricalImputer
from feature_engine.imputation import MeanMedianImputer
from feature_engine.encoding import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import metrics


FILE_PATH = os.path.dirname(__file__)
TRAIN_PATH = os.path.dirname(FILE_PATH)
SRC_PATH = os.path.dirname(TRAIN_PATH)
BASE_PATH = os.path.dirname(SRC_PATH)
DATA_PATH = os.path.join(BASE_PATH, 'data')
MODEL_PATH = os.path.join(BASE_PATH, 'model')
RAW_PATH = os.path.join(DATA_PATH, 'raw')
STRUCTURED_PATH = os.path.join(DATA_PATH, 'structured')

print("Importing titanic dataset... ",end="")
abt = pd.read_csv(os.path.join(STRUCTURED_PATH,"train.csv"))
print("Ok!")

print("Spliting train and test samples... ",end="")
target = abt['Survived']
abt_features = abt.drop(columns='Survived')
X_train, X_test, y_train, y_test = train_test_split(abt_features,target,
                                        test_size=0.2,
                                        random_state=0)
print("Ok!")

print("Feature enginnering... ", end="")
categorical = X_train.dtypes[X_train.dtypes=='object'].index.tolist()
numerical = list(set(X_train.columns) - set(categorical))

cat_imputer = CategoricalImputer(variables=categorical)
median_imputer = MeanMedianImputer(imputation_method='mean',variables=numerical)
encoder_group = list(set(categorical).union(set(['Pclass','Parch'])))
encoder = OneHotEncoder( top_categories=2, variables=categorical, drop_last=False)
print("Ok!")

print("Creating a RandomForestClassifier... ", end="")
model_rfc = RandomForestClassifier(random_state=0)
params = {
  'n_estimators': [200,300,400,500],
  'criterion': ['gini','entropy'],
  'min_samples_leaf': [1,5,10,25,50],
}

grid = GridSearchCV(model_rfc,
                    param_grid=params,
                    cv=4,
                    scoring='roc_auc',
                    n_jobs=-1,
                    verbose=2)
print("Ok!")

print("Creating pipeline... ", end="")
pipeline = Pipeline( [ ("Numerical Imputer",median_imputer),
                       ("Categorical Imputer",cat_imputer),
                       ("OneHotEncoder",encoder),
                       ("Model",grid) ] )
print("Ok!")

print("Training the model... ", end="")
pipeline.fit(X_train,y_train)
print("Ok!")

print('Performance:')
y_pred = pipeline.predict(X_test)

accuracy_score = metrics.accuracy_score(y_test,y_pred)
auc_score = metrics.roc_auc_score(y_test,y_pred)
recall_score = metrics.recall_score(y_test,y_pred)

print("Accuracy: ", accuracy_score)
print("AUC: ", auc_score)
print("Recall: ", recall_score)

performance = {
    "accuracy":accuracy_score,
    "auc":auc_score,
    "recall":recall_score
}

print("Saving the model... ", end="")
model_data = pd.Series(
    {
        'model':pipeline,
        'features': X_train.columns.tolist(),
        'performance':performance
    }
)
filename = 'example_model.pkl'
model_data.to_pickle(os.path.join(MODEL_PATH, filename))
print("Ok!")