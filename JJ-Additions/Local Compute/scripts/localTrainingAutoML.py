from azureml.core import Workspace, Dataset, Datastore
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core.run import Run

import numpy as np
import pandas as pd

import os
import sys
import getopt

# For splitting of data into train and test set
from sklearn.model_selection import train_test_split
#Importing the Decision Tree from scikit-learn library
from sklearn.tree import DecisionTreeClassifier
# Metrics for Evaluation of model Accuracy and F1-score, for classification
from sklearn.metrics  import f1_score,accuracy_score

os.makedirs('./outputs', exist_ok=True)
os.makedirs('./logs', exist_ok=True)

# Get Arguments
argv = sys.argv[1:]
try:
    long_options = [
        "tenant_id=", 
        "ws_name=", 
        "subscription_id=", 
        "resource_group=",
        "datastore_name=",
        "dataset_name="
    ]
    opts, args = getopt.getopt(argv, None, long_options)
except:
    print("Error")

for opt, arg in opts:
    if opt == '--tenant_id':
        p_tenant_id = arg
    elif opt == ['--ws_name']:
        p_ws_name = arg
    elif opt == ['--subscription_id']:
        p_subscription_id = arg
    elif opt == ['--resource_group']:
        p_resource_group = arg
    elif opt == ['--datastore_name']:
        p_datastore_name = arg
    elif opt == ['--dataset_name']:
        p_dataset_name = arg
    else:
        print("Unrecognized option passed, continuing run, it is: " + opt)

# Get the Workspace object from Azure
# (you can find tenant id under azure active directory->properties)
ia = InteractiveLoginAuthentication(tenant_id=[p_tenant_id])
ws_name = p_ws_name
subscription_id = p_subscription_id
resource_group = p_resource_group
ws = Workspace.get(name=ws_name,
                   subscription_id=subscription_id,
                   resource_group=resource_group,
                   auth=ia)
print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep='\n')

run = Run.get_context()

# Get Datastore via Workspace object
datastore = Datastore.get_default(ws)
datastore = Datastore.get(ws, p_datastore_name)

# Get Dataset
dataset = Dataset.get_by_name(ws, p_dataset_name, version = 'latest')
# Turn Dataset into Pandas Dataframe by transforming the initial data
dfRaw = dataset.to_pandas_dataframe()


# Prepare data for training
# first we split our data into input and output
# y is the output and is stored in "Class" column of dataframe
# X contains the other columns and are features or input
#
# y = train.Class
# train.drop(['Class'], axis=1, inplace=True)
# X = train
# 
# (https://www.educative.io/blog/scikit-learn-cheat-sheet-classification-regression-methods)

# Prepare pandas dataframes of train/test data to pass to train_test_split
dfRaw_columns = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']
target_column_name = 'Survived'
feature_columns = dfRaw_columns[1:]
df = dataset
df_x = dataset.drop_columns([target_column_name])
print("here are feature_columns: ", feature_columns)
df_y = dataset.drop_columns(feature_columns)
print("See df_x", df_x.to_pandas_dataframe())
print("See df_y", df_y.to_pandas_dataframe())
# Register Pandas Dataframe
Dataset.Tabular.register_pandas_dataframe(df_x, datastore, "Titanic Feature Column Data for train_test_split usage")
Dataset.Tabular.register_pandas_dataframe(df_y, datastore, "Titanic Target Column Data for train_test_split usage")

# What you need to pass to train_test_split...
# ... I need X and Y dataframe, X just with target missing, Y just with target column present
X_train, X_test, y_train, y_test = train_test_split(df_x, df_y,
                                                    test_size=0.2,
                                                    random_state=0)
data = {"train": {"X": X_train, "y": y_train},
        "test": {"X": X_test, "y": y_test}}

# Training the model is as simple as this
# Use the function imported above and apply fit() on it
DT = DecisionTreeClassifier()
DT.fit(X_train,y_train)

# We use the predict() on the model to predict the output
prediction = DT.predict(X_test)

# For classification, using accuracy and F1 score
print(accuracy_score(y_test,prediction))
print(f1_score(y_test,prediction))

# TODO Save the output model to a file...
# ... This is required for the model to get automatically uploaded by the Notebook using this script
# with open(model_file_name, "wb") as file:
#     joblib.dump(value=reg, filename=os.path.join('./outputs/',
#                                                     model_file_name))

 
# for regression...
#
# we use R2 score and MAE(mean absolute error)
# all other steps will be same as classification as shown above
# 
# from sklearn.metrics import mean_absolute_error
# from sklearn.metrics import r2_score
# print(mean_absolute_error(y_test,prediction))
# print(r2_score(y_test,prediction))