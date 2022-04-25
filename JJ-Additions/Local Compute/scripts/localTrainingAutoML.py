from azureml.core import Workspace, Dataset, Datastore
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core.run import Run

import numpy as np
import pandas as pd

import os
import sys
import getopt
import joblib

# For preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
# For null value imputation
from sklearn.impute import SimpleImputer
# For splitting of data into train and test set
from sklearn.model_selection import train_test_split
#Importing the Decision Tree from scikit-learn library
from sklearn.tree import DecisionTreeClassifier
# Metrics for Evaluation of model Accuracy and F1-score, for classification
from sklearn.metrics  import f1_score,accuracy_score

# Get Arguments
print("sus/argv: ", sys.argv)
argv = sys.argv[1:]
print("argv: ", argv)
# try:
long_options = [
    'tenant-id=', 
    'ws-name=', 
    'subscription-id=', 
    'resource-group=',
    'datastore-name=',
    'dataset-name=',
    'out-model-file-name='
]
opts, args = getopt.getopt(argv, None, long_options)
# except:
    # print("Error running getopt.getopt to get arguments passed to script")

print('here are opts ', opts)
for opt, arg in opts:
    print('here is opt,arg: ', opt, arg)
    if opt == '--tenant-id':
        p_tenant_id = arg
    elif opt == '--ws-name':
        p_ws_name = arg
    elif opt == '--subscription-id':
        p_subscription_id = arg
    elif opt == '--resource-group':
        p_resource_group = arg
    elif opt == '--datastore-name':
        p_datastore_name = arg
    elif opt == '--dataset-name':
        p_dataset_name = arg
    elif opt == '--out-model-file-name':
        p_out_model_file_name = arg
    else:
        print("Unrecognized option passed, continuing run, it is: " + opt)

# Get the Workspace object from Azure
# (you can find tenant id under azure active directory->properties)
# ia = InteractiveLoginAuthentication(tenant_id=[p_tenant_id])
ws_name = p_ws_name
subscription_id = p_subscription_id
resource_group = p_resource_group
ws = Workspace.get(name=ws_name,
                   subscription_id=subscription_id,
                   resource_group=resource_group)
#    auth=ia)
print('ws.name:', ws.name, 'ws.resource_group:', ws.resource_group, 'ws.location:', ws.location, 'ws.subscription_id:', ws.subscription_id, sep='\n')

run = Run.get_context()
run.log(name='creating outputs and logs directory...', value=0)
run.log(os.makedirs('./outputs', exist_ok=True), value=0)
run.log(os.makedirs('./outputs', exist_ok=True), value=0)

# Get Datastore via Workspace object
datastore = Datastore.get_default(ws)
datastore = Datastore.get(ws, p_datastore_name)

# Get Dataset
dataset = Dataset.get_by_name(ws, p_dataset_name, version = 'latest')
# Turn Dataset into Pandas Dataframe by transforming the initial data
df = dataset.to_pandas_dataframe()

# Prepare pandas dataframes of train/test data to pass to train_test_split
df_column_names = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']
target_column_name = 'Survived'
feature_column_names = df_column_names[1:]

# BEGIN Reform df dataframe so that it contains all numbers...
# For int column Age, Impute NaN numeric values and Remove outliers
print('Before Removing outliers or Imputing null values, df[Age]: ', df['Age'])
ageMedian = np.nanmedian(df['Age'])
print('ageMedian: ', ageMedian)
df['Age'] = np.where(np.isnan(df['Age']), ageMedian, df['Age'])
print('Before Removing outliers and after Imputing null values, df[Age]: ', df['Age'])

# Calculate 3STD and Mean for Age
ageThreeSD = np.std(df['Age']) * 3
ageMean = np.mean(df['Age'])
ageOutlierThreshold = round(ageThreeSD + ageMean)
print('Age Outlier Threshold: ', ageOutlierThreshold)

# Remove Outliers by replacing all values above Threshold (3STD + Mean) with Threshold Value
df['Age'] = df['Age'].mask(df['Age'] > ageOutlierThreshold, ageOutlierThreshold)
print('After Removing outliers and Imputing null values, df[Age]: ', df['Age'])

# Copy df, keeping only Age column, set type of this df copy to float
df_age_column = pd.DataFrame(data=df['Age'], columns=['Age'])

# Copy df, keeping only float numeric columns, set type of this df copy to float
df_float_column_names = ['Fare']
print('df_float_column_names: ', df_float_column_names)
df_float_columns = pd.DataFrame(data=df[df_float_column_names], dtype=np.float, columns=df_float_column_names)

# Copy df, keeping only categorical columns, and one-hot encode them
df_categorical_column_names_tmp = ['Survived', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Cabin', 'Embarked']
df_categorical_column_names = df_categorical_column_names_tmp
print('df_categorical_column_names: ', df_categorical_column_names)
df_categorical_columns = pd.DataFrame(data=df[df_categorical_column_names], dtype=np.str, columns=df_categorical_column_names)
encoder = OneHotEncoder(drop='first', handle_unknown='error', sparse=False).fit(df_categorical_columns)
# TODO resolve NaN in Age column...
print('encoder.categories: ', encoder.categories)
# Should get something similiar in form to this printed...
#   [array(['female', 'male'], dtype=object), array(['from Europe', 'from US'], dtype=object), array(['uses Firefox', 'uses Safari'], dtype=object)]
df_encoded_categorical_columns = pd.DataFrame(encoder.transform(df_categorical_columns))
# Should get something like this returned...
#   array([[1., 0., 0., 1., 0., 1.],
#     [0., 1., 1., 0., 0., 1.]])
#   By default, the values each feature can take is inferred automatically from the dataset and can be found in the categories_ attribute:
# Combine the numeric DF with the categorical DF

# dfTyped = df_numeric_columns.combine(df_encoded_categorical_columns)
# dfTyped = df_age_column.combine(df_float_columns, df_encoded_categorical_columns)
dfs = [df_age_column, df_float_columns, df_encoded_categorical_columns]
# TODO resolve NaN in Age column...
print('Before concatenation, df[Age]: ', df['Age'])
print('Before concatenation, df_age_column: ', df_age_column)
dfTyped = pd.concat(dfs)
print('dfTyped: ', dfTyped)

# TODO Potentially resolve weird columns?...

        # - ! With sklearn.preprocessing, preprocess your Dataframes before training model in the Python Script
        #     - [Guide at SciKit Learn site](https://scikit-learn.org/stable/modules/preprocessing.html)
        #     - Use OneHotEncoder
        #     - Use StandardScaler or  MinMaxScaler while you're at it
        #     - Don't worry about any other preprocessing to just get the training working
        #     - Strategy:
        #         - d Split dataframe into Numeric/Non-Categorial and Non-Numeric/Categorial columns
        #             - ! Use StandardScaler or MinMaxScaler on Numeric/Non-Categorical columns split
        #             - d Use OneHotEncoder on Non-Numeric/Categorical columns split


# TODO: ! Use StandardScaler or MinMaxScaler on Numeric/Non-Categorical columns split
scaler = StandardScaler().fit(dfTyped)
print('scaler.mean_: ', scaler.mean_)
print('scaler.scale: ', scaler.scale_)

dfScaled = scaler.transform(dfTyped)
print('dfScaled: ', dfScaled)
# Scaled data should have zero mean and unit variance, check with these prints:
print('dfScaled.mean(axis=0): ', dfScaled.mean(axis=0))
print('dfScaled.std(axis=0)', dfScaled.std(axis=0))

# TODO Initial Data Frame is now preprocessed in dfPreprocessed
dfPreprocessed = dfScaled
print('dfPreprocessed: ', dfPreprocessed)

# END Reform df dataframe so that it contains all numbers...



df_x = dfPreprocessed.drop_columns([target_column_name]).to_pandas_dataframe()
print("here are feature_columns: ", feature_column_names)
df_y = dfPreprocessed.drop_columns(feature_column_names).to_pandas_dataframe()
print("See df_x", df_x)
print("See df_y", df_y)
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

# Save the output model to a file...
# ... This is required for the model to get automatically uploaded by the Notebook using this script
with open(p_out_model_file_name, "wb") as file:
    joblib.dump(value=DT, filename=os.path.join('./outputs/', p_out_model_file_name))
file.close()

# for regression...
#
# we use R2 score and MAE(mean absolute error)
# all other steps will be same as classification as shown above
# 
# from sklearn.metrics import mean_absolute_error
# from sklearn.metrics import r2_score
# print(mean_absolute_error(y_test,prediction))
# print(r2_score(y_test,prediction))

exit(0)