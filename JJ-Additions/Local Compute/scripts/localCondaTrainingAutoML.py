from azureml.core import Workspace, Dataset, Datastore
from azureml.core.run import Run
from azureml.interpret import ExplanationClient

import getopt
import joblib
import os
import pickle
import sys

import pandas as pd
import numpy as np

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
    'out-model-file-name=',
    'numeric-feature-names=',
    'categoric-feature-names=',
    'x-train-test-y-train-test='
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
    elif opt == '--out-model-file-name':
        p_out_model_file_name = arg
    elif opt == '--numeric-feature-names':
        # Take in string encoding of list of numeric feature names, turn back into a list
        p_numeric_feature_names = arg
        # The arg value is a string that looks like a list: '["col1","col2"]'
        # So,turn string of features into list of features
        # (Remove square brackets and " marks)
        p_numeric_feature_names = p_numeric_feature_names.replace("[","")
        p_numeric_feature_names = p_numeric_feature_names.replace("]","")
        p_numeric_feature_names = p_numeric_feature_names.replace("\"","")
        print('On character replacement, p_numeric_feature_names was set with value: ', p_numeric_feature_names)
        # (Now split on comma to have a list)
        p_numeric_feature_names = p_numeric_feature_names.split(',')
        print('On split, p_numeric_feature_names was set with value: ', p_numeric_feature_names)
    elif opt == '--categoric-feature-names':
        # Take in string encoding of list of categoric feature names, turn back into a list
        p_categoric_feature_names = arg
        # The arg value is a string that looks like a list: '["col1","col2"]'
        # So,turn string of features into list of features
        # (Remove square brackets and " marks)
        p_categoric_feature_names = p_categoric_feature_names.replace("[","")
        p_categoric_feature_names = p_categoric_feature_names.replace("]","")
        p_categoric_feature_names = p_categoric_feature_names.replace("\"","")
        print('On character replacement, p_categoric_feature_names was set with value: ', p_categoric_feature_names)
        # (Now split on comma to have a list)
        p_categoric_feature_names = p_categoric_feature_names.split(',')
        print('On split, p_categoric_feature_names was set with value: ', p_categoric_feature_names)
    elif opt == '--x-train-test-y-train-test':
        # Take in string encoding of list of categoric feature names, turn back into a list
        xTrainTestYTrainTest = arg
        # The arg value is a string that looks like a list: '["datasetName1","datasetName2"]'
        # So,turn string of features into list of features
        # (Remove square brackets and " marks)
        xTrainTestYTrainTest = xTrainTestYTrainTest.replace("[","")
        xTrainTestYTrainTest = xTrainTestYTrainTest.replace("]","")
        xTrainTestYTrainTest = xTrainTestYTrainTest.replace("\"","")
        print('On character replacement, xTrainTestYTrainTest was set with value: ', xTrainTestYTrainTest)
        # (Now split on comma to have a list)
        xTrainTestYTrainTest = xTrainTestYTrainTest.split(',')
        print('On split, xTrainTestYTrainTest was set with value: ', xTrainTestYTrainTest)
        X_train_registered_name = xTrainTestYTrainTest[0]
        X_test_registered_name = xTrainTestYTrainTest[1]
        y_train_registered_name = xTrainTestYTrainTest[2]
        y_test_registered_name = xTrainTestYTrainTest[3]
    else:
        print("Unrecognized option passed, continuing run, it is: " + opt)

p_feature_names = [*p_numeric_feature_names, *p_categoric_feature_names]
print('p_feature_names was set with value: ', p_feature_names)
print('p_feature_names is now, after decoding numeric and categoric features into lists and concatenating them, a: ', type(p_feature_names))

# BEGIN Get the Workspace object from Azure
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

# Set up for Run that is to be linked with the Experiment this script is run with
run = Run.get_context()
run.log(name='creating outputs and logs directory...', value=0)
run.log(os.makedirs('./outputs', exist_ok=True), value=0)
run.log(os.makedirs('./outputs', exist_ok=True), value=0)

# BEGIN Get Data
# Create datastore, try getting datastore via Workspace object
datastore = Datastore.get_default(ws)
datastore = Datastore.get(ws, p_datastore_name)

# Get DataSet for training from the datastore of the Workspace
# (having registered them already in Notebook code)
# (later) TODO?: remove hardcoding the names of the datasets to get and allow passing them as a --argument 
X_train = Dataset.get_by_name(ws, X_train_registered_name, version = 'latest').to_pandas_dataframe()
X_test  = Dataset.get_by_name(ws, X_test_registered_name, version = 'latest').to_pandas_dataframe()
y_train = Dataset.get_by_name(ws, y_train_registered_name, version = 'latest').to_pandas_dataframe()
y_test = Dataset.get_by_name(ws, y_test_registered_name, version = 'latest').to_pandas_dataframe()
data = {"train": {"X": X_train, "y": y_train},
        "test": {"X": X_test, "y": y_test}}

# Unpickle the SciKit Pipline (that performs Transformation and Model Training)
with open('resources/classifier_pipeline.pickle', 'rb') as file:
    classifier_pipeline = pickle.load(file)
    print(classifier_pipeline)

# You should have a pipeline that looks similar to this one
# Pipeline(steps=[('imputer',
#                                                   SimpleImputer(strategy='median')),
#                                                  ('scaler', StandardScaler())]),
#                                  ['Age', 'Fare']),
#                                 ('cat',
#                                  Pipeline(steps=[('imputer',
#                                                   SimpleImputer(fill_value='missing',
#                                                                 strategy='constant')),
#                                                  ('onehot',
#                                                   OneHotEncoder(handle_unknown='ignore',
#                                                                 sparse=False))]),
#                                  ['Pclass', 'Sex', 'SibSp', 'Parch', 'Cabin',
#                                   'Embarked'])])

# Run training Pipeline
model_DT = classifier_pipeline.fit(X_train,y_train)

# Training the model is as simple as this
# We use the predict() on the model to predict the output
prediction = model_DT.predict(X_test)

# For classification, using accuracy and F1 score
print(accuracy_score(y_test,prediction))
print(f1_score(y_test,prediction))

# for regression...
#
# we use R2 score and MAE(mean absolute error)
# all other steps will be same as classification as shown above
# 
# from sklearn.metrics import mean_absolute_error
# from sklearn.metrics import r2_score
# print(mean_absolute_error(y_test,prediction))
# print(r2_score(y_test,prediction))


# Save the output model to a file...
# ... This is required for the model to get automatically uploaded by the Notebook using this script
with open(p_out_model_file_name, "wb") as file:
    joblib.dump(value=model_DT, filename=os.path.join('./outputs/', p_out_model_file_name))
file.close()




# BEGIN Add Explanations (In terms of both engineered and raw features)
from interpret.ext.blackbox import TabularExplainer
# BEGIN Add Raw Explanations

# Fit the model
classifier_pipeline.fit(X_train, y_train.values.ravel())

# Send ColumnTransformer preprocessor inside classifier_pipeline in TabularExplainer construction here
# classifier_pipeline.steps[-1][1] returns the trained classification model
# Pass transformation as an input to create the Explanation object in terms of Raw Features that are not One Hot Encoded
# Explain in terms of engineered features
# NOTE: classifier_pipeline.steps[-1][1] contains the Model
# NOTE: "features" and "classes" fields are optional for TabularExplainers
raw_explainer = TabularExplainer(classifier_pipeline.steps[-1][1],
                                     initialization_examples=X_train,
                                     features=p_feature_names,
                                     transformations=classifier_pipeline['preprocessor'])
# Explain results with this Explainer and upload the Explanation...

# Get Global Explanations of raw features, global as in 'of total data'...
# You can use the training data or the test data here, but test data would allow you to use Explanation Exploration
# print("X_train, line value before raw_explainer.explain_global: \n" + str(X_train))
global_explanation = raw_explainer.explain_global(X_train, y_train)
# If you used the PFIExplainer in the previous step, use the next line of code instead
# global_explanation = raw_explainer.explain_global(x_train, true_labels=y_train)
# Sorted feature importance values and feature names
sorted_global_importance_values = global_explanation.get_ranked_global_values()
sorted_global_importance_names = global_explanation.get_ranked_global_names()
globalFeatureExplanations = dict(zip(sorted_global_importance_names, sorted_global_importance_values))
print('globalFeatureExplanations: ', globalFeatureExplanations)
# Alternatively, you can print out a dictionary that holds the top K feature names and values
print('global_explanation.get_feature_importance_dict(): ', global_explanation.get_feature_importance_dict())

# print("y_train value the line before client.upload_model_explanation(): \n" + str(y_train))
# print("y_train.values.ravel() value passed as true_ys to client.upload_model_explanation(): \n" + str(y_train.values.ravel()))
client = ExplanationClient.from_run(run)
#           (perhaps can only have Explanation of one of Raw or Engineered features at once?)
# Upload global model explanation data...
# The explanation can then be downloaded on any compute
# Multiple explanations can be uploaded
# TODO? pass comments arguments as parameters
client.upload_model_explanation(global_explanation, true_ys=y_train.values.ravel(), comment='global explanation: train dataset features, raw')
# Or you can only upload the explanation object with the top k feature info with this...
# client.upload_model_explanation(global_explanation, top_k=2, comment='global explanation: Only top 2 features')
# END Add Raw Explanations




# BEGIN Add Engineered Feature Explanations

# Split training features into numeric and categoric dataframes
numeric_X_train = pd.DataFrame(X_train[p_numeric_feature_names], dtype=np.str, columns=p_numeric_feature_names)
categoric_X_train = pd.DataFrame(X_train[p_categoric_feature_names], dtype=np.str, columns=p_categoric_feature_names)
# Fit and Run the numeric ColumnTransformers on the split dataframe to perform feature engineering
classifier_pipeline['preprocessor'].transformers[0][1].fit(numeric_X_train)
numeric_X_train_preprocessed = classifier_pipeline['preprocessor'].transformers[0][1].transform(numeric_X_train)
numeric_X_train_preprocessed = pd.DataFrame(numeric_X_train_preprocessed, dtype=np.float, columns=p_numeric_feature_names)
# Fit and Run the categoric ColumnTransformers on the split dataframe to perform feature engineering
classifier_pipeline['preprocessor'].transformers[1][1].steps[0][1].fit(categoric_X_train)
categoric_X_train_preprocessed = classifier_pipeline['preprocessor'].transformers[1][1].steps[0][1].transform(categoric_X_train)
# Get and fit OneHotEncoder
one_hot_encoder = classifier_pipeline['preprocessor'].transformers[1][1][1]
one_hot_encoder.fit(categoric_X_train_preprocessed)
# Get new One Hot Encoded column names
df_encoded_categorical_column_names = one_hot_encoder.get_feature_names(p_categoric_feature_names)
print("df_encoded_categorical_column_names", df_encoded_categorical_column_names)
# Transform categoric, null-imputed features with fitted OneHotEncoder
categoric_X_train_preprocessed = one_hot_encoder.transform(categoric_X_train_preprocessed)
# Turn preprocessed categoric features into a DataFrame
categoric_X_train_preprocessed = pd.DataFrame(categoric_X_train_preprocessed, dtype=np.float64, columns=df_encoded_categorical_column_names)

# Combine the numeric DF with the categorical DF to submit to the classifier_pipeline
X_train_preprocessed_list = [numeric_X_train_preprocessed, categoric_X_train_preprocessed]
X_train_preprocessed = pd.concat(X_train_preprocessed_list, axis=1)


# Save engineered features' names to create TabularExplainer with them
engineeredFeatures=[*p_numeric_feature_names, *df_encoded_categorical_column_names]

# Fit the model
classifier_pipeline.steps[-1][1].fit(X_train_preprocessed, y_train)
# Explain in terms of engineered features
# NOTE: classifier_pipeline.steps[-1][1] contains the Model
# NOTE: "features" and "classes" fields are optional for TabularExplainers
from interpret.ext.blackbox import TabularExplainer
engineered_explainer = TabularExplainer(classifier_pipeline.steps[-1][1],
                                     initialization_examples=X_train_preprocessed,
                                     features=engineeredFeatures)
# Explain results with this Explainer and upload the Explanation...

# Get Global Explanations of raw features, global as in 'of total data'...
# You can use the training data or the test data here, but test data would allow you to use Explanation Exploration
# print("X_train, line value before engineered_explainer.explain_global: \n" + str(X_train))
global_explanation = engineered_explainer.explain_global(X_train_preprocessed, y_train)
# If you used the PFIExplainer in the previous step, use the next line of code instead
# global_explanation = engineered_explainer.explain_global(X_train, true_labels=y_train)
# Sorted feature importance values and feature names
sorted_global_importance_values = global_explanation.get_ranked_global_values()
sorted_global_importance_names = global_explanation.get_ranked_global_names()
globalFeatureExplanations = dict(zip(sorted_global_importance_names, sorted_global_importance_values))
print('globalFeatureExplanations: ', globalFeatureExplanations)
# Alternatively, you can print out a dictionary that holds the top K feature names and values
print('global_explanation.get_feature_importance_dict(): ', global_explanation.get_feature_importance_dict())

# Upload the explanation in terms of engineered features
from azureml.interpret import ExplanationClient
client = ExplanationClient.from_run(run)
client.upload_model_explanation(global_explanation, true_ys=y_train.values.ravel(), comment='global explanation: train dataset features, engineered')











# BEGIN Alternative code that was used to get Explanations of Model, it is more simple, not using a sklearn Pipeline to transform features...
# from interpret.ext.blackbox import TabularExplainer

# # "features" and "classes" fields are optional
# explainer = TabularExplainer(model_DT, 
#                              X_train, 
#                              features=p_feature_names)

# # BEGIN Get Global Explanations, global as in 'of total data'...

# # You can use the training data or the test data here, but test data would allow you to use Explanation Exploration
# # print("X_test, line value before explainer.explain_global: \n" + str(X_test))
# global_explanation = explainer.explain_global(X_test, y_test)
# # If you used the PFIExplainer in the previous step, use the next line of code instead
# # global_explanation = explainer.explain_global(x_train, true_labels=y_train)
# # Sorted feature importance values and feature names
# sorted_global_importance_values = global_explanation.get_ranked_global_values()
# sorted_global_importance_names = global_explanation.get_ranked_global_names()
# globalFeatureExplanations = dict(zip(sorted_global_importance_names, sorted_global_importance_values))
# print('globalFeatureExplanations: ', globalFeatureExplanations)
# # Alternatively, you can print out a dictionary that holds the top K feature names and values
# print('global_explanation.get_feature_importance_dict(): ', global_explanation.get_feature_importance_dict())

# # BEGIN Uploading global model explanation data...
# # The explanation can then be downloaded on any compute
# # Multiple explanations can be uploaded
# print("y_test value the line before client.upload_model_explanation(): \n" + str(y_test))
# print("y_test.values.ravel() value passed as true_ys to client.upload_model_explanation(): \n" + str(y_test.values.ravel()))
# client.upload_model_explanation(global_explanation, true_ys=y_test.values.ravel(), comment='global explanation: all features')
# # Or you can only upload the explanation object with the top k feature info with this...
# # client.upload_model_explanation(global_explanation, top_k=2, comment='global explanation: Only top 2 features')
# # END Uploading global model explanation data...

# # BEGIN Get local explanations of individual predictions
# # Get explanation for the first few data points in the test set
# # local_explanation = explainer.explain_local(X_test[0:5])
# # Sorted feature importance values and feature names
# # sorted_local_importance_names = local_explanation.get_ranked_local_names()
# # print('sorted_local_importance_names: ', sorted_local_importance_names)
# # print('len(sorted_local_importance_names): ', len(sorted_local_importance_names))
# # sorted_local_importance_values = local_explanation.get_ranked_local_values()
# # print('sorted_local_importance_values: ', sorted_local_importance_values)
# # print('len(sorted_local_importance_values): ', len(sorted_local_importance_values)) 
# # 
# # THIS DOES NOT WORK LIKE IT DOES WITH THE GLOBAL_EXPLANATION, HOWEVER!
# # client.upload_model_explanation(sorted_local_importance_values, comment='local explanation for data points 0-5: all features')
# #
# # END Get local explanations of individual predictions