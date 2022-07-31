from azureml.core import Workspace, Dataset, Datastore, Experiment
from azureml.core.run import Run

import getopt
import joblib
import os
import sys
import logging

import pandas as pd
import numpy as np

#Importing the Decision Tree from scikit-learn library
# Metrics for Evaluation of model Accuracy and F1-score, for classification
from sklearn.metrics  import f1_score,accuracy_score
# Metrics for Evaluation of model: RSME and R2 for regression
# from sklearn.metrics  import mean_squared_error,r2_score

# Import the rest
import subprocess
from dotenv import load_dotenv
from azureml.train.automl import AutoMLConfig
from azureml.core.authentication import ServicePrincipalAuthentication

import pkg_resources
print("(azureml.core.authentication is from this) azureml-core==", pkg_resources.get_distribution('azureml-core').version)
print("azureml-train-automl==", pkg_resources.get_distribution('azureml-train-automl').version)
print("azureml-interpret==", pkg_resources.get_distribution('azureml-interpret').version)
print("python-dotenv==", pkg_resources.get_distribution('python-dotenv').version)
print("interpret-community==", pkg_resources.get_distribution('interpret-community').version)
print("interpret-core==", pkg_resources.get_distribution('interpret-core').version)

# Print info about the local environment
# import subprocess
print("output of pwd: ", subprocess.check_output("pwd", shell=True).decode('ascii'))
print("output of ls -altr .: ", subprocess.check_output("ls -altr", shell=True).decode('ascii'))
print("output of ls -altr ./resources: ", subprocess.check_output("ls -altr ./resources", shell=True).decode('ascii'))

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
    'x-train-test-y-train-test-combined-train-test=',
    'num_classes=',
    'weight_column_name=',
    'automlconfig_experiment_name='
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
    elif opt == '--x-train-test-y-train-test-combined-train-test':
        # Take in string encoding of list of categoric feature names, turn back into a list
        xTrainTestYTrainTestCombinedTrainTest = arg
        # The arg value is a string that looks like a list: '["datasetName1","datasetName2"]'
        # So,turn string of features into list of features
        # (Remove square brackets and " marks)
        xTrainTestYTrainTestCombinedTrainTest = xTrainTestYTrainTestCombinedTrainTest.replace("[","")
        xTrainTestYTrainTestCombinedTrainTest = xTrainTestYTrainTestCombinedTrainTest.replace("]","")
        xTrainTestYTrainTestCombinedTrainTest = xTrainTestYTrainTestCombinedTrainTest.replace("\"","")
        print('On character replacement, xTrainTestYTrainTestCombinedTrainTest was set with value: ', xTrainTestYTrainTestCombinedTrainTest)
        # (Now split on comma to have a list)
        xTrainTestYTrainTestCombinedTrainTest = xTrainTestYTrainTestCombinedTrainTest.split(',')
        print('On split, xTrainTestYTrainTestCombinedTrainTest was set with value: ', xTrainTestYTrainTestCombinedTrainTest)
        X_train_registered_name = xTrainTestYTrainTestCombinedTrainTest[0]
        X_test_registered_name = xTrainTestYTrainTestCombinedTrainTest[1]
        y_train_registered_name = xTrainTestYTrainTestCombinedTrainTest[2]
        y_test_registered_name = xTrainTestYTrainTestCombinedTrainTest[3]
        train_data_registered_name = xTrainTestYTrainTestCombinedTrainTest[4]
        test_data_registered_name = xTrainTestYTrainTestCombinedTrainTest[5]
    elif opt == '--num_classes':
        p_num_classes = arg
    elif opt == '--weight_column_name':
        p_weight_column_name = arg
    elif opt == '--automlconfig_experiment_name':
        p_automlconfig_experiment_name = arg
    else:
        print("Unrecognized option passed, continuing run, it is this: " + opt)

p_feature_names = [*p_numeric_feature_names, *p_categoric_feature_names]
print('p_feature_names was set with value: ', p_feature_names)
print('p_feature_names is now, after decoding numeric and categoric features into lists and concatenating them, a: ', type(p_feature_names))

# BEGIN Get the Workspace object from Azure

# This will not work inside the headless docker container this runs in...
# from azureml.core.authentication import InteractiveLoginAuthentication
# ia = InteractiveLoginAuthentication(tenant_id=p_tenant_id)

# Get WS object with Service Principal authentication
ws_name = p_ws_name
subscription_id = p_subscription_id
resource_group = p_resource_group
# Get credentials for authentication 
# from dotenv import load_dotenv
load_dotenv('./resources/custom_env_vars_for_script_inside_docker_container')
# Authenticate with the Service Principal in order to get the Workspace object
#   (this is a workaround for interactive authentication ocurring within headless Docker container)
#   (you can find tenant id under azure active directory->properties)
#   (you can find clientId in the Service Principal's page in the Azure portal)
#   (you can find clientSecret in the Service Principal's page in the Azure portal)
# from azureml.core.authentication import ServicePrincipalAuthentication
sp = ServicePrincipalAuthentication(tenant_id=p_tenant_id,
                                    service_principal_id=os.environ["AML_PRINCIPAL_ID"], # clientId of service principal
                                    service_principal_password=os.environ["AML_PRINCIPAL_PASS"]) # clientSecret of service principal
ws = Workspace.get(name=ws_name,
                   subscription_id=subscription_id,
                   resource_group=resource_group,
                   auth=sp)
print("After re-authenticating to the workspace with Service Principal", 
        "ws.name: " + ws.name, 
        "ws.resource_group: " + ws.resource_group, 
        "ws.location: " + ws.location, 
        "ws.subscription_id: " + ws.subscription_id, 
sep='\n')

# Set up for Run that is to be linked with the Experiment this script is run with
run = Run.get_context()
run.log(name='creating outputs and logs directory...', value=0)
run.log(os.makedirs('./outputs', exist_ok=True), value=0)
run.log(os.makedirs('./logs', exist_ok=True), value=0)

# BEGIN Get Data
# Create datastore, try getting datastore via Workspace object
datastore = Datastore.get_default(ws)
datastore = Datastore.get(ws, p_datastore_name)

# Get the split Datasets for training and testing from the datastore of the Workspace (split by "whether target column") 
# (these Datasets must must already be registered, they should have been registered in the Notebook code)

# X_train = Dataset.get_by_name(ws, X_train_registered_name, version = 'latest').to_pandas_dataframe()
X_test  = Dataset.get_by_name(ws, X_test_registered_name, version = 'latest').to_pandas_dataframe()
# REMOVE THE weight_column
X_test_dropped_weight_column = X_test.drop([p_weight_column_name], axis=1)

# y_train = Dataset.get_by_name(ws, y_train_registered_name, version = 'latest').to_pandas_dataframe()
y_test = Dataset.get_by_name(ws, y_test_registered_name, version = 'latest').to_pandas_dataframe()

# Create training and testing data to give to AutoMLConfig
train_data = Dataset.get_by_name(ws, train_data_registered_name, version = 'latest').to_pandas_dataframe()
test_data = Dataset.get_by_name(ws, test_data_registered_name, version = 'latest').to_pandas_dataframe()

# Use AutoML to generate models
# from azureml.train.automl import AutoMLConfig

# Define Compute Cluster to use
compute_target = 'local'

# Have gotten this error before...
#      "ValueError: The truth value of a DataFrame is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().""
#       Resolve by: 
#           - Perhaps this link will help, it seems the error is a data format problem
#                (https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-auto-train#data-source-and-format)

print("Printing test_data: ", test_data)

print("Printing train_data: ", train_data)

# AutoMLConfig for properties that may change value for optimization and configuration purposes
automl_settings = {
    "primary_metric":'accuracy',
    "featurization":'auto',
    "experiment_timeout_minutes":15,
    "enable_early_stopping":True,
    "n_cross_validations":5,
    "model_explainability":True,
    "max_concurrent_iterations": 8,
    "max_cores_per_iteration": -1,
    "verbosity": logging.INFO,
}
# Leave the more or less unchanging properties as non kwargs
autoMLConfig = AutoMLConfig(task='classification',
                      compute_target=compute_target,
                      training_data=train_data,
                      label_column_name='Survived',
                      num_classes=p_num_classes,
                      weight_column_name=p_weight_column_name,
                      **automl_settings)


# Run AutoML training from here
experiment = Experiment(workspace=ws, name=p_automlconfig_experiment_name)
AutoML_run = experiment.submit(autoMLConfig, show_output = True)
print("calling wait_for_completion on the AutoML_run")
AutoML_run.wait_for_completion()

# Get the best model
bestRunAndModel = AutoML_run.get_output()
print("Printing bestRunAndModel[0]: ", bestRunAndModel[0])
print("Printing bestRunAndModel[1]: ", bestRunAndModel[1])
bestModel = bestRunAndModel[1]
print("Printing bestModel:", bestModel)

# Training the model is as simple as this
# We use the predict() on the model to predict the output
prediction = bestModel.predict(X_test_dropped_weight_column)

# Log classification metrics to evaluate the model with, using accuracy and F1 score for Classification here
accuracy = accuracy_score(y_test, prediction)
f1 = f1_score(y_test, prediction)
print("accuracy: ", accuracy)
print("f1: ", f1)
run.log('accuracy', accuracy)
run.log('f1', f1)

# for regression...
# 
# Log regression metrics to evaluate the model with, using R2 score, MSE score, and MAE score for Regression here
# from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
# r2 = r2_score(y_test, prediction)
# mse = mean_squared_error(y_test, prediction)
# mae = mean_absolute_error(y_test, prediction)
# print("r2: ", r2)
# print("mse: ", mse)
# print("mae: ", mae)
# run.log('r2', r2)
# run.log('mse', mse)
# run.log('mae', ,mae)


# TODO? Somehow get the best model downloaded to access in the Notebook (which runs in the local WSL environment)
# Save the output model to a file...
# ... This is required for the model to get automatically uploaded by the Notebook using this script
with open(p_out_model_file_name, "wb") as file:
    joblib.dump(value=bestModel, filename=os.path.join('./outputs/', p_out_model_file_name))
file.close()