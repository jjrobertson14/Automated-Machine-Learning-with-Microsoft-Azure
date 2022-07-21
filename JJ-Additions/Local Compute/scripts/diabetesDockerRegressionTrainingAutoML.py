from azureml.core import Workspace, Dataset, Datastore, Experiment
from azureml.core.run import Run

import getopt
import joblib
import os
import sys

import pandas as pd
import numpy as np

#Importing the Decision Tree from scikit-learn library
# Metrics for Evaluation of model: RSME and R2 for regression
from sklearn.metrics  import mean_squared_error,r2_score

# Import the rest
from interpret.ext.blackbox import TabularExplainer
from azureml.interpret import ExplanationClient
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
    'x-train-test-y-train-test-combined-train-test='
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
    else:
        print("Unrecognized option passed, continuing run, it is: " + opt)

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

# Get the split (by whether target column) Datasets for training and testing from the datastore of the Workspace
# (having registered them already in Notebook code)
# X_train = Dataset.get_by_name(ws, X_train_registered_name, version = 'latest').to_pandas_dataframe()
X_test  = Dataset.get_by_name(ws, X_test_registered_name, version = 'latest').to_pandas_dataframe()
# y_train = Dataset.get_by_name(ws, y_train_registered_name, version = 'latest').to_pandas_dataframe()
y_test = Dataset.get_by_name(ws, y_test_registered_name, version = 'latest').to_pandas_dataframe()

# Create training and testing data to give to AutoMLConfig
train_data = Dataset.get_by_name(ws, train_data_registered_name, version = 'latest').to_pandas_dataframe()
test_data = Dataset.get_by_name(ws, test_data_registered_name, version = 'latest').to_pandas_dataframe()

# Use AutoML to generate models
# from azureml.train.automl import AutoMLConfig

# Basic Variables for AutoMLConfig
target_column = 'Y'
task = 'regression'
primary_metric = 'normalized_root_mean_squared_error'
featurization = 'auto'

# Define Compute Cluster to use
compute_target = 'local'

# TODO get experiment.submit() working with the autoMLConfig
# Getting this error
#      "ValueError: The truth value of a DataFrame is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().""
#       Resolve by: 
#           - Perhaps this link will help, it seems the error is a data format problem
#                (https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-auto-train#data-source-and-format)

print(test_data)

print(train_data)

# AutoMLConfig
autoMLConfig = AutoMLConfig(task=task,
                      primary_metric=primary_metric,
                      featurization=featurization,
                      compute_target=compute_target,
                      training_data=train_data,
                      label_column_name=target_column,
                      experiment_timeout_minutes=15,
                      enable_early_stopping=True,
                      n_cross_validations=5,
                      model_explainability=True)
                      #   test_data=test_data,


# TODO replace with the following notation
#    automl_settings = {
#        "n_cross_validations": 3,
#        "primary_metric": 'r2_score',
#        "enable_early_stopping": True,
#        "experiment_timeout_hours": 1.0,
#        "max_concurrent_iterations": 4,
#        "max_cores_per_iteration": -1,
#        "verbosity": logging.INFO,
#    }

#    autoMLConfig = AutoMLConfig(task = 'regression',
#                                compute_target = compute_target,
#                                training_data = train_data,
#                                label_column_name = label,
#                                **automl_settings
#                                )

# Run autoML training from here
#       - perhaps create a child run (Run.child_run to create a child run)
#           - (https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.run.run?view=azure-ml-py)
#       - perhaps look up using ScriptRunConfig along with AutoMLConfig
# Get Best Model from the AutoML run
experiment_name = 'Diabetes_Docker_Regression_Training_AutoML'
experiment = Experiment(workspace=ws, name=experiment_name)
AutoML_run = experiment.submit(autoMLConfig, show_output = True)
print("calling wait_for_completion on the AutoML_run")
AutoML_run.wait_for_completion()
# TODO stop "Exiting early"
# TODO make sure rest of script works
print("Exiting early")
exit()

# TODO Set best model
bestModel = "blah"

# Training the model is as simple as this
# We use the predict() on the model to predict the output
prediction = bestModel.predict(X_test)

# Log regression metrics to evaluate the model with, using R2 score and RSME score for Regression here
r2 = r2_score(y_test, prediction)
rsme = mean_squared_error(y_test, prediction)
print("r2: ", r2)
print("rsme: ", rsme)
run.log('r2', r2)
run.log('rsme', rsme)

# for regression...
#
# we use R2 score and RSME(root mean squared error)
# all other steps will be same as classification as shown above
# 
# from sklearn.metrics import mean_squared_error
# from sklearn.metrics import r2_score
# print(mean_squared_error(y_test,prediction))
# print(r2_score(y_test,prediction))


# Save the output model to a file...
# ... This is required for the model to get automatically uploaded by the Notebook using this script
with open(p_out_model_file_name, "wb") as file:
    joblib.dump(value=bestModel, filename=os.path.join('./outputs/', p_out_model_file_name))
file.close()




# BEGIN Add Explanations (In terms of engineered features)
# from interpret.ext.blackbox import TabularExplainer
# from azureml.interpret import ExplanationClient
client = ExplanationClient.from_run(run)

# TODO get explanations working by giving model to the explainer
# BEGIN Add Engineered Feature Explanations
# Fit the model
# Explain in terms of engineered features
# NOTE: regressor_pipeline.steps[-1][1] contains the Model
# NOTE: "features" field is optional for TabularExplainers
# from interpret.ext.blackbox import TabularExplainer
engineered_explainer = TabularExplainer(bestModel,
                                     initialization_examples=X_test,
                                     features=p_feature_names)
# Explain results with this Explainer and upload the Explanation...

# Get Global Explanations of raw features, global as in 'of total data'...
# You can use the training data or the test data here, but test data would allow you to use Explanation Exploration
# print("X_test, line value before engineered_explainer.explain_global: \n" + str(X_test))
global_explanation = engineered_explainer.explain_global(X_test, y_test)
# If you used the PFIExplainer in the previous step, use the next line of code instead
# global_explanation = engineered_explainer.explain_global(X_test, true_labels=y_test)
# Sorted feature importance values and feature names
sorted_global_importance_values = global_explanation.get_ranked_global_values()
sorted_global_importance_names = global_explanation.get_ranked_global_names()
globalFeatureExplanations = dict(zip(sorted_global_importance_names, sorted_global_importance_values))
print('globalFeatureExplanations: ', globalFeatureExplanations)
# Alternatively, you can print out a dictionary that holds the top K feature names and values
print('global_explanation.get_feature_importance_dict(): ', global_explanation.get_feature_importance_dict())

# Upload the explanation in terms of engineered features
client = ExplanationClient.from_run(run)
client.upload_model_explanation(global_explanation, true_ys=y_test.values.ravel(), comment='global explanation: test dataset features, engineered')