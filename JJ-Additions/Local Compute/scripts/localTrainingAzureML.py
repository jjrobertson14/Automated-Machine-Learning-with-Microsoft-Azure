from azureml.core import Workspace, Dataset, Datastore
from azureml.core.run import Run
from azureml.interpret import ExplanationClient

import os
import sys
import getopt
import joblib

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
    'features='
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
    elif opt == '--features':
        # take in list of strings argument
        p_features = arg
        print('p_features was set with value: ', p_features)
        print('p_features is now a: ', type(p_features))
        # The arg value is a string that looks like a list: '["col1","col2"]'
        # So,turn string of features into list of features
        # (Remove square brackets and " marks)
        p_features = p_features.replace("[","")
        p_features = p_features.replace("]","")
        p_features = p_features.replace("\"","")
        print('On character replacement, p_features was set with value: ', p_features)
        # (Now split on comma)
        p_features = p_features.split(',')
        print('On split, p_features was set with value: ', p_features)
        print('p_features is now, after decoding string into list, a: ', type(p_features))
    else:
        print("Unrecognized option passed, continuing run, it is: " + opt)

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
client = ExplanationClient.from_run(run)
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
X_train = Dataset.get_by_name(ws, "Titanic Feature Column Data for training", version = 'latest').to_pandas_dataframe()
X_test  = Dataset.get_by_name(ws, "Titanic Feature Column Data for testing", version = 'latest').to_pandas_dataframe()
y_train = Dataset.get_by_name(ws, "Titanic Target Column Data for training", version = 'latest').to_pandas_dataframe()
y_test = Dataset.get_by_name(ws, "Titanic Target Column Data for testing", version = 'latest').to_pandas_dataframe()
data = {"train": {"X": X_train, "y": y_train},
        "test": {"X": X_test, "y": y_test}}

# Training the model is as simple as this
# Use the function imported above and apply fit() on it
model_DT = DecisionTreeClassifier()
model_DT.fit(X_train,y_train)

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




# BEGIN Add Explanations
from interpret.ext.blackbox import TabularExplainer

# "features" and "classes" fields are optional
explainer = TabularExplainer(model_DT, 
                             X_train, 
                             features=p_features)

# BEGIN Get Global Explanations, global as in 'of total data'...

# you can use the training data or the test data here, but test data would allow you to use Explanation Exploration
print("X_test line before explainer.explain_global: \n" + str(X_test))
global_explanation = explainer.explain_global(X_test)
# if you used the PFIExplainer in the previous step, use the next line of code instead
# globa l_explanation = explainer.explain_global(x_train, true_labels=y_train)
# sorted feature importance values and feature names
sorted_global_importance_values = global_explanation.get_ranked_global_values()
sorted_global_importance_names = global_explanation.get_ranked_global_names()
globalFeatureExplanations = dict(zip(sorted_global_importance_names, sorted_global_importance_values))
print('globalFeatureExplanations: ', globalFeatureExplanations)
# alternatively, you can print out a dictionary that holds the top K feature names and values
print('global_explanation.get_feature_importance_dict(): ', global_explanation.get_feature_importance_dict())

# Uploading global model explanation data...
# the explanation can then be downloaded on any compute
# multiple explanations can be uploaded
client.upload_model_explanation(global_explanation, true_ys=y_test.values.ravel(), comment='global explanation: all features')
# or you can only upload the explanation object with the top k feature info
#client.upload_model_explanation(global_explanation, top_k=2, comment='global explanation: Only top 2 features')

# BEGIN Get Local Explanation, local as in 'of specified part of data'...

# get explanation for the first few data points in the test set
local_explanation = explainer.explain_local(X_test[0:5])
# sorted feature importance values and feature names
sorted_local_importance_names = local_explanation.get_ranked_local_names()
print('sorted_local_importance_names: ', sorted_local_importance_names)
sorted_local_importance_values = local_explanation.get_ranked_local_values()
print('sorted_local_importance_values: ', sorted_local_importance_values)
# Warning, there was an error when I ran this, and I left print statements to help
# localFeatureExplanations = dict(zip(sorted_local_importance_names, sorted_local_importance_values))
# print('localFeatureExplanations: ', localFeatureExplanations)

exit(0)

# TODO? get to appear in visualization's Model Performance tab: the accuracy, precision, f1 scores, false positive rates, false negative rates