from sklearn.model_selection import train_test_split
from azureml.core.run import Run
import os
import numpy as np

# sklearn.externals.joblib is removed in 0.23
try:
    from sklearn.externals import joblib
except ImportError:
    import joblib

os.makedirs('./outputs', exist_ok=True)

X, y = 'Populate', 'Us' #load_diabetes(return_X_y=True)

run = Run.get_context()

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=0)
data = {"train": {"X": X_train, "y": y_train},
        "test": {"X": X_test, "y": y_test}}