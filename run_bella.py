from bella import *
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from utils import *
import warnings
import numpy as np

warnings.filterwarnings('ignore')

eps = 0.05

def run_bella(train, target_column, binary_features, categorical_features, numerical_features, explain_point,
              standardize=True, reference_value=None):
    test = pd.DataFrame(explain_point)
    features = [f_name for f_name in train.columns if f_name != target_column]
    train_dummy = None
    test_dummy = None
    categorical_dis = None
    if len(categorical_features) > 0:
        categorical_dis = compute_categorical_distances(data, categorical_features, numerical_features)

    # Standardize numerical features
    if len(numerical_features) > 0 and standardize:
        standardizer = StandardScaler()
        scaled_train = train.copy(deep=True)
        scaled_test = test.copy(deep=True)
        train_num = scaled_train[numerical_features]
        test_num = scaled_test[numerical_features]
        train_num = standardizer.fit_transform(train_num.values)
        test_num = standardizer.transform(test_num.values)
        train[numerical_features] = train_num
        test[numerical_features] = test_num

    X = train.drop(target_column, axis=1)
    train = train.drop(target_column, axis=1)
    test = test.drop(target_column, axis=1)
    if len(categorical_features) > 0:
        train_dummy = pd.get_dummies(data=X, columns=categorical_features)
        test_dummy = pd.get_dummies(data=test, columns=categorical_features)
        dummy_features = train_dummy.columns
    explain_indexes = test.index.tolist()

    # For smaller datasets,
    if train_dummy is not None:
        for c in train_dummy.columns:
            if c != target_column:
                if c not in test_dummy.columns:
                    test_dummy[c] = 0

        for c in test_dummy.columns:
            if c != target_column:
                if c not in train_dummy.columns:
                    train_dummy[c] = 0

    for explain_index in tqdm(explain_indexes):
        exp_point = pd.DataFrame([test.loc[explain_index]])
        explain_point_dummy = None
        if train_dummy is not None:
            explain_point_dummy = pd.DataFrame([test_dummy.loc[explain_index]])
        if reference_value:
            c_exp_model, new_data_point, counterfactual = explain(train, exp_point, binary_features, categorical_dis,
                                                                  numerical_features,
                                                                  explain_point_dummy=explain_point_dummy,
                                                                  train_dummy=train_dummy,
                                                                  reference_value=ref_target, verbose=True)
            print(counterfactual)
        else:
            exp_box, exp_model, exp = explain(train, exp_point, binary_features, categorical_dis, numerical_features,
                                          explain_point_dummy=explain_point_dummy, train_dummy=train_dummy,
                                          verbose=True)

            print(exp)