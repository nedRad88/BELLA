from sklearn.model_selection import train_test_split
from tqdm import tqdm
import scipy.stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import sys
from bella import *


dataset_name = sys.argv[1]
classification = False

lower_bound90 = {}
predict_error = []


def compute_exp_distance(dict1, dict2, features):
    distance = 0.0
    for f in features:
        if f in dict1:
            v1 = dict1[f]
        else:
            v1 = 0.0
        if f in dict2:
            v2 = dict2[f]
        else:
            v2 = 0.0
        distance += abs(v2 - v1)

    return distance


data = pd.read_csv("./datasets/regression/" + dataset_name + ".csv", header=0)

n_data_points = {}
l_n_points = []
l_exp_len = []
metric = []
exp_len = {}
target_max = max(data['target'])
target_min = min(data['target'])
thirty_qle = 0.3 * (target_max - target_min)
features = [f_name for f_name in data.columns if f_name != 'target']
train, test = train_test_split(data, test_size=0.1)
binary_features = []
categorical_features = []
numerical_features = []
if dataset_name == 'bike':
    binary_features = ['holiday', 'functioningday']
    categorical_features = ['seasons']
if dataset_name == 'auto':
    categorical_features = ['origin']
if dataset_name == 'servo':
    for f in features:
        if f not in binary_features and f not in numerical_features:
            categorical_features.append(f)
for f in features:
    if f not in binary_features and f not in categorical_features:
        numerical_features.append(f)

if len(categorical_features) > 0:
    categorical_dis = compute_categorical_distances(train, categorical_features, numerical_features)
else:
    categorical_dis = None

if dataset_name != 'servo':
    standardizer = StandardScaler()
    scaled_train = train.copy(deep=True)
    scaled_test = test.copy(deep=True)
    train_num = scaled_train[numerical_features]
    test_num = scaled_test[numerical_features]
    train_num = standardizer.fit_transform(train_num.values)
    test_num = standardizer.transform(test_num.values)
    train[numerical_features] = train_num
    test[numerical_features] = test_num
explain_indexes = test[:20]

bb_model = RandomForestRegressor(n_estimators=1000)
# bb_model = MLPRegressor(hidden_layer_sizes=(500,), max_iter=4000, learning_rate_init=0.005, learning_rate='adaptive',
#                        batch_size=256)
Y = train['target']
X = train.drop('target', axis=1)
X_dummy = pd.get_dummies(data=X, columns=categorical_features)
X_test_dummy = pd.get_dummies(data=test, columns=categorical_features)
for c in X_dummy.columns:
    if c != 'target':
        if c not in X_test_dummy.columns:
            X_test_dummy[c] = 0

for c in X_test_dummy.columns:
    if c != 'target':
        if c not in X_dummy.columns:
            X_dummy[c] = 0
dummy_features = X_dummy.columns
bb_model.fit(X_dummy, Y)
y_bb = bb_model.predict(X_dummy)
y_test = test['target']
test = test.drop('target', axis=1)
test['target'] = bb_model.predict(X_test_dummy[dummy_features])
train.drop('target', axis=1)
train['target'] = y_bb
X_dummy['target'] = y_bb
X_test_dummy['target'] = test['target']
bb_model_acc = mean_squared_error(y_test, bb_model.predict(X_test_dummy[dummy_features]))
bb_model_acc_root = mean_squared_error(y_test, bb_model.predict(X_test_dummy[dummy_features]), squared=False)
print(bb_model_acc, bb_model_acc_root)
total_count = 0
ref_errors = {}
exp_acc = []
error3 = {}
error3['e'] = []
error3['l'] = []
for explain_index in tqdm(explain_indexes):
    eps = 0.05
    features = [f_name for f_name in data.columns if f_name != 'target']
    exp_point = pd.DataFrame([test.loc[explain_index]])
    exp_point_dummy = pd.DataFrame([X_test_dummy.loc[explain_index]])
    if exp_point['target'].values[0] + thirty_qle < target_max:
        ref_target = exp_point['target'].values[0] + thirty_qle
    else:
        ref_target = exp_point['target'].values[0] - thirty_qle

    exp_box, exp_model, exp, new_fs = explain(train, X_dummy, exp_point, exp_point_dummy, binary_features,
                                              categorical_dis, numerical_features, reference_value = ref_target,
                                              verbose=False)



print("Dataset RF: ", dataset_name)
print("BB error: ", bb_model_acc, bb_model_acc_root)
print("exp error", mean(exp_acc), exp_acc)
print("Counter error: ", mean(error3['e']), error3['e'])
print("length of change: ", mean(error3['l']), error3['l'])
