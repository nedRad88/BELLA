from statistics import mean
from math import sqrt
import itertools
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict, KFold
from tqdm import tqdm
from sklearn import linear_model
import numpy as np
import scipy.stats
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import sys

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


def discretize(data, cols, n_bins, labels=None):
    """
    Discretization by binning.
    :param data: Dataframe
    :param cols: list of numerical attributes
    :param n_bins: number of bins for discretization
    :param labels: labels for discrete values
    :return: DataFrame with discretized numerical columns
    """
    if not labels:
        labels = [str(i) for i in range(n_bins)]
    for col in cols:
        data[col] = pd.cut(data[col], bins=n_bins, labels=labels)
    return data


def compute_categorical_distances(data_train, categorical_features, num_features):
    """

    :param data_train:
    :param categorical_features:
    :param num_features:
    :return:
    """
    cat_distances = {}
    data = data_train.copy(deep=True)
    data = discretize(data, num_features, n_bins=10)

    for f in categorical_features:
        cat_distances[f] = {}
        unique_values = data[f].unique()
        for item in itertools.product(unique_values, unique_values):
            if item[0] != item[1]:
                s = 0.0
                for f2 in data.columns:
                    d = 0.0
                    if f != f2 and f2 != 'target':
                        unique_f2 = data[f2].unique()
                        for val in unique_f2:
                            d += max(len(data[(data[f] == item[0]) & (data[f2] == val)]) / len(data),
                                     len(data[(data[f] == item[1]) & (data[f2] == val)]) / len(data))
                        d = d - 1.0
                        s += d
                cat_distances[f][(item[0], item[1])] = s / (len(categorical_features) + len(num_features) - 1)

    return cat_distances


def compute_distances_cat(dict_of_dist, point, train_set):
    cat_distance = pd.Series(index=train_set.index, dtype=float)
    for index, row in train_set[dict_of_dist.keys()].iterrows():
        distance = 0.0
        row_dict = row.to_dict()
        for c in dict_of_dist.keys():
            if point[c].values[0] != row_dict[c]:
                if isinstance(point[c].values[0], float) or isinstance(row_dict[c], float):
                    value1 = int(point[c].values[0])
                    value2 = int(row_dict[c])
                else:
                    value1 = point[c].values[0]
                    value2 = row_dict[c]
                distance += abs(dict_of_dist[c][(value1, value2)])
        cat_distance.loc[index] = distance

    return cat_distance


def calculate_vif(df, atts):
    stop = False
    atts2 = [item for item in atts]
    df = df.drop('target', axis=1)
    while len(atts2) > 1 and stop is False:
        max_vif = 0.0
        worst_feature = None
        for feature in atts2:
            X = [f for f in atts2 if f != feature]
            X, y = df[X], df[feature]  # extract r-squared from the fit
            r2 = linear_model.LinearRegression().fit(X, y).score(X, y)
            if r2 == 1:
                vif = 100.0
            else:
                vif = 1 / (1 - r2)  # return VIF DataFrame
            if vif >= max_vif:
                max_vif = vif
                worst_feature = feature
        if max_vif >= 10.0:
            atts2.remove(worst_feature)
            stop = False
        else:
            stop = True

    return atts2


def compute_stats(values):
    return mean(values), np.var(values)


def margin_of_error(true_values, predicted):
    delta_values = np.square(np.subtract(predicted, true_values))
    all_mu = np.square(predicted[None, :] - true_values[:, None])
    mu_values = all_mu.mean(axis=1)

    delta_mean, delta_var = compute_stats(delta_values)
    mu_mean = mean(mu_values)

    moe_delta95 = scipy.stats.t.ppf(q=1 - .05 / 2, df=len(true_values) - 1) * sqrt(delta_var) / sqrt(len(true_values))

    return 1 - min(1, (abs(delta_mean + moe_delta95)) / (abs(mu_mean)))


@ignore_warnings(category=ConvergenceWarning)
def train_lin_model(data, atts):
    new_atts = calculate_vif(data, atts=atts)
    lmodel = linear_model.LassoCV(cv=5, random_state=1)
    lmodel.fit(data[new_atts], data['target'])
    alpha_i = lmodel.alphas_.tolist().index(lmodel.alpha_)
    mses = lmodel.mse_path_[alpha_i]
    ste = sqrt(np.var(mses)) / sqrt(len(mses))
    best_alpha = lmodel.alpha_
    max_mse = np.mean(mses) + ste
    for i in range(len(lmodel.alphas_)):
        if np.mean(lmodel.mse_path_[i]) < max_mse:
            best_alpha = lmodel.alphas_[i]
            break
    lmodel = linear_model.Lasso(alpha=best_alpha)
    lmodel.fit(data[new_atts], data['target'])
    new_atts = []
    for i in range(len(lmodel.coef_)):
        if lmodel.coef_[i] != 0.0:
            new_atts.append(lmodel.feature_names_in_[i])

    if len(new_atts) == 0:
        return 0.0, None, None, 0.0
    lmodel = linear_model.LinearRegression()
    cv_splitter = KFold()
    predicted = cross_val_predict(lmodel, data[new_atts], data['target'], cv=cv_splitter)
    lmodel = linear_model.LinearRegression()
    lmodel.fit(data[new_atts], data['target'])
    lb95 = margin_of_error(data['target'].values, predicted)
    mse = mean_squared_error(data['target'].tolist(), lmodel.predict(data[new_atts]))
    return lb95, lmodel, new_atts, mse


def show_explanation(explanation, offset):
    exp_features = list(explanation.keys())
    importance = list(explanation.values())
    x_pos = np.arange(len(exp_features))
    colors = ['b'] * len(importance)
    for i in range(len(importance)):
        if importance[i] < 0.0:
            colors[i] = 'r'
    plt.bar(x_pos, importance, align='center', color=colors, bottom=offset)
    plt.xticks(x_pos, exp_features, rotation=45)
    plt.ylabel('Feature contribution')
    plt.show()


def explain(train, train_dummy, explain_point, explain_point_dummy, bin_fs, cat_dist, num_fs, verbose=False):
    global lower_bound, predict_error
    df = train.copy(deep=True)
    data_dist = train_dummy.copy(deep=True)
    e_index = explain_point.index[0]
    if e_index not in train.index:
        df = pd.concat([pd.DataFrame(explain_point), df])
        data_dist = pd.concat([pd.DataFrame(explain_point_dummy), data_dist])

    data_dist['distance_metric'] = compute_distances_cat(cat_dist, explain_point, df)
    if dataset_name != 'servo':
        data_dist['distance_metric'] += manhattan_distances(explain_point[num_fs + bin_fs], data_dist[num_fs + bin_fs])[0]
    # Sort by target diff and distance_metric
    data_dist = data_dist.sort_values(['distance_metric'], ascending=[True])
    # data_dist['target'] = df['target']
    data_dist = data_dist.drop('distance_metric', axis=1)
    fs = [c for c in data_dist.columns if c != 'target']
    outside_box = data_dist.copy(deep=True)
    best_f1 = 0.0
    explain_box = 0
    best_model = None
    inside_box = pd.DataFrame([data_dist.loc[explain_point.index[0]]])
    outside_box = outside_box.drop([explain_point.index[0]])
    data_dist = data_dist.drop([explain_point.index[0]])
    best_vars = None
    new_vars = None
    for i, g in data_dist.groupby(np.arange(len(data_dist)) // (round(len(data_dist) / 100))):
        inside_box = pd.concat([inside_box, g])
        outside_box = outside_box.drop(list(g.index))
        if len(inside_box) >= max(10, 2 * len(fs)):
            f95, model, new_vars, me = train_lin_model(inside_box, fs)
            if len(inside_box) not in lower_bound90.keys():
                lower_bound90[len(inside_box)] = {}
                lower_bound90[len(inside_box)]['bound_sum'] = f95
                lower_bound90[len(inside_box)]['n_app'] = 1
            else:
                lower_bound90[len(inside_box)]['bound_sum'] += f95
                lower_bound90[len(inside_box)]['n_app'] += 1
        else:
            model = None
            f95 = 0.0
        if f95 > best_f1:
            best_f1 = f95
            best_model = model
            best_vars = new_vars
            explain_box = inside_box.copy(deep=True)
    explanation = {}
    e_point = df.loc[e_index]
    if best_model:
        for i in range(len(best_model.feature_names_in_)):
            if abs(best_model.coef_[i]) > 0.0:
                explanation[best_model.feature_names_in_[i]] = best_model.coef_[i]
        explanation = dict(sorted(explanation.items(), key=lambda x: abs(x[1]), reverse=True))

    if verbose:
        show_explanation(explanation, best_model.intercept_)

    return explain_box, best_model, explanation, best_vars


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
explain_indexes = explain_indexes[:20]

bb_model = RandomForestRegressor(n_estimators=1000)
# bb_model = MLPRegressor(hidden_layer_sizes=(500,), max_iter=4000, learning_rate_init=0.005, learning_rate='adaptive', batch_size=256)
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
    exp_box, exp_model, exp, new_fs = explain(train, X_dummy, exp_point, exp_point_dummy, binary_features, categorical_dis, numerical_features, verbose=False)
    exp_acc.append(mean_squared_error(exp_point['target'], exp_model.predict(exp_point_dummy[exp_model.feature_names_in_])))
    exp_point_original = pd.DataFrame([data.loc[explain_index]])
    exp_point_original_copy = exp_point_original.copy(deep=True)
    if exp_point['target'].values[0] + thirty_qle < target_max:
        ref_target = exp_point['target'].values[0] + thirty_qle
    else:
        ref_target = exp_point['target'].values[0] - thirty_qle
    stop = False
    while not stop:
        potential_refs = train[(ref_target - ref_target * eps <= train['target']) & (train['target'] <= ref_target + eps * ref_target)]
        if len(potential_refs) == 0:
            eps += 0.05
            stop = False
        else:
            stop = True

    top_k_potential = {}
    for index, row in potential_refs.iterrows():
        candidate = pd.DataFrame([train.loc[index]])
        dist = compute_distances_cat(categorical_dis, exp_point, candidate).values[0]
        if dataset_name != 'servo':
            dist += manhattan_distances(exp_point[numerical_features + binary_features], candidate[numerical_features + binary_features])[0][0]
        top_k_potential[index] = dist
    top_k_potential = dict(sorted(top_k_potential.items(), key=lambda x: abs(x[1]), reverse=True)[:5])
    potential_refs = potential_refs.loc[list(top_k_potential.keys())]
    potential_refs_dummy = X_dummy.loc[list(top_k_potential.keys())]
    ref_models = {}
    dist3_min = 10000000000.0
    exp_box_size = len(train)
    counter3 = None

    for ref_i, ref_row in potential_refs.iterrows():
        exp_point_copy = exp_point_dummy.copy(deep=True)
        ref_exp_point = pd.DataFrame([potential_refs.loc[ref_i]])
        ref_exp_point_dummy = pd.DataFrame([X_dummy.loc[ref_i]])
        exp_box, exp_model, exp, new_fs = explain(train, X_dummy, ref_exp_point, ref_exp_point_dummy, binary_features, categorical_dis, numerical_features, verbose=False)
        for feature in exp_model.feature_names_in_:
            exp_point_copy.iloc[0][feature] = ref_exp_point_dummy[feature]
        if len(exp_model.feature_names_in_) in ref_errors:
            ref_errors[len(exp_model.feature_names_in_)]['error'] += (ref_row['target'] - bb_model.predict(exp_point_copy[dummy_features])[0])**2
            ref_errors[len(exp_model.feature_names_in_)]['n'] += 1
        else:
            ref_errors[len(exp_model.feature_names_in_)] = {}
            ref_errors[len(exp_model.feature_names_in_)]['error'] = (ref_row['target'] - bb_model.predict(exp_point_copy[dummy_features])[0])**2
            ref_errors[len(exp_model.feature_names_in_)]['n'] = 1
        ref_models[ref_i] = exp_model
        if len(new_fs) > 1:
            dist3 = compute_distances_cat(categorical_dis, ref_exp_point, exp_point).values[0]
            if dataset_name != 'servo':
                # print([ref_row[numerical_features + binary_features]], [exp_point[numerical_features+binary_features].squeeze()])
                dist3 += manhattan_distances([ref_row[numerical_features + binary_features]], [exp_point[numerical_features+binary_features].squeeze()])[0][0]
            # print(ref_exp_point_dummy[new_fs], exp_point_copy[new_fs])
            dist3 += manhattan_distances(ref_exp_point_dummy[new_fs], exp_point_copy[new_fs]) / len(exp_model.feature_names_in_)
        else:
            dist3 = compute_distances_cat(categorical_dis, ref_exp_point, exp_point).values[0]
            if dataset_name != 'servo':
                dist3 += manhattan_distances([ref_row[numerical_features + binary_features]], [exp_point[numerical_features + binary_features].squeeze()])[0][0]
            dist3 += abs(ref_exp_point_dummy[new_fs].values - exp_point_copy[new_fs].values)

        if dist3 <= dist3_min:
            dist3_min = dist3
            counter3 = ref_i
            f_len3 = len(exp_model.feature_names_in_)

    exp_point_copy = exp_point_dummy.copy(deep=True)
    if counter3 and counter3 in ref_models:
        for feature in ref_models[counter3].feature_names_in_:
            exp_point_copy.iloc[0][feature] = X_dummy.loc[counter3][feature]
        error3['e'].append(
            (potential_refs.loc[counter3]['target'] - bb_model.predict(exp_point_copy[dummy_features])[0]) ** 2)
        error3['l'].append(len(ref_models[counter3].feature_names_in_))

        """
        counter_explanation = {}
        for f in features:
            val = exp_point_original_copy[f].values[0] - exp_point_original[f].values[0]
            if val != 0.0:
                counter_explanation[f] = val
        counter_explanation = dict(sorted(counter_explanation.items(), key=lambda x: abs(x[1]), reverse=True))
        """

print("Dataset RF: ", dataset_name)
print("BB error: ", bb_model_acc, bb_model_acc_root)
print("exp error", mean(exp_acc), exp_acc)
print("Counter error: ", mean(error3['e']), error3['e'])
print("length of change: ", mean(error3['l']), error3['l'])
print(ref_errors)
print(error3)

output_file = open("lcounter_results_" + dataset_name + "_rf.txt", 'a')
output_file.writelines(dataset_name+", counter \nBB Model acc: {},\nBB RMSE: {},\nMean Exp acc: {} \nExp acc: {} \n"
                       "Counter error: {} \nMean Counter error: {} \nmean Length: {} \nLength: {} \n"
                        .format(bb_model_acc, bb_model_acc_root, mean(exp_acc), exp_acc, error3['e'], mean(error3['e']),
                       mean(error3['l']), error3['l']))
