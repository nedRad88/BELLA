from lime import lime_tabular
import re
from bella import *
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import shap
import warnings
warnings.filterwarnings('ignore')


classification = False

predict_error = []
dataset_name = "customer_churn"

datasets = ['auto', 'concrete', 'customer_churn', 'real_estate', 'servo', 'winequality', 'bike', 'cpu', 'echo', 'wind'
            'electrical', 'superconduct']


def compute_confidence(Xtrain, path):
    total = 0
    for i in range(len(Xtrain)):
        d = Xtrain.iloc[i]
        if eval(path):
            total += 1
    return total


def explain_predict(explanation_model, neighbourhood_data):
    true = []
    predicted = []
    for index, row in neighbourhood_data.iterrows():
        prediction = explanation_model['base']
        true.append(row['target'])
        for key, value in explanation_model.items():
            if key != 'base':
                prediction += value
        predicted.append(prediction)

    return mean_squared_error(true, predicted)


def to_code(string):
    code = string
    code = code.replace(" ", "")
    code = re.sub("[a-z]+", lambda m: "d['%s']" % m.group(0), code)
    try:
        code = re.sub("<(d['[a-z]+'])<=", lambda m: "<%s and %s<=" % (m.group(1), m.group(1)), code)
    except re.error:
        pass
    try:
        code = re.sub("(d['[a-z]+'])=", lambda m: "%s==" % m.group(1), code)
    except re.error:
        pass
    # code = re.sub("\b(?!and)\b\S+", lambda m: "d['%s']" % m.group(0), code)
    code = code.replace(",", ") and (")
    code = code.replace(";", " or ")
    return code


def compute_exp_distance(dict1, dict2, features):
    distance = 0.0
    n_features = 0
    for f in features:
        if f in dict1:
            v1 = dict1[f]
        else:
            v1 = 0.0
        if f in dict2:
            v2 = dict2[f]
        else:
            v2 = 0.0
        if f in dict1 or f in dict2:
            n_features += 1
        if v1 != 0.0 or v2 != 0.0:
            distance += abs(v2 - v1)/(abs(v1) + abs(v2))

    return 1 - distance/len(features)


def compute_exp_distance_shap(l1, l2, num_of_features):
    distance = 0.0
    for i in range(len(l1)):
        if l1[i] != 0.0 or l2[i] != 0.0:
            distance += abs(l2[i] - l1[i]) / (abs(l1[i]) + abs(l2[i]))
    return 1 - distance/num_of_features


for dataset_name in datasets:
    data = pd.read_csv("./datasets/" + dataset_name + ".csv", header=0)
    eps = 0.05
    bella_results = {'accuracy': [], 'length': [], 'generality': [], 'robustness': []}
    lime_results = {'accuracy': [], 'length': [], 'generality': [], 'robustness': []}
    shap_results = {'accuracy': [], 'length': [], 'generality': [], 'robustness': []}

    attributes = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                  'u', 'v', 'w', 'x', 'y', 'z', 'aa', 'ab', 'ac', 'ad', 'ae', 'af', 'ag', 'ah', 'ai', 'aj', 'ak', 'al',
                  'am', 'an', 'ao', 'ap', 'aq', 'ar', 'as', 'at', 'au', 'av', 'aw', 'ax', 'ay', 'az', 'ba', 'bb', 'bc',
                  'bd', 'be', 'bf', 'bg', 'bh', 'bi', 'bj', 'bk', 'bl', 'bm', 'bn', 'bo', 'bp', 'bq', 'br', 'bs', 'bt',
                  'bu', 'bv', 'bw', 'bx', 'by', 'bz']

    features = [f_name for f_name in data.columns if f_name != 'target']
    train_dummy = None
    test_dummy = None
    dummy_features = None
    categorical_dis = None

    # Set features' types for each dataset
    if dataset_name == 'servo':
        binary_features = []
        categorical_features = []
        numerical_features = []
        for f in features:
            if f not in binary_features and f not in numerical_features:
                categorical_features.append(f)
        categorical_dis = compute_categorical_distances(data, categorical_features, numerical_features)

    elif dataset_name == 'bike':
        binary_features = ['Holiday', 'Functioning_day']
        categorical_features = ['Seasons']
        numerical_features = []
        for f in features:
            if f not in binary_features and f not in categorical_features:
                numerical_features.append(f)

        categorical_dis = compute_categorical_distances(data, categorical_features, numerical_features)

    elif dataset_name == 'auto':
        binary_features = []
        categorical_features = ['origin']
        numerical_features = []
        for f in features:
            if f not in binary_features and f not in categorical_features:
                numerical_features.append(f)

        categorical_dis = compute_categorical_distances(data, categorical_features, numerical_features)

    elif dataset_name == 'echo':
        binary_features = ['still_alive', 'pericardial', 'alive_at_1']
        categorical_features = []
        numerical_features = []
        for f in features:
            if f not in binary_features and f not in categorical_features:
                numerical_features.append(f)

    elif dataset_name == 'customer_churn':
        binary_features = ["contract", "active", "complains"]
        categorical_features = []
        numerical_features = []
        for f in features:
            if f not in binary_features and f not in categorical_features:
                numerical_features.append(f)

    else:
        binary_features = []
        categorical_features = []
        numerical_features = []
        for f in features:
            if f not in binary_features and f not in categorical_features:
                numerical_features.append(f)

    train, test = train_test_split(data, test_size=0.1)

    if len(numerical_features) > 0:
        standardizer = StandardScaler()
        scaled_train = train.copy(deep=True)
        scaled_test = test.copy(deep=True)
        train_num = scaled_train[numerical_features]
        test_num = scaled_test[numerical_features]
        train_num = standardizer.fit_transform(train_num.values)
        test_num = standardizer.transform(test_num.values)
        train[numerical_features] = train_num
        test[numerical_features] = test_num

    Y = train['target']
    X = train.drop('target', axis=1)
    y_test = test['target']
    train = train.drop('target', axis=1)
    test = test.drop('target', axis=1)
    if len(categorical_features) > 0:
        train_dummy = pd.get_dummies(data=X, columns=categorical_features)
        test_dummy = pd.get_dummies(data=test, columns=categorical_features)
        dummy_features = train_dummy.columns
    explain_indexes = test.index.tolist()

    bb_model = RandomForestRegressor(n_estimators=1000)
    # bb_model = MLPRegressor(hidden_layer_sizes=(500,), max_iter=4000, learning_rate_init=0.005,
    #                        learning_rate='adaptive', batch_size=128)
    if train_dummy is not None:
        bb_model.fit(train_dummy, Y)
        y_bb = bb_model.predict(train_dummy)
        X_train_summary = shap.kmeans(train_dummy, 10)  # Train dataset summary for SHAP
        train_dummy['target'] = y_bb
        bb_model_acc = mean_squared_error(y_test, bb_model.predict(test_dummy[dummy_features]))
        ytest_bb = bb_model.predict(test_dummy[dummy_features])
        test_dummy['target'] = ytest_bb
    else:
        bb_model.fit(train, Y)
        y_bb = bb_model.predict(train)
        X_train_summary = shap.kmeans(train, 10)  # Train dataset summary for SHAP
        train['target'] = y_bb
        bb_model_acc = mean_squared_error(y_test, bb_model.predict(test))
        ytest_bb = bb_model.predict(test)

    test['target'] = ytest_bb
    train['target'] = y_bb

    if train_dummy is not None:
        for c in train_dummy.columns:
            if c != 'target':
                if c not in test_dummy.columns:
                    test_dummy[c] = 0

        for c in test_dummy.columns:
            if c != 'target':
                if c not in train_dummy.columns:
                    train_dummy[c] = 0

    # Setup for LIME
    # Change feature names to the ones defined in the list 'attributes' (above)
    # Needed for computing generality
    feature_names = []
    old_column_names = []
    lime_categorical_features = []
    if train_dummy is not None:
        for col in train_dummy.columns:
            if col != 'target':
                feature_names.append(col)
                old_column_names.append(col)
    else:
        for col in train.columns:
            if col != 'target':
                feature_names.append(col)
                old_column_names.append(col)

    new_column_names_dict = {}
    new_column_names = []

    i = 0

    for col in old_column_names:
        new_column_names_dict[col] = attributes[i]
        new_column_names.append(attributes[i])
        if col not in numerical_features:
            lime_categorical_features.append(attributes[i])
        i += 1

    if train_dummy is not None:
        train_lime = train_dummy.copy(deep=True)
        train_lime.columns = new_column_names + ['target']
        lime_exp = lime_tabular.LimeTabularExplainer(train_dummy[feature_names].values, feature_names=new_column_names,
                                                     class_names=['target'],
                                                     categorical_features=lime_categorical_features, verbose=False,
                                                     mode='regression')
    else:
        train_lime = train.copy(deep=True)
        train_lime.columns = new_column_names + ['target']
        lime_exp = lime_tabular.LimeTabularExplainer(train[feature_names].values, feature_names=new_column_names,
                                                     class_names=['target'],
                                                     categorical_features=lime_categorical_features, verbose=False,
                                                     mode='regression')
    ##############

    # Setup for SHAP
    shap_exp = shap.KernelExplainer(bb_model.predict, X_train_summary)
    ##############
    total_count = 0

    for explain_index in tqdm(explain_indexes):
        exp_point = pd.DataFrame([test.loc[explain_index]])
        explain_point_dummy = None
        if train_dummy is not None:
            explain_point_dummy = pd.DataFrame([test_dummy.loc[explain_index]])
        features = [f_name for f_name in data.columns if f_name != 'target']

        exp_box, exp_model, exp = explain(train, train_dummy, exp_point, explain_point_dummy, binary_features,
                                          categorical_dis, numerical_features, verbose=False)
        if train_dummy is not None:
            bella_results['accuracy'].append((sqrt((explain_point_dummy['target'].values[0] -
                                                    exp_model.predict(
                                                        explain_point_dummy[exp_model.feature_names_in_])[0])**2)))
        else:
            bella_results['accuracy'].append(
                (sqrt((exp_point['target'].values[0] -
                       exp_model.predict(explain_point_dummy[exp_model.feature_names_in_])[0]) ** 2)))

        if train_dummy is not None:
            shap_explanation = shap_exp.shap_values(explain_point_dummy[feature_names], silent=True)
            lime_explanation = lime_exp.explain_instance(test_dummy[feature_names].loc[explain_index].values,
                                                         bb_model.predict,
                                                         num_features=len(exp_model.feature_names_in_))
        else:
            shap_explanation = shap_exp.shap_values(exp_point[feature_names], silent=True)
            lime_explanation = lime_exp.explain_instance(test[feature_names].loc[explain_index].values,
                                                         bb_model.predict,
                                                         num_features=len(exp_model.feature_names_in_))

        features = [f_name for f_name in data.columns if f_name != 'target']
        exp_features = lime_explanation.domain_mapper.exp_feature_names
        exp_lime = lime_explanation.local_exp[0]
        lime_pred = lime_explanation.local_pred[0]
        expl = {}
        for item in exp_lime:
            expl[exp_features[item[0]]] = item[1]
        bella_results['length'].append(len(exp_model.feature_names_in_))
        lime_results['length'].append(len(exp_model.feature_names_in_))
        lime_results['accuracy'].append(sqrt((exp_point['target'].values[0] - lime_pred) ** 2))
        shap_results['accuracy'].append(0.0)
        e = lime_explanation.as_list()
        shap_length = 0
        for item in shap_explanation[0]:
            if item != 0.0:
                shap_length += 1

        shap_results['length'].append(shap_length)
        explanation = []
        weights = []
        if total_count < 20:
            r_box95 = exp_box.head(11)
            r = 0.0
            r_lime = 0.0
            r_count = 0.0
            r_shap = 0.0
            for index, row in r_box95.iterrows():
                r_point_dummy = None
                if index != explain_index:
                    r_point = pd.DataFrame([train.loc[index]])
                    if train_dummy is not None:
                        r_point_dummy = pd.DataFrame([train_dummy.loc[index]])

                    r_exp_box, r_exp_model, r_exp = explain(train, train_dummy, r_point, r_point_dummy, binary_features,
                                                            categorical_dis, numerical_features, verbose=False)

                    if train_dummy is not None:
                        r_lime_explanation = lime_exp.explain_instance(train_dummy[feature_names].loc[index].values,
                                                                       bb_model.predict,
                                                                       num_features=len(exp_model.feature_names_in_))
                        r_shap_explanation = shap_exp.shap_values(r_point_dummy[feature_names], silent=True)

                    else:
                        r_lime_explanation = lime_exp.explain_instance(train[feature_names].loc[index].values,
                                                                       bb_model.predict,
                                                                       num_features=len(exp_model.feature_names_in_))
                        r_shap_explanation = shap_exp.shap_values(r_point[feature_names], silent=True)

                    if r_exp_model:
                        r_exp_features = r_lime_explanation.domain_mapper.exp_feature_names
                        r_exp_lime = r_lime_explanation.local_exp[0]
                        r_expl = {}
                        for item in r_exp_lime:
                            r_expl[r_exp_features[item[0]]] = item[1]
                        r_lime += compute_exp_distance(expl, r_expl, features=new_column_names)
                        r += compute_exp_distance(exp, r_exp, features=feature_names)
                        r_shap += compute_exp_distance_shap(shap_explanation[0], r_shap_explanation[0],
                                                            len(feature_names))
                        r_count += 1.0

            if len(r_box95) >= 2:
                bella_results['robustness'].append(r / r_count)
                lime_results['robustness'].append(r_lime / r_count)
                shap_results['robustness'].append(r_shap/r_count)
            total_count += 1

        for t in e:
            explanation.append(t[0])
            weights.append(t[1])
        conditions = ''
        for cond in explanation:
            new_value = to_code(cond)
            if conditions == '':
                conditions = new_value
            else:
                conditions = conditions + " and " + new_value

        total = compute_confidence(train_lime, conditions)
        bella_results['generality'].append(len(exp_box) - 1)
        shap_results['generality'].append(0.0)
        lime_results['generality'].append(total)

    print("Black box - RF, dataset: {}".format(dataset_name))
    print("BELLA: ", bella_results)
    print("LIME: ", lime_results)
    print("SHAP:", shap_results)
