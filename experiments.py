from lime import lime_tabular
from bella import *
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from utils import *
from MAPLE import *
import shap
import warnings
import numpy as np
warnings.filterwarnings('ignore')


classification = False

datasets = ['servo', 'auto', 'concrete', 'customer_churn', 'real_estate', 'winequality', 'bike', 'cpu', 'echo', 'wind'
            'electrical', 'superconduct']

for dataset_name in datasets:
    data = pd.read_csv("./datasets/" + dataset_name + ".csv", header=0)
    eps = 0.05
    bella_results = {'accuracy': [], 'length': [], 'generality': [], 'robustness': [], 'counterfactual_accuracy': [],
                     'counterfactual_length': []}
    lime_results = {'accuracy': [], 'length': [], 'generality': [], 'robustness': []}
    shap_results = {'accuracy': [], 'length': [], 'generality': [], 'robustness': []}
    maple_results = {'accuracy': [], 'length': [], 'generality': [], 'robustness': []}
     
    attributes = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                  'u', 'v', 'w', 'x', 'y', 'z', 'aa', 'ab', 'ac', 'ad', 'ae', 'af', 'ag', 'ah', 'ai', 'aj', 'ak', 'al',
                  'am', 'an', 'ao', 'ap', 'aq', 'ar', 'as', 'at', 'au', 'av', 'aw', 'ax', 'ay', 'az', 'ba', 'bb', 'bc',
                  'bd', 'be', 'bf', 'bg', 'bh', 'bi', 'bj', 'bk', 'bl', 'bm', 'bn', 'bo', 'bp', 'bq', 'br', 'bs', 'bt',
                  'bu', 'bv', 'bw', 'bx', 'by', 'bz']

    target_max = max(data['target'])
    target_min = min(data['target'])
    thirty_qle = 0.3 * (target_max - target_min)
    features = [f_name for f_name in data.columns if f_name != 'target']
    train_dummy = None
    test_dummy = None
    dummy_features = None
    categorical_dis = None

    ###########################################################################################
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
    ###########################################################################################
    train, test = train_test_split(data, test_size=0.1)

    # Standardize numerical features
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

    # For smaller datasets,
    if train_dummy is not None:
        for c in train_dummy.columns:
            if c != 'target':
                if c not in test_dummy.columns:
                    test_dummy[c] = 0

        for c in test_dummy.columns:
            if c != 'target':
                if c not in train_dummy.columns:
                    train_dummy[c] = 0

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

    ###########################################################################################
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
    ###########################################################################################

    # Setup for SHAP
    shap_exp = shap.KernelExplainer(bb_model.predict, X_train_summary)
    ###########################################################################################
    
    # Setup for MAPLE
    
    if train_dummy is not None:
        train_maple = train_dummy.copy(deep=True)
        train_maple['target'] = y_bb
        X_val = train_maple.sample(frac=0.3)
        MR_val = X_val['target']
        X_val = X_val.drop('target', axis=1)
        train_maple = train_maple.drop('target', axis=1)
        MR_train = y_bb

    else:
        train_maple = train.copy(deep=True)
        train_maple['target'] = y_bb
        X_val = train_maple.sample(frac=0.3)
        MR_val = X_val['target']
        X_val = X_val.drop('target', axis=1)
        train_maple = train_maple.drop('target', axis=1)
        MR_train = y_bb
    maple_explainer = MAPLE(train_maple.values, MR_train, X_val.values, MR_val)

    ###########################################################################################
    
    total_count = 0

    for explain_index in tqdm(explain_indexes):
        exp_point = pd.DataFrame([test.loc[explain_index]])
        if exp_point['target'].values[0] + thirty_qle < target_max:
            ref_target = exp_point['target'].values[0] + thirty_qle
        else:
            ref_target = exp_point['target'].values[0] - thirty_qle
        explain_point_dummy = None
        if train_dummy is not None:
            explain_point_dummy = pd.DataFrame([test_dummy.loc[explain_index]])
        features = [f_name for f_name in data.columns if f_name != 'target']

        exp_box, exp_model, exp = explain(train, exp_point, binary_features, categorical_dis, numerical_features,
                                          explain_point_dummy=explain_point_dummy, train_dummy=train_dummy,
                                          verbose=False)

        c_exp_model, new_data_point, counterfactual = explain(train, exp_point, binary_features, categorical_dis,
                                                              numerical_features,
                                                              explain_point_dummy=explain_point_dummy,
                                                              train_dummy=train_dummy,
                                                              reference_value=ref_target, verbose=False)

        bella_results['counterfactual_length'].append(len(c_exp_model.feature_names_in_))

        if train_dummy is not None:
            bella_results['accuracy'].append((sqrt((explain_point_dummy['target'].values[0] -
                                                    exp_model.predict(
                                                        explain_point_dummy[exp_model.feature_names_in_])[0])**2)))
            bella_results['counterfactual_accuracy'].append(sqrt((counterfactual['target'] -
                                                                  bb_model.predict(
                                                                      new_data_point[old_column_names])[0]) ** 2))
        else:
            bella_results['accuracy'].append(
                (sqrt((exp_point['target'].values[0] -
                       exp_model.predict(exp_point[exp_model.feature_names_in_])[0]) ** 2)))
            bella_results['counterfactual_accuracy'].append(sqrt((counterfactual['target'] -
                                                                  bb_model.predict(new_data_point[features])[0]) ** 2))

        if train_dummy is not None:
            shap_explanation = shap_exp.shap_values(explain_point_dummy[feature_names], silent=True)
            lime_explanation = lime_exp.explain_instance(test_dummy[feature_names].loc[explain_index].values,
                                                         bb_model.predict,
                                                         num_features=len(exp_model.feature_names_in_))
            maple_explanation = maple_explainer.explain(test_dummy[feature_names].loc[explain_index].values)
                                                              
        else:
            shap_explanation = shap_exp.shap_values(exp_point[feature_names], silent=True)
            lime_explanation = lime_exp.explain_instance(test[feature_names].loc[explain_index].values,
                                                         bb_model.predict,
                                                         num_features=len(exp_model.feature_names_in_))
            maple_explanation = maple_explainer.explain(test[feature_names].loc[explain_index].values)

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
        maple_results['accuracy'].append(sqrt((exp_point['target'].values[0] - maple_explanation['pred'][0])**2))
        expl_maple = {}
        for i in range(len(train_maple.columns)):
            expl_maple[train_maple.columns[i]] = maple_explanation['coefs'][i + 1]
        maple_exp_features = maple_explanation['coefs'][1:]
        exp_len = 0
        for item in maple_exp_features:
            if item != 0.0:
                exp_len += 1
        maple_results['length'].append(exp_len)
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
            r_maple = 0.0
            for index, row in r_box95.iterrows():
                r_point_dummy = None
                if index != explain_index:
                    r_point = pd.DataFrame([train.loc[index]])
                    if train_dummy is not None:
                        r_point_dummy = pd.DataFrame([train_dummy.loc[index]])

                    r_exp_box, r_exp_model, r_exp = explain(train, r_point, binary_features, categorical_dis,
                                                            numerical_features, train_dummy=train_dummy,
                                                            explain_point_dummy=r_point_dummy, verbose=False)

                    if train_dummy is not None:
                        r_lime_explanation = lime_exp.explain_instance(train_dummy[feature_names].loc[index].values,
                                                                       bb_model.predict,
                                                                       num_features=len(exp_model.feature_names_in_))
                        r_shap_explanation = shap_exp.shap_values(r_point_dummy[feature_names], silent=True)
                        r_maple_explanation = maple_explainer.explain(train_dummy[train_maple.columns].loc[index].values)

                    else:
                        r_lime_explanation = lime_exp.explain_instance(train[feature_names].loc[index].values,
                                                                       bb_model.predict,
                                                                       num_features=len(exp_model.feature_names_in_))
                        r_shap_explanation = shap_exp.shap_values(r_point[feature_names], silent=True)
                        r_maple_explanation = maple_explainer.explain(train[train_maple.columns].loc[index].values)

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
                        r_expl_maple = {}
                        for i in range(len(train_maple.columns)):
                            r_expl_maple[train_maple.columns[i]] = r_maple_explanation['coefs'][i + 1]
                        r_maple += compute_exp_distance(expl_maple, r_expl_maple, features=train_maple.columns)
                        r_count += 1.0

            if len(r_box95) >= 2:
                bella_results['robustness'].append(r / r_count)
                lime_results['robustness'].append(r_lime / r_count)
                shap_results['robustness'].append(r_shap/r_count)
                maple_results['robustness'].append(r_maple / r_count)
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
        maple_results['generality'].append(np.count_nonzero(maple_explanation['weights']))

    print("Black box - RF, dataset: {}".format(dataset_name))
    print("BELLA: ", bella_results)
    for key, value in bella_results.items():
        print("{}: {:.2f}\n".format(key, mean(value)))
    print("LIME: ", lime_results)
    for key, value in lime_results.items():
        print("{}: {:.2f}\n".format(key, mean(value)))
    print("SHAP:", shap_results)
    for key, value in shap_results.items():
        print("{}: {:.2f}\n".format(key, mean(value)))
    print("MAPLE:", maple_results)
    for key, value in maple_results.items():
        print("{}: {:.2f}\n".format(key, mean(value)))
