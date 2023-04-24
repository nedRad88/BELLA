import re
from sklearn.metrics import mean_squared_error


def compute_confidence(x_train, path):
    total = 0
    for i in range(len(x_train)):
        d = x_train.iloc[i]
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
