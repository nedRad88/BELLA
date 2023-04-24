from statistics import mean
from math import sqrt
import itertools
import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np
import scipy.stats
from sklearn.metrics.pairwise import manhattan_distances
import pandas as pd
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error


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
    Computing distance for categorical features.
    :param data_train:
    :param categorical_features:
    :param num_features:
    :return:
    """
    cat_distances = {}
    data = data_train.copy(deep=True)
    if len(num_features) > 0:
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
    """
    Compute the distance for categorical attributes between a given data point and the training data points
    :param dict_of_dist:
    :param point:
    :param train_set:
    :return:
    """
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
                distance += dict_of_dist[c][(value1, value2)]
        cat_distance.loc[index] = distance

    return cat_distance


def calculate_vif(df, atts):
    """
    Compute VIF to exclude the multi-collinear features
    :param df:
    :param atts:
    :return:
    """
    stop = False
    atts2 = [item for item in atts]
    df = df.drop('target', axis=1)
    while len(atts2) > 1 and stop is False:
        max_vif = 0.0
        worst_feature = None
        for feature in atts2:
            x = [f for f in atts2 if f != feature]
            x, y = df[x], df[feature]
            r2 = linear_model.LinearRegression().fit(x, y).score(x, y)
            if r2 == 1:
                vif = 100.0
            else:
                vif = 1 / (1 - r2)
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
    """
    Compute lower bound and margin of error for R value.
    :param true_values:
    :param predicted:
    :return:
    """
    delta_values = np.square(np.subtract(predicted, true_values))
    all_mu = np.square(predicted[None, :] - true_values[:, None])
    mu_values = all_mu.mean(axis=1)

    delta_mean, delta_var = compute_stats(delta_values)
    mu_mean = mean(mu_values)

    moe_delta95 = scipy.stats.t.ppf(q=1 - .05 / 2, df=len(true_values) - 1) * sqrt(delta_var) / sqrt(len(true_values))

    return 1 - min(1, (abs(delta_mean + moe_delta95)) / (abs(mu_mean)))


@ignore_warnings(category=ConvergenceWarning)
def train_lin_model(data, atts):
    """
    trains a local linear model to explain given prediction.
    :param data:
    :param atts:
    :return:
    """
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
        return 0.0, None, 0.0
    lmodel = linear_model.LinearRegression()
    cv_splitter = KFold()
    predicted = cross_val_predict(lmodel, data[new_atts], data['target'], cv=cv_splitter)
    lmodel = linear_model.LinearRegression()
    lmodel.fit(data[new_atts], data['target'])
    lb95 = margin_of_error(data['target'].values, predicted)
    mse = mean_squared_error(data['target'].tolist(), lmodel.predict(data[new_atts]))
    return lb95, lmodel, mse


def show_explanation(explanation, expl_model, data_point):
    """
    Visualize the BELLA explanation.
    :param explanation:
    :param expl_model:
    :param data_point:
    :return:
    """
    exp_features = list(explanation.keys())
    importance = list(explanation.values())
    coeffs = []
    for item in exp_features:
        for i in range(len(expl_model.feature_names_in_)):
            if expl_model.feature_names_in_[i] == item:
                coeffs.append(expl_model.coef_[i])
    positives = []
    negatives = []
    neg_sum = 0.0
    poz_sum = 0.0

    colors = ['tab:blue'] * len(exp_features)
    for i in range(len(exp_features)):
        if importance[i] < 0.0:
            negatives.append((exp_features[i], importance[i], coeffs[i]))
            colors[i] = 'tab:red'
            neg_sum += importance[i]
        else:
            positives.append((exp_features[i], importance[i], coeffs[i]))
            poz_sum += importance[i]

    y_max = max(expl_model.intercept_, expl_model.intercept_ + poz_sum)
    y_min = min(expl_model.intercept_, expl_model.intercept_ + poz_sum + neg_sum)

    y_range = y_max - y_min

    prev_bar = expl_model.intercept_
    plt.figure()
    plt.hlines(y=expl_model.intercept_, xmin=-0.1, xmax=1, linewidth=1, linestyles='--', color='k', alpha=0.4)
    plt.annotate("Base value: {:0,.2f}".format(expl_model.intercept_), (0.1, prev_bar - 0.03 * prev_bar), ha='left',
                 va='top',
                 size=10, xytext=(0, -1),
                 textcoords='offset points')

    for item in positives:
        if abs(item[1]) > 0.05 * y_range:
            plt.arrow(x=1, y=prev_bar, dx=0, dy=item[1], lw=0, length_includes_head=True, head_width=0.3,
                      width=0.3, head_length=0.05 * y_range, facecolor="tab:blue")
        else:
            plt.bar(1, item[1], align='center', color="tab:blue", bottom=prev_bar, width=0.3, linewidth=1,
                    edgecolor='w')
        prev_bar += item[1]

    plt.hlines(y=prev_bar, xmin=1, xmax=1.5, linewidth=1, linestyles='--', color='k', alpha=0.4)

    for item in negatives:
        if abs(item[1]) > 0.05 * y_range:
            plt.arrow(x=1.5, y=prev_bar, dx=0, dy=item[1], lw=0, length_includes_head=True, head_width=0.3,
                      width=0.3, head_length=0.05 * y_range, facecolor="tab:red")
        else:
            plt.bar(1.5, item[1], align='center', color="tab:red", bottom=prev_bar, width=0.3, linewidth=1,
                    edgecolor='w')
        prev_bar += item[1]

    prev_bar = expl_model.intercept_
    pos_offset = 0
    for item in positives:
        if abs(item[1]) > 0.05 * y_range:
            plt.annotate(" {:0,.2f} x {} ".format(item[2], item[0]), (0.85, prev_bar + item[1] / 2), ha='right',
                         va='center',
                         size=10, xytext=(0, 0),
                         textcoords='offset points')
        else:
            plt.annotate(" {:0,.2f} x {}  ".format(item[2], item[0]), (0.85, prev_bar + item[1] / 2), ha='right',
                         va='center',
                         size=10, xytext=(0, pos_offset),
                         textcoords='offset points')
            pos_offset += 10
        prev_bar += item[1]

    neg_offset = 0
    for item in negatives:
        if abs(item[1]) > 0.05 * y_range:
            plt.annotate(" {:0,.2f} x {} ".format(item[2], item[0]), (1.65, prev_bar + item[1] / 2), ha='left',
                         va='bottom',
                         size=10, xytext=(0, 0),
                         textcoords='offset points')
        else:
            plt.annotate(" {:0,.2f} x {} ".format(item[2], item[0]), (1.65, prev_bar + item[1] / 2), ha='left',
                         va='bottom',
                         size=10, xytext=(0, neg_offset),
                         textcoords='offset points')
            neg_offset -= 10

        prev_bar += item[1]
    plt.hlines(y=prev_bar, xmin=1.5, xmax=2.4, linewidth=1, linestyles='--', color='k', alpha=0.4)
    plt.annotate(" Explained value: {:0,.2f}".format(prev_bar), (1.8, prev_bar), ha='center', va='top', size=10,
                 xytext=(3, neg_offset),
                 textcoords='offset points')
    plt.xlim(-0.1, 2.4)
    plt.xticks([1, 1.5], ['increase', 'decrease'])
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off
    plt.tick_params(labelright=True)
    plt.ylim(y_min - 0.17 * y_range, y_max + 0.17 * y_range)
    plt.savefig('bella.png', bbox_inches='tight')
    # plt.show()
    plt.close()


def counterfactual_explanation(train_data, train_data_dummy, exp_point, exp_point_dummy, binary_features,
                               categorical_dis, numerical_features, ref_target, eps):
    # Find potential counterfactual candidates
    stop = False
    while not stop:
        if train_data_dummy is not None:
            potential_refs = train_data_dummy[(ref_target - ref_target * eps <= train_data_dummy['target']) &
                                              (train_data_dummy['target'] <= ref_target + eps * ref_target)]
        else:
            potential_refs = train_data[(ref_target - ref_target * eps <= train_data['target']) &
                                        (train_data['target'] <= ref_target + eps * ref_target)]
        # If there are no candidates for current epsilon, increase epsilon
        if len(potential_refs) == 0:
            eps += 0.05
            stop = False
        else:
            stop = True

    top_k_potential = {}
    # Sort counterfactual candidates by distance
    for index, row in potential_refs.iterrows():
        candidate = pd.DataFrame([train_data.loc[index]])
        if categorical_dis is not None:
            dist = compute_distances_cat(categorical_dis, exp_point, candidate).values[0]
            if len(numerical_features) > 0:
                dist += manhattan_distances(exp_point[numerical_features + binary_features],
                                            candidate[numerical_features + binary_features])[0][0]
        else:
            dist = manhattan_distances(exp_point[numerical_features + binary_features],
                                       candidate[numerical_features + binary_features])[0][0]
        top_k_potential[index] = dist
    top_k_potential = dict(sorted(top_k_potential.items(), key=lambda x: abs(x[1]), reverse=True)[:5])
    potential_refs = train_data.loc[list(top_k_potential.keys())]

    ref_models = {}
    dist_counter_min = 10000000000.0
    best_counterfactual_index = None

    # For each candidate use BELLA to compute the local model
    for ref_i, ref_row in potential_refs.iterrows():
        ref_exp_point = pd.DataFrame([potential_refs.loc[ref_i]])
        if train_data_dummy is not None:
            exp_point_copy = exp_point_dummy.copy(deep=True)
            ref_exp_point_dummy = pd.DataFrame([train_data_dummy.loc[ref_i]])
        else:
            exp_point_copy = exp_point.copy(deep=True)
            ref_exp_point_dummy = None

        exp_box, exp_model, exp = explain(train_data, ref_exp_point, binary_features, categorical_dis,
                                          numerical_features, train_dummy=train_data_dummy,
                                          explain_point_dummy=ref_exp_point_dummy, verbose=False)
        # Apply the changes to the original data point
        for feature in exp_model.feature_names_in_:
            if train_data_dummy is not None:
                exp_point_copy.iloc[0][feature] = ref_exp_point_dummy[feature]
            else:
                exp_point_copy.iloc[0][feature] = ref_exp_point[feature]
        ref_models[ref_i] = exp_model
        # Compute the distance metric to choose the best counterfactual explanation
        # Equation 6 in the paper
        if len(exp_model.feature_names_in_) > 1:
            if categorical_dis is not None:
                dist_counter = compute_distances_cat(categorical_dis, ref_exp_point, exp_point).values[0]
                if len(numerical_features) > 0:
                    dist_counter += manhattan_distances([ref_row[numerical_features + binary_features]],
                                                        [exp_point[numerical_features +
                                                                   binary_features].squeeze()])[0][0]
                dist_counter += manhattan_distances(ref_exp_point_dummy[exp_model.feature_names_in_],
                                                    exp_point_copy[exp_model.feature_names_in_]) / \
                                len(exp_model.feature_names_in_)
            else:
                dist_counter = manhattan_distances([ref_row[numerical_features + binary_features]],
                                                   [exp_point[numerical_features + binary_features].squeeze()])[0][0]
                dist_counter += manhattan_distances(ref_exp_point[exp_model.feature_names_in_],
                                                    exp_point_copy[exp_model.feature_names_in_]) / \
                                len(exp_model.feature_names_in_)

        else:
            if categorical_dis is not None:
                dist_counter = compute_distances_cat(categorical_dis, ref_exp_point, exp_point).values[0]
                if len(numerical_features) > 0:
                    dist_counter += manhattan_distances([ref_row[numerical_features + binary_features]],
                                                        [exp_point[numerical_features +
                                                                   binary_features].squeeze()])[0][0]
                dist_counter += abs(ref_exp_point_dummy[exp_model.feature_names_in_].values -
                                    exp_point_copy[exp_model.feature_names_in_].values)
            else:
                dist_counter = manhattan_distances([ref_row[numerical_features + binary_features]],
                                                   [exp_point[numerical_features + binary_features].squeeze()])[0][0]
                dist_counter += abs(ref_exp_point[exp_model.feature_names_in_].values -
                                    exp_point_copy[exp_model.feature_names_in_].values)

        if dist_counter <= dist_counter_min:
            dist_counter_min = dist_counter
            best_counterfactual_index = ref_i
    # Output the updated data point using the best counterfactual
    if train_data_dummy is not None:
        exp_point_copy = exp_point_dummy.copy(deep=True)
        for feature in ref_models[best_counterfactual_index].feature_names_in_:
            exp_point_copy.iloc[0][feature] = train_data_dummy.loc[best_counterfactual_index][feature]
    else:
        exp_point_copy = exp_point.copy(deep=True)
        for feature in ref_models[best_counterfactual_index].feature_names_in_:
            exp_point_copy.iloc[0][feature] = potential_refs.loc[best_counterfactual_index][feature]

    # counter_explanation = {}
    # for f in features:
    # val = exp_point_copy[f].values[0] - exp_point[f].values[0]
    # if val != 0.0:
    # counter_explanation[f] = val

    return ref_models[best_counterfactual_index], exp_point_copy, potential_refs.loc[best_counterfactual_index]


def explain(train, explain_point, bin_fs, cat_dist, num_fs, train_dummy=None, explain_point_dummy=None,
            reference_value=None, epsilon=0.05, verbose=False):
    if reference_value:
        return counterfactual_explanation(train, train_dummy, explain_point, explain_point_dummy, bin_fs, cat_dist,
                                          num_fs, reference_value, epsilon)
    else:
        df = train.copy(deep=True)
        if train_dummy is not None:
            data_dist = train_dummy.copy(deep=True)
        else:
            data_dist = df.copy(deep=True)

        e_index = explain_point.index[0]
        if e_index not in train.index:
            df = pd.concat([pd.DataFrame(explain_point), df])
            if train_dummy is not None:
                data_dist = pd.concat([pd.DataFrame(explain_point_dummy), data_dist])
            else:
                data_dist = pd.concat([pd.DataFrame(explain_point), data_dist])

        # Compute distances between current data point and the rest of data in the training set
        if cat_dist is not None:
            data_dist['distance_metric'] = compute_distances_cat(cat_dist, explain_point, df)
            if len(num_fs) > 0:
                data_dist['distance_metric'] += manhattan_distances(explain_point[num_fs + bin_fs],
                                                                    data_dist[num_fs + bin_fs])[0]
        else:
            data_dist['distance_metric'] = manhattan_distances(explain_point[num_fs + bin_fs],
                                                               data_dist[num_fs + bin_fs])[0]

        # Sort by target diff and distance_metric
        data_dist = data_dist.sort_values(['distance_metric'], ascending=[True])
        data_dist = data_dist.drop('distance_metric', axis=1)
        fs = [c for c in data_dist.columns if c != 'target']
        outside_box = data_dist.copy(deep=True)
        inside_box = pd.DataFrame([data_dist.loc[explain_point.index[0]]])
        outside_box = outside_box.drop([explain_point.index[0]])
        data_dist = data_dist.drop([explain_point.index[0]])

        # Initialization
        best_lb = 0.0
        explain_box = 0
        best_model = None

        # Optimal neighbourhood search
        for i, g in data_dist.groupby(np.arange(len(data_dist)) // (round(len(data_dist) / 100))):
            inside_box = pd.concat([inside_box, g])
            outside_box = outside_box.drop(list(g.index))
            if len(inside_box) >= max(10, 2 * len(fs)):
                # Train linear model
                lb, model, me = train_lin_model(inside_box, fs)
            else:
                model = None
                lb = 0.0
            if lb > best_lb:
                best_lb = lb
                best_model = model
                explain_box = inside_box.copy(deep=True)

        # Form and visualize explanation
        explanation = {}
        e_point = df.loc[e_index]
        if best_model:
            for i in range(len(best_model.feature_names_in_)):
                if abs(best_model.coef_[i]) > 0.0:
                    explanation[best_model.feature_names_in_[i]] = best_model.coef_[i]

        if verbose:
            show_explanation(explanation, best_model, e_point)

        return explain_box, best_model, explanation
