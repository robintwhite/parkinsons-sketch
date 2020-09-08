from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import copy

def create_interactions(ndata):
    data_old = copy.deepcopy(ndata)
    data = copy.deepcopy(ndata)
    s = '_'
    j = data.shape[1] - 1
    for column_i in data_old:
        x = data[column_i]
        for column_j in data_old:
            if column_j != column_i and s.join((column_j,column_i)) not in data.columns:
                y = data[column_j]
                inter_terms = np.multiply(x,y)
                data.insert(j+1, s.join((column_i,column_j)), inter_terms)
                j += 1 #column index
    return data


def dummy(ndata):
    # create boolean variables for discrete variables
    data = copy.deepcopy(ndata)
    s = '_'
    b = []
    j = 0
    tmp = np.zeros(len(data))
    for column in data:
        # create unqie variable for discrete catagories
        if data[column].dtype == 'O':
            a = data[column].unique()
            for i in range(a.size):
                k = 0
                for row in data.iterrows():
                    if row[1][column] == a[i]:
                        tmp[k] = int(1)
                    else:
                        tmp[k] = int(0)
                    k += 1  # row index
                data.insert(j + 1, s.join((column, a[i])), tmp)
                b.append(s.join((column, a[i])))
            data = data.drop([column], axis=1)

        j += 1  # column index

    return data, b


def standardize(ndata_train, ndata_test, verbose = True):
    # standardize continuous data
    scaler = StandardScaler()
    train_data = copy.deepcopy(ndata_train)
    test_data = copy.deepcopy(ndata_test)
    for column in train_data:
        # create unique variable for discrete catagories
        x_train = train_data[column].values.astype(np.float64)
        x_train = x_train.reshape(-1, 1)

        x_test = test_data[column].values.astype(np.float64)
        x_test = x_test.reshape(-1, 1)

        train_data[column] = scaler.fit_transform(x_train)
        test_data[column] = scaler.transform(x_test)

        if verbose:
            print(f'{column}, mean: {scaler.mean_}, var: {scaler.var_}')
    return train_data, test_data


def feature_importance(model):
    # from https://stackoverflow.com/questions/42128545/how-to-print-the-order-of-important-features-in-random-forest-regression-using-p
    important_features_dict = {}
    for x, i in enumerate(model.feature_importances_):
        important_features_dict[x] = i

    important_features_list = sorted(important_features_dict,
                                     key=important_features_dict.get,
                                     reverse=True)
    return important_features_list

