import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.ensemble import RandomForestRegressor


def statistical(org_data, new_data, numerical_cols):
    m_mean = abs(new_data[numerical_cols].mean() - org_data[numerical_cols].mean()).median()
    m_std = abs(new_data[numerical_cols].std() - org_data[numerical_cols].std()).median()
    m_skew = abs(new_data[numerical_cols].skew() - org_data[numerical_cols].skew()).median()
    m_kurtosis = abs(new_data[numerical_cols].kurtosis() - org_data[numerical_cols].kurtosis()).median()

    return m_mean, m_std, m_skew, m_kurtosis


def clustering(org_data, new_data, test_data):
    if test_data.shape[1] > org_data.shape[1]:
        missing_attr = list(set(test_data) - set(org_data))
        for attr in missing_attr:
            org_data[attr] = 0
        org_data = org_data[list(test_data.columns)]
    elif test_data.shape[1] < org_data.shape[1]:
        missing_attr = list(set(org_data) - set(test_data))
        for attr in missing_attr:
            test_data[attr] = 0
        test_data = test_data[list(org_data.columns)]
    if new_data.shape[1] != org_data.shape[1]:
        missing_attr = list(set(org_data) - set(new_data))
        for attr in missing_attr:
            new_data[attr] = 0
        new_data = new_data[list(org_data.columns)]

    iterations = 10

    ARI = 0
    for _ in range(iterations):
        kmeans_new = KMeans().fit(new_data).predict(test_data)
        kmeans_org = KMeans().fit(org_data).predict(test_data)
        ARI += adjusted_rand_score(kmeans_org, kmeans_new)/iterations
    return ARI


def classification(org_data, new_data, test_data, class_column):
    org_X_train = pd.get_dummies(org_data.drop(class_column, axis=1))
    org_Y_train = org_data[class_column].values

    org_X_test = pd.get_dummies(test_data.drop(class_column, axis=1))
    org_Y_test = test_data[class_column].values

    gen_X = pd.get_dummies(new_data.drop(class_column, axis=1))
    gen_Y = new_data[class_column].values

    if org_X_test.shape[1] > org_X_train.shape[1]:
        missing_attr = list(set(org_X_test) - set(org_X_train))
        for attr in missing_attr:
            org_X_train[attr] = 0
        org_X_train = org_X_train[list(org_X_test.columns)]
    elif org_X_test.shape[1] < org_X_train.shape[1]:
        missing_attr = list(set(org_X_train) - set(org_X_test))
        for attr in missing_attr:
            org_X_test[attr] = 0
        org_X_test = org_X_test[list(org_X_train.columns)]
    if gen_X.shape[1] != org_X_train.shape[1]:
        missing_attr = list(set(org_X_train) - set(gen_X))
        for attr in missing_attr:
            gen_X[attr] = 0
        gen_X = gen_X[list(org_X_train.columns)]

    org_X_test = org_X_test.values
    org_X_train = org_X_train.values
    gen_X = gen_X.values

    clf_org = RandomForestClassifier()
    clf_gen = RandomForestClassifier()

    clf_org.fit(org_X_train, org_Y_train)
    clf_gen.fit(gen_X, gen_Y)

    m1d1 = clf_org.score(org_X_test, org_Y_test)
    m2d1 = clf_gen.score(org_X_test, org_Y_test)

    return m1d1, m2d1


def regression(org_data, new_data, test_data, class_column):
    org_X_train = pd.get_dummies(org_data.drop(class_column, axis=1))
    org_Y_train = org_data[class_column].values

    org_X_test = pd.get_dummies(test_data.drop(class_column, axis=1))
    org_Y_test = test_data[class_column].values

    gen_X = pd.get_dummies(new_data.drop(class_column, axis=1))
    gen_Y = new_data[class_column].values

    if org_X_test.shape[1] > org_X_train.shape[1]:
        missing_attr = list(set(org_X_test) - set(org_X_train))
        for attr in missing_attr:
            org_X_train[attr] = 0
        org_X_train = org_X_train[list(org_X_test.columns)]
    elif org_X_test.shape[1] < org_X_train.shape[1]:
        missing_attr = list(set(org_X_train) - set(org_X_test))
        for attr in missing_attr:
            org_X_test[attr] = 0
        org_X_test = org_X_test[list(org_X_train.columns)]
    if gen_X.shape[1] != org_X_train.shape[1]:
        missing_attr = list(set(org_X_train) - set(gen_X))
        for attr in missing_attr:
            gen_X[attr] = 0
        gen_X = gen_X[list(org_X_train.columns)]

    org_X_test = org_X_test.values
    org_X_train = org_X_train.values
    gen_X = gen_X.values

    clf_org = RandomForestRegressor()
    clf_gen = RandomForestRegressor()

    clf_org.fit(org_X_train, org_Y_train)
    clf_gen.fit(gen_X, gen_Y)

    m1d1 = clf_org.score(org_X_test, org_Y_test)
    m2d1 = clf_gen.score(org_X_test, org_Y_test)

    return m1d1, m2d1
