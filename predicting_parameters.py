import json
import math
import pandas as pd
import scipy.stats as sc
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import data_read
import preprocessing


def parameter_classifier_grid_bassed(file, datasets):
    with open(file) as json_data:
        d = json.load(json_data)
    ds_list = list(set([x['dataset'] for x in d]))
    ds_list = sorted(ds_list, key=lambda x: x.lower())
    grouped = [[x for x in d if x['dataset'] == ds] for ds in ds_list]

    attr_types = []
    no_classes = []
    no_cases = []
    no_attrs = []
    entropies = []
    acts = []
    epochs = []
    drops = []
    batchs = []
    cuts = []
    ae_param_str = []
    vae_param_str = []

    for key in datasets:
        dataset = key

        data = pd.read_csv('./01_data/' + dataset)
        if data_read.DROP in datasets[dataset].keys():
            data = data.drop(datasets[dataset][data_read.DROP], axis=1)

        cases = data.shape[0]
        attrs = data.shape[1]

        numerical_cols = list(data._get_numeric_data().columns)
        nominal_cols = list(set(list(data.columns)) - set(numerical_cols))
        attr_type = -1 if len(nominal_cols) == 0 else (1 if len(numerical_cols) == 0 else 0)

        class_count = data[datasets[dataset][data_read.CLASS]].value_counts(normalize=True)
        entropy = sc.entropy(class_count)/math.log(class_count.size, 2)
        no_of_classes = class_count.size

        dataset_group = [x for group in grouped for x in group if x['dataset'] == dataset]

        i = 1
        for ae_case in dataset_group:
            ae_param_str.append(abs(ae_case['results_ae']['classification']['m1d1'] -
                                    ae_case['results_ae']['classification']['m2d1']))
            vae_param_str.append(abs(ae_case['results_vae']['classification']['m1d1'] -
                                     ae_case['results_vae']['classification']['m2d1']))

            attr_types.append(attr_type)
            no_classes.append(no_of_classes)
            no_cases.append(cases)
            no_attrs.append(attrs)
            entropies.append(entropy)

            acts.append(ae_case['parameters']['activation'])
            cuts.append(ae_case['parameters']['cutting'])
            batchs.append(ae_case['parameters']['batch'])
            drops.append(ae_case['parameters']['drop'])
            epochs.append(ae_case['parameters']['epoch'])
            i = i + 1

    fin_df = pd.DataFrame({'cases': no_cases,
                           'attributes': no_attrs,
                           'classes': no_classes,
                           'entropy': entropies,
                           'batches': batchs,
                           'activations': acts,
                           'drops': drops,
                           'epochs': epochs,
                           'cutting': cuts,
                           'ae_param': ae_param_str,
                           'vae_param': vae_param_str
                           })

    X = pd.get_dummies(fin_df.drop(['ae_param', 'vae_param'], axis=1)).values

    clf_ae = RandomForestRegressor()
    clf_ae.fit(X, fin_df['ae_param'].values)

    clf_vae = RandomForestRegressor()
    clf_vae.fit(X, fin_df['vae_param'].values)

    # predict for some datasets
    cutting = [0.1, 0.25, 0.5, 0.75]
    epochs = [20, 50, 100]
    batches = [16, 64, 256]
    drop_rates = [0.0, 0.20, 0.40]
    activations = ['tanh', 'relu']

    param_combs = [[x, y, z, w, u]
                   for x in batches
                   for y in activations
                   for z in drop_rates
                   for w in epochs
                   for u in cutting]
    combs_df = pd.DataFrame(param_combs)
    combs_df.columns = ['batches', 'activations', 'drops', 'epochs', 'cutting']
    columns = ['cases', 'attributes', 'classes', 'entropy', 'batches', 'activations', 'drops', 'epochs', 'cutting']

    datasets = data_read.datasets
    for dataset in datasets:
        data = pd.read_csv('./01_data/' + dataset)
        data = data.drop(data.columns[0], axis=1)
        if data_read.DROP in datasets[dataset].keys():
            data = data.drop(datasets[dataset][data_read.DROP], axis=1)

        data = preprocessing.fill_missing(data)
        cases = data.shape[0]
        attrs = data.shape[1]
        class_count = data[datasets[dataset][data_read.CLASS]].value_counts(normalize=True)
        entropy = sc.entropy(class_count) / math.log(class_count.size, 2)
        no_of_classes = class_count.size

        ds_df = combs_df.copy()
        ds_df['cases'] = cases
        ds_df['attributes'] = attrs
        ds_df['classes'] = no_of_classes
        ds_df['entropy'] = entropy
        ds_df = ds_df[columns]

        ae_param = clf_ae.predict(pd.get_dummies(ds_df).values)
        ae_pred_row = ds_df.iloc[np.argmin(ae_param)]
        print(ae_pred_row)
        vae_param = clf_vae.predict(pd.get_dummies(ds_df).values)
        vae_pred_row = ds_df.iloc[np.argmin(vae_param)]
        print(vae_pred_row)
