import pandas as pd
from preprocessing import preprocess, postprocess, fill_missing
from model import get_ae, get_vae
from generator import generate_data_ae, generate_data_vae
import evaluation
import data_read
import json
from timeit import default_timer as timer
from sklearn.model_selection import KFold
from keras import backend as K


# values for grid search:
# cutting = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# epochs = [5, 10, 20, 50, 100]
# batches = [16, 32, 64, 128, 256]
# drop_rates = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50]
# activations = ['tanh', 'relu']

datasets = data_read.datasets
i = 0
n = 5

results_json = []
for dataset in datasets:
    for j in range(n):
        data = pd.read_csv('./01_data/' + dataset)
        data = data.drop(data.columns[0], axis=1)
        if data_read.DROP in datasets[dataset].keys():
            data = data.drop(datasets[dataset][data_read.DROP], axis=1)
        class_column = datasets[dataset][data_read.CLASS]

        data = fill_missing(data)
        data = data.sample(frac=1).reset_index(drop=True)
        preprocessed_data, columns, numerical_cols, norm_map, nominal_cols = preprocess(data)

        splits = 10
        kf = KFold(n_splits=splits)
        for train_index, test_index in kf.split(preprocessed_data):
            results = \
                {
                    'vae': {
                        'time': 0,
                        'statistical': {'m(mean)': 0, 'm(std)': 0, 'm(skew)': 0, 'm(kurtosis)': 0},
                        'clustering': {'ARI': 0},
                        'classification': {'m1d1': 0, 'm2d1': 0}
                    },
                    'ae': {
                        'time': 0,
                        'statistical': {'m(mean)': 0, 'm(std)': 0, 'm(skew)': 0, 'm(kurtosis)': 0},
                        'clustering': {'ARI': 0},
                        'classification': {'m1d1': 0, 'm2d1': 0}
                    }
                }
            pr_dat_train = preprocessed_data.iloc[train_index]
            pr_dat_test = preprocessed_data.iloc[test_index]
            dat_train = data.iloc[train_index]
            dat_test = data.iloc[test_index]

            # vae gen
            start = timer()
            gen, enc = get_vae(pr_dat_train)
            new_data = generate_data_vae(generator=gen, encoder=enc, org_data=pr_dat_train, n=1000)
            end = timer()
            time_vae = end - start
            results['vae']['time'] += time_vae

            stats_eval = evaluation.statistical(pr_dat_train, new_data, numerical_cols=numerical_cols)
            results['vae']['statistical']['m(mean)'] += stats_eval[0]
            results['vae']['statistical']['m(std)'] += stats_eval[1]
            results['vae']['statistical']['m(skew)'] += stats_eval[2]
            results['vae']['statistical']['m(kurtosis)'] += stats_eval[3]
            results['vae']['clustering']['ARI'] += evaluation.clustering(pr_dat_train, new_data, pr_dat_test)

            new_data = postprocess(new_data, numerical_cols, norm_map, nominal_cols)
            class_sc = evaluation.classification(dat_train, new_data, dat_test, class_column=class_column)
            results['vae']['classification']['m1d1'] += class_sc[0] * 100
            results['vae']['classification']['m2d1'] += class_sc[1] * 100
            ###

            # ae gen
            start = timer()
            gen, enc = get_ae(pr_dat_train)
            new_data_2 = generate_data_ae(generator=gen, encoder=enc, org_data=pr_dat_train, n=1000)
            end = timer()
            time_ae = end - start
            results['ae']['time'] += time_ae

            stats_eval = evaluation.statistical(pr_dat_train, new_data_2, numerical_cols=numerical_cols)
            results['ae']['statistical']['m(mean)'] += stats_eval[0]
            results['ae']['statistical']['m(std)'] += stats_eval[1]
            results['ae']['statistical']['m(skew)'] += stats_eval[2]
            results['ae']['statistical']['m(kurtosis)'] += stats_eval[3]
            results['ae']['clustering']['ARI'] += evaluation.clustering(pr_dat_train, new_data_2, pr_dat_train)

            new_data_2 = postprocess(new_data_2, numerical_cols, norm_map, nominal_cols)
            class_sc = evaluation.classification(dat_train, new_data_2, dat_test, class_column=class_column)
            results['ae']['classification']['m1d1'] += class_sc[0] * 100
            results['ae']['classification']['m2d1'] += class_sc[1] * 100
            print(results)
            ###
            K.clear_session()
            results_json.append({
                'dataset': dataset,
                'shape': {
                    'rows': data.shape[0],
                    'columns': data.shape[1]
                },
                'results_vae': results['vae'],
                'results_ae': results['ae']
            })

    i = i+1

json_name = 'results.json'
with open(json_name, 'w', encoding='utf-8') as outfile:
    json.dump(results_json, outfile, indent=4)
