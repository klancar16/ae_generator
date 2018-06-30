import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity


def generate_data_ae(generator, encoder, org_data, n):
    encoded = encoder.predict(org_data)
    input_vectors_list = []

    kde = KernelDensity().fit(encoded)
    new_col = kde.sample(n)
    input_vectors_list.append(new_col)

    input_data = np.column_stack(input_vectors_list)

    generated_data = generator.predict(input_data)
    new_data = pd.DataFrame(data=generated_data, columns=list(org_data.columns))
    return new_data


def generate_data_vae(generator, encoder, org_data, n):
    z_mean, z_var, z = encoder.predict(org_data.values)
    z_mean = np.mean(z_mean, axis=0)
    z_var = np.mean(z_var, axis=0)
    rand_input = z_var * np.random.randn(n, z_var.shape[0]) + z_mean
    generated_data = generator.predict(rand_input)
    new_data = pd.DataFrame(data=generated_data, columns=list(org_data.columns))
    return new_data


