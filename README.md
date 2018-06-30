# Data Generators Based on Autoencoders and Variational Autoencoders

The project was developed as part of the Master's thesis: "Autoencoder based generators of semi-artificial data" at the University of Ljubljana. In case of any questions related to the project, feel free to contact me.

The solution consists of four modules:
- [preprocessing](preprocessing.py),
- [dynamic model choosing and training](model.py),
- [data generating](generator.py),
- [evaluation](evaluation.py).

The simplest and the shortest example of generating data is the following code:
```python
import pandas as pd
from preprocessing import preprocess, postprocess, fill_missing
from model import get_ae, get_vae
from generator import generate_data_ae, generate_data_vae

data = pd.read_csv(dataset)
data = fill_missing(data)
preprocessed_data, columns, numerical_cols, norm_map, nominal_cols = preprocess(data)

# generate with variational autoencoders
gen, enc = get_vae(preprocessed_data)
new_data_gen = generate_data_vae(generator=gen, encoder=enc, org_data=preprocessed_data, n=1000)
new_data = postprocess(new_data_gen, numerical_cols, norm_map, nominal_cols)

# generate with autoencoders
gen, enc = get_ae(preprocessed_data)
new_data_gen = generate_data_ae(generator=gen, encoder=enc, org_data=preprocessed_data, n=1000)
new_data = postprocess(new_data_gen, numerical_cols, norm_map, nominal_cols)
```

More advanced examples can be seen in files:
- testing_script.py
- testing_script_regression.py
- testing_script_rbf_datasets.py

## Data sets
Data sets used in the project, also available in file 01_data, were obtained from different sources listed below. Some of the data sets were slightly modified.

1. House Sales in King County, USA, https://www.kaggle.com/harlfoxem/housesalesprediction, accessed: 26.05.2018 (2016).
1. Diamonds data set, https://www.kaggle.com/shivam2503/diamonds, accessed: 26.05.2018 (2017).
1. V. Arel-Bundock, R datasets, https://vincentarelbundock.github.io/Rdatasets/, accessed: 26.05.2018 (2018).
2. N. Budincsevity, Weather in Szeged 2006-2016, https://www.kaggle.com/budincsevity/szeged-weather, accessed: 26.05.2018 (2017).
3. M. Choi, Medical Cost Personal Datasets, https://www.kaggle.com/mirichoi0218/insurance/data, accessed: 26.05.2018 (2018).
4. D. Dheeru, E. Karra Taniskidou, UCI Machine Learning Repository, http://archive.ics.uci.edu/ml (2017).
5. J. Li, Honey production In The USA (1998-2012), https://www.kaggle.com/jessicali9530/honey-production, accessed: 26.05.2018 (2018).
6. S. Sheather, A modern approach to regression with R (data sets), http://www.stat.tamu.edu/~sheather/book/data_sets.php, accessed: 26.05.2018 (2009).
