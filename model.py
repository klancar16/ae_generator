from keras.layers import Input, Dense, Lambda, Dropout
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras import backend as K
import numpy as np
from math import ceil


def get_ae(df, cutting_parameter=0.9, epochs=100, batch_size=16, activation='relu', drop_rate=0.0,
           threshold_rate=0.1):

    data_mat = df.values
    np.random.shuffle(data_mat)
    n_samples, n_attributes = data_mat.shape

    early_stop = EarlyStopping(monitor='val_loss',
                               min_delta=0.001,
                               patience=5,
                               verbose=0, mode='auto')

    layer_sizes = []
    if n_attributes < 100:
        layer_sizes.append(n_attributes * 2)
    layer_sizes.extend([int(ceil((layer_sizes[0] if layer_sizes else n_attributes) * (cutting_parameter**x)))
                        for x in range(1, 10)])
    steps = len(layer_sizes)-1

    losses = []
    val_losses = []

    threshold = 0
    cur_loss = 1
    for i in range(0, len(layer_sizes)):
        new_layer = layer_sizes[i]
        ae_input = Input(shape=(n_attributes,))
        encoded = ae_input
        for layer_size in layer_sizes[:i]:
            encoded = Dense(layer_size, activation=activation)(encoded)
            encoded = Dropout(drop_rate)(encoded)

        encoded = Dense(new_layer, activation=activation)(encoded)
        decoded = encoded

        for layer_size in layer_sizes[:i][::-1]:
            decoded = Dense(layer_size, activation=activation)(decoded)
            decoded = Dropout(drop_rate)(decoded)
        decoded = Dense(n_attributes, activation='sigmoid')(decoded)

        autoencoder = Model(ae_input, decoded)
        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
        autoencoder.fit(data_mat, data_mat,
                        epochs=epochs,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_split=0.1,
                        verbose=0,
                        callbacks=[early_stop]
                        )
        if autoencoder.history.history["val_loss"][-1] + threshold >= cur_loss:
            steps = i
            break

        cur_loss = autoencoder.history.history["loss"][-1]
        losses.append(cur_loss)
        threshold = cur_loss * threshold_rate
        val_losses.append(autoencoder.history.history["val_loss"][-1])

    ae_input = Input(shape=(n_attributes,))
    encoded = ae_input
    for layer_size in layer_sizes[:steps+1]:
        encoded = Dense(layer_size, activation=activation)(encoded)
        encoded = Dropout(drop_rate)(encoded)

    decoded = encoded
    for layer_size in layer_sizes[:steps][::-1]:
        decoded = Dense(layer_size, activation=activation)(decoded)
        decoded = Dropout(drop_rate)(decoded)

    decoded = Dense(n_attributes, activation='sigmoid')(decoded)
    autoencoder = Model(ae_input, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    autoencoder.fit(data_mat, data_mat,
                    epochs=epochs,
                    batch_size=batch_size,  # usually 32--512 data points
                    shuffle=True,
                    validation_split=0.1,
                    verbose=0,
                    callbacks=[early_stop]
                    )

    gen_input = Input(shape=(layer_sizes[steps],))
    decoder_layers = gen_input
    for layer in autoencoder.layers[-(2 * steps) - 1:]:
        if 'dense' in layer.name:
            decoder_layers = layer(decoder_layers)
    generator = Model(gen_input, decoder_layers)
    encoder = Model(ae_input, encoded)
    return generator, encoder


def init_vae_loss(z_mean, z_log_var):
    beta = 0.01

    def vae_loss(input, output):
        xent_loss = K.sum(K.binary_crossentropy(input, output), axis=1)
        kl_loss = - beta * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        total_loss = K.mean(xent_loss + kl_loss)
        return total_loss
    return vae_loss


def init_sampling(latent_dim):
    epsilon_std = 1.0

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                                  stddev=epsilon_std)
        return z_mean + K.exp(z_log_var/2) * epsilon
    return sampling


def get_vae(df, cutting_parameter=0.1, epochs=50, batch_size=16, activation='tanh', threshold_rate=0.1, drop_rate=0.2):
    early_stop = EarlyStopping(monitor='val_loss',
                               min_delta=0.01,
                               patience=5,
                               verbose=0, mode='auto')

    data_mat = df.values
    np.random.shuffle(data_mat)
    n_samples, original_dim = data_mat.shape

    layer_sizes = []
    if original_dim < 100: layer_sizes.append(original_dim * 2)
    layer_sizes.extend([int(ceil((layer_sizes[0] if layer_sizes else original_dim) * (cutting_parameter**x)))
                        for x in range(1, 10)])
    steps = len(layer_sizes)

    losses = []
    val_losses = []

    threshold = 0
    cur_loss = 1000
    for i in range(0, len(layer_sizes)):
        new_layer = layer_sizes[i-1] if i > 0 else original_dim
        latent_dim = int(ceil(new_layer*2))
        x = Input(shape=(original_dim,))

        encoded = x
        for layer_size in layer_sizes[:i]:
            encoded = Dense(layer_size, activation=activation)(encoded)
            encoded = Dropout(drop_rate)(encoded)

        z_mean = Dense(latent_dim)(encoded)
        z_log_var = Dense(latent_dim)(encoded)

        sampling = init_sampling(latent_dim)

        z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
        decoded = z
        decoded = Dropout(drop_rate)(decoded)

        for layer_size in layer_sizes[:i][::-1]:
            decoded = Dense(layer_size, activation=activation)(decoded)
            decoded = Dropout(drop_rate)(decoded)
        decoded = Dense(original_dim, activation='sigmoid')(decoded)

        vae = Model(x, decoded)

        vae_loss = init_vae_loss(z_mean, z_log_var)

        vae.compile(optimizer='rmsprop', loss=vae_loss)
        vae.fit(data_mat, data_mat,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                validation_split=0.2,
                verbose=0,
                callbacks=[early_stop]
                )

        if vae.history.history["val_loss"][-1] + threshold >= cur_loss:
            val_losses.append(vae.history.history["val_loss"][-1])
            steps = i - 1
            break

        losses.append(vae.history.history["loss"][-1])
        cur_loss = vae.history.history["val_loss"][-1]
        threshold = cur_loss * threshold_rate
        val_losses.append(vae.history.history["val_loss"][-1])

    ae_input = Input(shape=(original_dim,))

    latent_dim = int(ceil((layer_sizes[steps - 1] if steps > 0 else original_dim) * 2))
    encoded = ae_input
    for layer_size in layer_sizes[:steps]:
        encoded = Dense(layer_size, activation=activation)(encoded)
        encoded = Dropout(drop_rate)(encoded)

    z_mean = Dense(latent_dim)(encoded)
    z_log_var = Dense(latent_dim)(encoded)

    sampling = init_sampling(latent_dim)

    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
    encoder = Model(ae_input, [z_mean, z_log_var, z], name='encoder')
    decoded = z
    decoded = Dropout(drop_rate)(decoded)
    for layer_size in layer_sizes[:steps][::-1]:
        decoded = Dense(layer_size, activation=activation)(decoded)
        decoded = Dropout(drop_rate)(decoded)

    decoded = Dense(original_dim, activation='sigmoid')(decoded)

    vae_loss = init_vae_loss(z_mean, z_log_var)

    vae = Model(ae_input, decoded)
    vae.compile(optimizer='rmsprop', loss=vae_loss)
    vae.fit(data_mat, data_mat,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_split=0.2,
            verbose=0,
            callbacks=[early_stop])

    # print(vae.summary())
    gen_input = Input(shape=(latent_dim,))
    decoder_layers = gen_input
    for layer in vae.layers[-(2*steps+1):]:
        if 'dense' in layer.name:
            decoder_layers = layer(decoder_layers)
    generator = Model(gen_input, decoder_layers)

    return generator, encoder
