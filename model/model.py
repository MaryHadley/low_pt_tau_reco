import tensorflow as tf
import numpy as np
import uproot
from array import array

#check whether you are using CPUs or GPUs
from tensorflow.python.client import device_lib
print('Available devices are', device_lib.list_local_devices())
print('#######')

tau_feature_names = [
    b'pi_minus1_pt',
    b'pi_minus1_eta',
    b'pi_minus1_phi',
    b'pi_minus2_pt',
    b'pi_minus2_eta',
    b'pi_minus2_phi',
    b'pi_minus3_pt',
    b'pi_minus3_eta',
    b'pi_minus3_phi',
]
tau_label_names = [
    b'neutrino_pt',
    b'neutrino_eta',
    b'neutrino_phi',
]

antitau_feature_names = [
    b'pi_plus1_pt',
    b'pi_plus1_eta',
    b'pi_plus1_phi',
    b'pi_plus2_pt',
    b'pi_plus2_eta',
    b'pi_plus2_phi',
    b'pi_plus3_pt',
    b'pi_plus3_eta',
    b'pi_plus3_phi',
]
antitau_label_names = [
    b'antineutrino_pt',
    b'antineutrino_eta',
    b'antineutrino_phi',
]

#file = uproot.open('cartesian_upsilon_taus.root')['tree']
#file = uproot.open('momentum_vector_data100k.root')['tree']
#file = uproot.open('cartesian_upsilon_taus_ALL.root')['tree']
#file = uproot.open('cartesian_upsilon_taus_15GeV_Mary_ALL.root')['tree']
#file = uproot.open('cartesian_upsilon_tausGS_88_plus_91.root')['tree'] #ok this is probably too big, need to figure out how to get this to work
#file = uproot.open('cartesian_upsilon_taus_GSwBP77.root')['tree'] #still too big...
#file  = uproot.open('cartesian_upsilon_taus_GSwBP6.root')['tree'] #3204 GS events
file  = uproot.open('momentum_vector_data100k_WSO.root')['tree']

tau_features = []
tau_labels = []
antitau_features = []
antitau_labels = []

for name in tau_feature_names:
    if b'_phi' in name:
        tau_features.append(
            np.sin(file.array(name))
        )
        tau_features.append(
            np.cos(file.array(name))
        )
    else:
        tau_features.append(file.array(name))

for name in tau_label_names:
    if b'_phi' in name:
        tau_labels.append(
            np.sin(file.array(name))
        )
        tau_labels.append(
            np.cos(file.array(name))
        )
    else:
        tau_labels.append(file.array(name))

for name in antitau_feature_names:
    if b'_phi' in name:
        antitau_features.append(
            np.sin(file.array(name))
        )
        antitau_features.append(
            np.cos(file.array(name))
        )
    else:
        antitau_features.append(file.array(name))

for name in antitau_label_names:
    if b'_phi' in name:
        antitau_labels.append(
            np.sin(file.array(name))
        )
        antitau_labels.append(
            np.cos(file.array(name))
        )
    else:
        antitau_labels.append(file.array(name))

tau_features = np.transpose(np.array(tau_features))

total_tau_pt = tau_features[:, 0] + tau_features[:, 4] + tau_features[:, 8]
#comment out the lines below if you do NOT want normalized pT
#tau_features[:, 0] = tau_features[:, 0] / total_tau_pt
#tau_features[:, 4] = tau_features[:, 4] / total_tau_pt
#tau_features[:, 8] = tau_features[:, 8] / total_tau_pt

tau_features_train = tau_features[0: int(0.9 * tau_features.shape[0]), :]
tau_features_test = tau_features[int(0.9 * tau_features.shape[0]):, :]

tau_labels = np.transpose(np.array(tau_labels))
tau_labels_train = tau_labels[0: int(0.9 * tau_labels.shape[0]), :]
tau_labels_test = tau_labels[int(0.9 * tau_labels.shape[0]):, :]

antitau_features = np.transpose(np.array(antitau_features))

total_antitau_pt = antitau_features[:, 0] + antitau_features[:, 4] + antitau_features[:, 8]
#Comment out the lines below if you do NOT want normalized pT
#antitau_features[:, 0] = antitau_features[:, 0] / total_antitau_pt
#antitau_features[:, 4] = antitau_features[:, 4] / total_antitau_pt
#antitau_features[:, 8] = antitau_features[:, 8] / total_antitau_pt

antitau_features_train = antitau_features[0: int(0.9 * antitau_features.shape[0]), :]
antitau_features_test = antitau_features[int(0.9 * antitau_features.shape[0]):, :]

antitau_labels = np.transpose(np.array(antitau_labels))
antitau_labels_train = antitau_labels[0: int(0.9 * antitau_labels.shape[0]), :]
antitau_labels_test = antitau_labels[int(0.9 * antitau_labels.shape[0]):, :]


def create_model():
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Dense(640, activation=tf.keras.activations.relu, input_shape=(12,)) 
    )
    model.add(
        tf.keras.layers.Dropout(0.3)
    )
    model.add(
        tf.keras.layers.Dense(1280, activation=tf.keras.activations.relu)
    )
    model.add(
        tf.keras.layers.Dropout(0.5)
    )
    model.add(
        tf.keras.layers.Dense(2560, activation=tf.keras.activations.relu)
    )
    model.add(
        tf.keras.layers.Dropout(0.5)
    )
    model.add(
        tf.keras.layers.Dense(1280, activation=tf.keras.activations.relu)
    )
    model.add(
        tf.keras.layers.Dropout(0.3)
    )
    model.add(
        tf.keras.layers.Dense(640, activation=tf.keras.activations.relu)
    )
    model.add(
        tf.keras.layers.Dropout(0.2)
    )
    model.add(
        tf.keras.layers.Dense(320, activation=tf.keras.activations.relu)
    )
    model.add(
        tf.keras.layers.Dropout(0.1)
    )
    model.add(
        tf.keras.layers.Dense(160, activation=tf.keras.activations.relu)
    )
    model.add(
        tf.keras.layers.Dropout(0.1)
    )
    model.add(
        tf.keras.layers.Dense(128, activation=tf.keras.activations.relu)
    )
    model.add(
        tf.keras.layers.Dropout(0.1)
    )
    model.add(
        tf.keras.layers.Dense(64, activation=tf.keras.activations.relu)
    )
    model.add(
        tf.keras.layers.Dropout(0.1)
    )
    model.add(
        tf.keras.layers.Dense(32, activation=tf.keras.activations.relu)
    )
    model.add(
        tf.keras.layers.Dropout(0.05)
    )
    model.add(
        tf.keras.layers.Dense(8, activation=tf.keras.activations.relu)
    )
    model.add(
        tf.keras.layers.Dense(4)
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=0.001, decay=0.0001),
        loss=tf.keras.losses.mean_squared_error
    )
    return model


tau_model = create_model()
antitau_model = create_model()

tau_model.fit(
    tau_features_train,
    tau_labels_train,
    batch_size=20,
    epochs=400,
    validation_data=(tau_features_test, tau_labels_test)
)
tau_model.evaluate(
    tau_features_test,
    tau_labels_test,
    batch_size=10
)
antitau_model.fit(
    antitau_features_train,
    antitau_labels_train,
    batch_size=20,
    epochs=400,
    validation_data=(antitau_features_test, antitau_labels_test)
)
antitau_model.evaluate(
    antitau_features_test,
    antitau_labels_test,
    batch_size=10
)

pred = tau_model.predict(
    tau_features_test
)
anti_pred = antitau_model.predict(
    antitau_features_test
)

tau_model.save('tau_model_reproduce_WSO_my_own_try_no_norm.hdf5')
antitau_model.save('antitau_model_reproduce_WSO_my_own_no_norm.hdf5')
print ('tau_model summary is:', tau_model.summary())
print ('antitau_model summary is:', antitau_model.summary())
