import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import uproot
from array import array
import ROOT
from ROOT import TLorentzVector
from collections import OrderedDict

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


def normalize_sin_cos(value):
    if value > 1:
        return 1
    elif value < -1:
        return -1
    return value


def get_radians_from_sin_cos(sin_value, cos_value):
    sin_value = normalize_sin_cos(sin_value)
    cos_value = normalize_sin_cos(cos_value)
    if sin_value > 0 and cos_value > 0:
        return (np.arcsin(sin_value) + np.arccos(cos_value)) / 2
    elif sin_value > 0 > cos_value:
        return ((np.pi - np.arcsin(sin_value)) + np.arccos(cos_value)) / 2
    elif sin_value < 0 < cos_value:
        return (np.arcsin(sin_value) - np.arccos(cos_value)) / 2
    elif sin_value < 0 and cos_value < 0:
        return ((- np.arccos(cos_value)) - (np.pi + np.arcsin(sin_value))) / 2


file = uproot.open('momentum_vector_data100k.root')['tree']

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
tau_features[:, 0] = tau_features[:, 0] / total_tau_pt
tau_features[:, 4] = tau_features[:, 4] / total_tau_pt
tau_features[:, 8] = tau_features[:, 8] / total_tau_pt

tau_features_train = tau_features[0: int(0.9 * tau_features.shape[0]), :]
tau_features_test = tau_features[int(0.9 * tau_features.shape[0]):, :]

tau_labels = np.transpose(np.array(tau_labels))
tau_labels_train = tau_labels[0: int(0.9 * tau_labels.shape[0]), :]
tau_labels_test = tau_labels[int(0.9 * tau_labels.shape[0]):, :]

antitau_features = np.transpose(np.array(antitau_features))

total_antitau_pt = antitau_features[:, 0] + antitau_features[:, 4] + antitau_features[:, 8]
antitau_features[:, 0] = antitau_features[:, 0] / total_antitau_pt
antitau_features[:, 4] = antitau_features[:, 4] / total_antitau_pt
antitau_features[:, 8] = antitau_features[:, 8] / total_antitau_pt

antitau_features_train = antitau_features[0: int(0.9 * antitau_features.shape[0]), :]
antitau_features_test = antitau_features[int(0.9 * antitau_features.shape[0]):, :]

antitau_labels = np.transpose(np.array(antitau_labels))
antitau_labels_train = antitau_labels[0: int(0.9 * antitau_labels.shape[0]), :]
antitau_labels_test = antitau_labels[int(0.9 * antitau_labels.shape[0]):, :]

branches = [
    'tau_mass',
    'antitau_mass',
    'epsilon_mass'
]
file_out = ROOT.TFile('nn_tau_antitau_mass_with_neutrinos.root', 'recreate')
file_out.cd()

tofill = OrderedDict(zip(branches, [-99.] * len(branches)))

masses = ROOT.TNtuple('tree', 'tree', ':'.join(branches))


def create_model():
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Dense(640, activation=tf.keras.activations.relu)
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

tau_model.save('tau_model.hdf5')
antitau_model.save('antitau_model.hdf5')

for event in range(pred.shape[0]):
    tau_lorentz = TLorentzVector()
    for index in range(0, pred.shape[1], 4):
        lorentz = TLorentzVector()
        phi = get_radians_from_sin_cos(
            pred[event][index + 2],
            pred[event][index + 3]
        )
        lorentz.SetPtEtaPhiM(
            pred[event][index],
            pred[event][index + 1],
            phi,
            0
        )
        tau_lorentz += lorentz
    for index in range(0, tau_features_test.shape[1], 4):
        lorentz = TLorentzVector()
        phi = get_radians_from_sin_cos(
            tau_features_test[event][index + 2],
            tau_features_test[event][index + 3]
        )
        lorentz.SetPtEtaPhiM(
            tau_features_test[event][index],
            tau_features_test[event][index + 1],
            phi,
            0.139
        )
        tau_lorentz += lorentz

    antitau_lorentz = TLorentzVector()
    for index in range(0, anti_pred.shape[1], 4):
        lorentz = TLorentzVector()
        phi = get_radians_from_sin_cos(
            anti_pred[event][index + 2],
            anti_pred[event][index + 3]
        )
        lorentz.SetPtEtaPhiM(
            anti_pred[event][index],
            anti_pred[event][index + 1],
            phi,
            0
        )
        antitau_lorentz += lorentz
    for index in range(0, tau_features_test.shape[1], 4):
        lorentz = TLorentzVector()
        phi = get_radians_from_sin_cos(
            antitau_features_test[event][index + 2],
            antitau_features_test[event][index + 3]
        )
        lorentz.SetPtEtaPhiM(
            antitau_features_test[event][index],
            antitau_features_test[event][index + 1],
            phi,
            0.139
        )
        antitau_lorentz += lorentz

    epsilon_lorentz = tau_lorentz + antitau_lorentz

    tofill['tau_mass'] = tau_lorentz.M()
    tofill['antitau_mass'] = antitau_lorentz.M()
    tofill['epsilon_mass'] = epsilon_lorentz.M()
    masses.Fill(array('f', tofill.values()))

file_out.cd()
masses.Write()
file_out.Close()



