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
tau_features_test = tau_features[int(0.9 * tau_features.shape[0]):, :]

tau_labels = np.transpose(np.array(tau_labels))
tau_labels_test = tau_labels[int(0.9 * tau_labels.shape[0]):, :]

antitau_features = np.transpose(np.array(antitau_features))
antitau_features_test = antitau_features[int(0.9 * antitau_features.shape[0]):, :]

antitau_labels = np.transpose(np.array(antitau_labels))
antitau_labels_test = antitau_labels[int(0.9 * antitau_labels.shape[0]):, :]

branches = [
    'tau_pt',
    'tau_eta',
    'tau_phi',
    'tau_mass',
    'tau_pt_no_neutrino',
    'tau_eta_no_neutrino',
    'tau_phi_no_neutrino',
    'tau_mass_no_neutrino',
    'antitau_pt',
    'antitau_eta',
    'antitau_phi',
    'antitau_mass',
    'antitau_pt_no_neutrino',
    'antitau_eta_no_neutrino',
    'antitau_phi_no_neutrino',
    'antitau_mass_no_neutrino',
    'upsilon_pt',
    'upsilon_eta',
    'upsilon_phi',
    'upsilon_mass',
    'upsilon_pt_no_neutrino',
    'upsilon_eta_no_neutrino',
    'upsilon_phi_no_neutrino',
    'upsilon_mass_no_neutrino',
]
file_out = ROOT.TFile('3_prong_no_neutral_graphs.root', 'recreate')
file_out.cd()

tofill = OrderedDict(zip(branches, [-99.] * len(branches)))

masses = ROOT.TNtuple('tree', 'tree', ':'.join(branches))

tau_model = tf.keras.models.load_model('tau_model.hdf5')
antitau_model = tf.keras.models.load_model('antitau_model.hdf5')

pred = tau_model.predict(
    tau_features_test
)
anti_pred = antitau_model.predict(
    antitau_features_test
)


def arr_normalize(arr):
    arr = np.where(arr > 1, 1, arr)
    arr = np.where(arr < -1, -1, arr)
    return arr


def arr_get_angle(sin_value, cos_value):
    sin_value = arr_normalize(sin_value)
    cos_value = arr_normalize(cos_value)
    return np.where(
        sin_value > 0,
        np.where(
            cos_value > 0,
            (np.arcsin(sin_value) + np.arccos(cos_value)) / 2,
            ((np.pi - np.arcsin(sin_value)) + np.arccos(cos_value)) / 2
        ),
        np.where(
            cos_value > 0,
            (np.arcsin(sin_value) - np.arccos(cos_value)) / 2,
            ((- np.arccos(cos_value)) - (np.pi + np.arcsin(sin_value))) / 2
        )
    )


pred[:, 2] = arr_get_angle(pred[:, 2], pred[:, 3])
pred = pred[:, 0: 3]

anti_pred[:, 2] = arr_get_angle(anti_pred[:, 2], anti_pred[:, 3])
anti_pred = anti_pred[:, 0: 3]

tau_features_test[:, 2] = arr_get_angle(tau_features_test[:, 2], tau_features_test[:, 3])
tau_features_test[:, 6] = arr_get_angle(tau_features_test[:, 6], tau_features_test[:, 7])
tau_features_test[:, 10] = arr_get_angle(tau_features_test[:, 10], tau_features_test[:, 11])
tau_features_test = tau_features_test[:, [0, 1, 2, 4, 5, 6, 8, 9, 10]]

tau_labels_test[:, 2] = arr_get_angle(tau_labels_test[:, 2], tau_labels_test[:, 3])
tau_labels_test = tau_labels_test[:, 0: 3]

antitau_features_test[:, 2] = arr_get_angle(antitau_features_test[:, 2], antitau_features_test[:, 3])
antitau_features_test[:, 6] = arr_get_angle(antitau_features_test[:, 6], antitau_features_test[:, 7])
antitau_features_test[:, 10] = arr_get_angle(antitau_features_test[:, 10], antitau_features_test[:, 11])
antitau_features_test = antitau_features_test[:, [0, 1, 2, 4, 5, 6, 8, 9, 10]]

antitau_labels_test[:, 2] = arr_get_angle(antitau_labels_test[:, 2], antitau_labels_test[:, 3])
antitau_labels_test = antitau_labels_test[:, 0: 3]

# Tau plots
plt.plot(tau_labels_test[:, 0], pred[:, 0], 'ro')
fig1 = plt.gcf()
fig1.savefig('tau_pt.png')
plt.clf()

plt.plot(tau_labels_test[:, 1], pred[:, 1], 'ro')
fig1 = plt.gcf()
fig1.savefig('tau_eta.png')
plt.clf()

plt.plot(tau_labels_test[:, 2], pred[:, 2], 'ro')
fig1 = plt.gcf()
fig1.savefig('tau_phi.png')
plt.clf()

# Antitau plots
plt.plot(antitau_labels_test[:, 0], anti_pred[:, 0], 'ro')
fig1 = plt.gcf()
fig1.savefig('antitau_pt.png')
plt.clf()

plt.plot(antitau_labels_test[:, 1], anti_pred[:, 1], 'ro')
fig1 = plt.gcf()
fig1.savefig('antitau_eta.png')
plt.clf()

plt.plot(antitau_labels_test[:, 2], anti_pred[:, 2], 'ro')
fig1 = plt.gcf()
fig1.savefig('antitau_phi.png')
plt.clf()

for event in range(pred.shape[0]):
    tau_lorentz_no_neutrino = TLorentzVector()

    for index in range(0, tau_features_test.shape[1], 3):
        lorentz = TLorentzVector()
        lorentz.SetPtEtaPhiM(
            tau_features_test[event][index],
            tau_features_test[event][index + 1],
            tau_features_test[event][index + 2],
            0.139
        )
        tau_lorentz_no_neutrino += lorentz

    tofill['tau_pt_no_neutrino'] = tau_lorentz_no_neutrino.Pt()
    tofill['tau_eta_no_neutrino'] = tau_lorentz_no_neutrino.Eta()
    tofill['tau_phi_no_neutrino'] = tau_lorentz_no_neutrino.Phi()
    tofill['tau_mass_no_neutrino'] = tau_lorentz_no_neutrino.M()

    tau_lorentz = TLorentzVector()
    tau_lorentz.SetPtEtaPhiM(
        tau_lorentz_no_neutrino.Pt(),
        tau_lorentz_no_neutrino.Eta(),
        tau_lorentz_no_neutrino.Phi(),
        tau_lorentz_no_neutrino.M(),
    )

    for index in range(0, pred.shape[1], 3):
        lorentz = TLorentzVector()
        lorentz.SetPtEtaPhiM(
            pred[event][index],
            pred[event][index + 1],
            pred[event][index + 2],
            0
        )
        tau_lorentz += lorentz

    tofill['tau_pt'] = tau_lorentz.Pt()
    tofill['tau_eta'] = tau_lorentz.Eta()
    tofill['tau_phi'] = tau_lorentz.Phi()
    tofill['tau_mass'] = tau_lorentz.M()

    antitau_lorentz_no_neutrino = TLorentzVector()

    for index in range(0, tau_features_test.shape[1], 3):
        lorentz = TLorentzVector()
        lorentz.SetPtEtaPhiM(
            antitau_features_test[event][index],
            antitau_features_test[event][index + 1],
            antitau_features_test[event][index + 2],
            0.139
        )
        antitau_lorentz_no_neutrino += lorentz

    tofill['antitau_pt_no_neutrino'] = antitau_lorentz_no_neutrino.Pt()
    tofill['antitau_eta_no_neutrino'] = antitau_lorentz_no_neutrino.Eta()
    tofill['antitau_phi_no_neutrino'] = antitau_lorentz_no_neutrino.Phi()
    tofill['antitau_mass_no_neutrino'] = antitau_lorentz_no_neutrino.M()

    antitau_lorentz = TLorentzVector()
    antitau_lorentz.SetPtEtaPhiM(
        antitau_lorentz_no_neutrino.Pt(),
        antitau_lorentz_no_neutrino.Eta(),
        antitau_lorentz_no_neutrino.Phi(),
        antitau_lorentz_no_neutrino.M(),
    )

    for index in range(0, anti_pred.shape[1], 3):
        lorentz = TLorentzVector()
        lorentz.SetPtEtaPhiM(
            anti_pred[event][index],
            anti_pred[event][index + 1],
            anti_pred[event][index + 2],
            0
        )
        antitau_lorentz += lorentz

    tofill['antitau_pt'] = antitau_lorentz.Pt()
    tofill['antitau_eta'] = antitau_lorentz.Eta()
    tofill['antitau_phi'] = antitau_lorentz.Phi()
    tofill['antitau_mass'] = antitau_lorentz.M()

    upsilon_lorentz = tau_lorentz + antitau_lorentz
    upsilon_lorentz_no_neutrino = tau_lorentz_no_neutrino + antitau_lorentz_no_neutrino

    tofill['upsilon_pt_no_neutrino'] = upsilon_lorentz_no_neutrino.Pt()
    tofill['upsilon_eta_no_neutrino'] = upsilon_lorentz_no_neutrino.Eta()
    tofill['upsilon_phi_no_neutrino'] = upsilon_lorentz_no_neutrino.Phi()
    tofill['upsilon_mass_no_neutrino'] = upsilon_lorentz_no_neutrino.M()
    tofill['upsilon_pt'] = upsilon_lorentz.Pt()
    tofill['upsilon_eta'] = upsilon_lorentz.Eta()
    tofill['upsilon_phi'] = upsilon_lorentz.Phi()
    tofill['upsilon_mass'] = upsilon_lorentz.M()

    masses.Fill(array('f', tofill.values()))

file_out.cd()
masses.Write()
file_out.Close()

