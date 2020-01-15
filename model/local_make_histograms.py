import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import uproot
from array import array
import ROOT
from ROOT import TLorentzVector
from collections import OrderedDict
import math
from frameChangingUtilities import *
import sys
sys.stdout.flush()

# I think to avoid needing to rewrite stuff, I can do this in the two chunks (tau and antitau) just load the big model for both
# then can reuse the code, just need to do an unconversion back to the labe frame at the very end in each tau/anti tau chunk before adding the lorentz vectors

tau_feature_names = [
    b'toUse_local_pi_m_lv1_pt',
    b'toUse_local_pi_m_lv1_theta',
    b'toUse_local_pi_m_lv1_phi',
    b'toUse_local_pi_m_lv2_pt',
    b'toUse_local_pi_m_lv2_theta',
    b'toUse_local_pi_m_lv2_phi',
    b'toUse_local_pi_m_lv3_pt',
    b'toUse_local_pi_m_lv3_theta',
    b'toUse_local_pi_m_lv3_phi',
] #change branch names

#create what I will need to do the conversion
tau_to_unrotate_info_names = [
     b'initial_leadPt_pi_m_in_AllInZFrame_phi',
     b'orig_vis_taum_theta',
     b'orig_vis_taum_phi'
     ]  #TO DO!!!




tau_label_names = [
    b'toUse_local_neu_lv_pt',
    b'toUse_local_neu_lv_theta',
    b'toUse_local_neu_lv_phi',
]

antitau_feature_names =  [
    b'toUse_local_pi_p_lv1_pt',
    b'toUse_local_pi_p_lv1_theta',
    b'toUse_local_pi_p_lv1_phi',
    b'toUse_local_pi_p_lv2_pt',
    b'toUse_local_pi_p_lv2_theta',
    b'toUse_local_pi_p_lv2_phi',
    b'toUse_local_pi_p_lv3_pt',
    b'toUse_local_pi_p_lv3_theta',
    b'toUse_local_pi_p_lv3_phi',
]

antitau_to_unrotate_info_names = [
b'initial_leadPt_pi_p_in_AllInZFrame_phi',
b'orig_vis_taup_theta',
b'orig_vis_taup_phi'
]

antitau_label_names = [
    b'toUse_local_antineu_lv_pt',
    b'toUse_local_antineu_lv_theta',
    b'toUse_local_antineu_lv_phi',
]

#create what I will need to do the conversion
antitau_to_unrotate_info = [] #TO DO!!!

file  = uproot.open("cartesian_upsilon_taus_.root")["tree"] #test file
 
tau_features = []
tau_labels = []
antitau_features = []
antitau_labels = []
tau_to_unrotate_info =[]
antitau_to_unrotate_info = []

for name in tau_feature_names:
    tau_features.append(file.array(name))

for name in tau_label_names:
    tau_labels.append(file.array(name))
    
for name in tau_to_unrotate_info_names:
    tau_to_unrotate_info.append(file.array(name))
    

for name in antitau_feature_names:
    antitau_features.append(file.array(name))

for name in antitau_label_names:
    antitau_labels.append(file.array(name))

for name in antitau_to_unrotate_info_names:
    antitau_to_unrotate_info.append(file.array(name))

tau_to_unrotate_info = np.transpose(np.array(tau_to_unrotate_info))
antitau_to_unrotate_info = np.transpose(np.array(antitau_to_unrotate_info))
print ("tau_to_unrotate_info is:", tau_to_unrotate_info)
print ("antitau_to_unrotate_info is:", antitau_to_unrotate_info)

tau_features = np.transpose(np.array(tau_features))
tau_features_test = tau_features[int(0.9 * tau_features.shape[0]):, :]
print ("tau_features_test.shape", tau_features_test.shape)

tau_labels = np.transpose(np.array(tau_labels))
tau_labels_test = tau_labels[int(0.9 * tau_labels.shape[0]):, :]
print ("tau_labels_test.shape", tau_labels_test.shape)

antitau_features = np.transpose(np.array(antitau_features))
antitau_features_test = antitau_features[int(0.9 * antitau_features.shape[0]):, :]
print ("antitau_features_test.shape", antitau_features_test.shape)

antitau_labels = np.transpose(np.array(antitau_labels))
antitau_labels_test = antitau_labels[int(0.9 * antitau_labels.shape[0]):, :]
print ("antitau_labels_test.shape", antitau_labels_test.shape)

branches = [
    'tau_pt',
    'tau_theta',
    'tau_phi',
    'tau_mass',
    'tau_pt_no_neutrino',
    'tau_theta_no_neutrino',
    'tau_phi_no_neutrino',
    'tau_mass_no_neutrino',
    'antitau_pt',
    'antitau_theta',
    'antitau_phi',
    'antitau_mass',
    'antitau_pt_no_neutrino',
    'antitau_theta_no_neutrino',
    'antitau_phi_no_neutrino',
    'antitau_mass_no_neutrino',
    'upsilon_pt',
    'upsilon_theta',
    'upsilon_phi',
    'upsilon_mass',
    'upsilon_pt_no_neutrino',
    'upsilon_theta_no_neutrino',
    'upsilon_phi_no_neutrino',
    'upsilon_mass_no_neutrino',
]
file_out = ROOT.TFile('local_test_15Jan2020.root', 'recreate')
file_out.cd()

tofill = OrderedDict(zip(branches, [-99.] * len(branches)))

masses = ROOT.TNtuple('tree', 'tree', ':'.join(branches))

#here we are going to want to make a big_features_train, big_features_test, big_labels_train, and big_labels_test 
#by concatenating the tau and antitau features train, tau and anti tau features test, 
#tau and antitau labels test, and tau and anti tau labels test. Do concatenation row wise. This way the first half will be tau data, the second half antitau
#and we will know the row ordering is event 0-N for the first half, then event 0-N for the second half aka if we had 3 events
# we would have rows 0,1,2 with the tau data from events 1,2,3 and then rows 3,4,5 for the antitau data from events 1,2,3


big_features_test = np.concatenate((tau_features_test, antitau_features_test))
print ("big_features_test.shape is:", big_features_test.shape)


big_labels_test = np.concatenate((tau_labels_test, antitau_labels_test))
print ("big_labels_test.shape is:", big_labels_test.shape)

big_model = tf.keras.models.load_model('big_local_model_14Jan2020.hdf5')

big_pred = big_model.predict(
    big_features_test
 )

print ("big_pred is:", big_pred)
print ("big_pred.shape is:", big_pred.shape)


# 
# def arr_normalize(arr):
#     arr = np.where(arr > 1, 1, arr)
#     arr = np.where(arr < -1, -1, arr)
#     return arr
# 
# 
# def arr_get_angle(sin_value, cos_value):
#     sin_value = arr_normalize(sin_value)
#     cos_value = arr_normalize(cos_value)
#     return np.where(
#         sin_value > 0,
#         np.where(
#             cos_value > 0,
#             (np.arcsin(sin_value) + np.arccos(cos_value)) / 2,
#             ((np.pi - np.arcsin(sin_value)) + np.arccos(cos_value)) / 2
#         ),
#         np.where(
#             cos_value > 0,
#             (np.arcsin(sin_value) - np.arccos(cos_value)) / 2,
#             ((- np.arccos(cos_value)) - (np.pi + np.arcsin(sin_value))) / 2
#         )
#     )
# 
# 
# pred[:, 2] = arr_get_angle(pred[:, 2], pred[:, 3])
# pred = pred[:, 0: 3]
# print('pred after converting to radians as opposed to sine and cosine is:', pred)
# 
# anti_pred[:, 2] = arr_get_angle(anti_pred[:, 2], anti_pred[:, 3])
# anti_pred = anti_pred[:, 0: 3]
# 
# tau_features_test[:, 2] = arr_get_angle(tau_features_test[:, 2], tau_features_test[:, 3])
# tau_features_test[:, 6] = arr_get_angle(tau_features_test[:, 6], tau_features_test[:, 7])
# tau_features_test[:, 10] = arr_get_angle(tau_features_test[:, 10], tau_features_test[:, 11])
# tau_features_test = tau_features_test[:, [0, 1, 2, 4, 5, 6, 8, 9, 10]]
# 
# tau_labels_test[:, 2] = arr_get_angle(tau_labels_test[:, 2], tau_labels_test[:, 3])
# tau_labels_test = tau_labels_test[:, 0: 3]
# 
# antitau_features_test[:, 2] = arr_get_angle(antitau_features_test[:, 2], antitau_features_test[:, 3])
# antitau_features_test[:, 6] = arr_get_angle(antitau_features_test[:, 6], antitau_features_test[:, 7])
# antitau_features_test[:, 10] = arr_get_angle(antitau_features_test[:, 10], antitau_features_test[:, 11])
# antitau_features_test = antitau_features_test[:, [0, 1, 2, 4, 5, 6, 8, 9, 10]]
# 
# antitau_labels_test[:, 2] = arr_get_angle(antitau_labels_test[:, 2], antitau_labels_test[:, 3])
# antitau_labels_test = antitau_labels_test[:, 0: 3]
# 
# # Overall nu plots
plt.plot(big_labels_test[:, 0], big_pred[:, 0], 'ro')
fig1 = plt.gcf()
fig1.savefig('big_nu_pt.png')
plt.clf()

plt.plot(big_labels_test[:, 1], big_pred[:, 1], 'ro')
fig1 = plt.gcf()
fig1.savefig('big_nu_theta.png')
plt.clf()

plt.plot(big_labels_test[:, 2], big_pred[:, 2], 'ro')
fig1 = plt.gcf()
fig1.savefig('big_nu_phi.png')
plt.clf()

# # Tau anti nu plots
# plt.plot(antitau_labels_test[:, 0], anti_pred[:, 0], 'ro')
# fig1 = plt.gcf()
# fig1.savefig('tau_antinu_pt.png')
# plt.clf()
# 
# plt.plot(antitau_labels_test[:, 1], anti_pred[:, 1], 'ro')
# fig1 = plt.gcf()
# fig1.savefig('tau_antinu_eta.png')
# plt.clf()
# 
# plt.plot(antitau_labels_test[:, 2], anti_pred[:, 2], 'ro')
# fig1 = plt.gcf()
# fig1.savefig('tau_antinu_phi.png')
# plt.clf()
# 
# 
# #print tau_labels_test
# print(tau_labels_test)
# #error for pt,done really quickly, can make this code better 
max_error = 0
total_error = 0
for x in range(big_labels_test.shape[0]): #get number of rows with shape[0]
    error = abs(big_labels_test[x][0] - big_pred[x][0])
    max_error = max(error, max_error)
    total_error += error

max_error =max_error
mean_error = total_error/big_labels_test.shape[0]
# 

print('max_error for nu pt is:', max_error)
print('mean_error for nu pt is:', mean_error)
# 
# max_error = 0
# total_error = 0
# for x in range(antitau_labels_test.shape[0]): #get number of rows with shape[0]
# #     error = abs(antitau_labels_test[x][0] - anti_pred[x][0])
# #     max_error = max(error, max_error)
# #     total_error += error
# # 
# # max_error =max_error
# # mean_error = total_error/antitau_labels_test.shape[0]
# # 
# # #print 'max_error for anti tau pt is:', max_error
# # print('max_error for tau anti nu pt is:', max_error)
# # print('mean_error for tau anti nu pt is:', mean_error)
# # 
# # #print tau_labels_test.shape[0]
# # #print tau_labels_test.shape[1]
# 
#error for eta
max_error = 0
total_error = 0
for x in range(big_labels_test.shape[0]):
    error = abs(big_labels_test[x][1] -big_pred[x][1])
    max_error = max(error, max_error)
    total_error += error
# 
max_error =max_error
mean_error = total_error/big_labels_test.shape[0]
# 
# #print 'max_error for tau eta is:', max_error
print('max_error for nu theta is:', max_error)
print('mean_error for nu theta is:', mean_error)
# 
# max_error = 0
# total_error = 0
# for x in range(antitau_labels_test.shape[0]):
#     error = abs(antitau_labels_test[x][1] - anti_pred[x][1])
#     max_error = max(error, max_error)
#     total_error += error
# 
# max_error =max_error
# mean_error = total_error/antitau_labels_test.shape[0]
# 
# print('max_error for tau anti nu  eta is:', max_error)
# print('mean_error for tau anti nu  eta is:', mean_error)
# 
# 
#error for phi...this is hacked together, need to think about how deal with getting phi correctly in the most general case
max_error = 0
total_error = 0
for x in range(big_labels_test.shape[0]):
    error = abs(big_labels_test[x][2] - big_pred[x][2])
    print ('nu phi error is:', error)
    if error >= math.pi:
        print ('POODLE!')
        error = abs((2*math.pi)-error)
        print ('new tau nu phi error is:', error)
    else: 
        error = error
    max_error = max(error, max_error)
    total_error += error
# 
max_error =max_error
mean_error = total_error/tau_labels_test.shape[0]
# 
print ('max_error for nu phi is:', max_error)
print ('mean_error for nu phi is:', mean_error)
# 
# max_error = 0
# total_error = 0
# for x in range(antitau_labels_test.shape[0]):
#     error = abs(antitau_labels_test[x][2] - anti_pred[x][2])
#     print('tau anti nu phi error is:', error)
#     if error >= math.pi:
#         print('DRAGON')
#         error = abs((2*math.pi)-error) 
#         print ('new tau anti nu phi error is:', error)
#     else:
#         error = error
#     max_error = max(error, max_error)
#     total_error += error
# 
# max_error =max_error
# mean_error = total_error/antitau_labels_test.shape[0]
# 
# print ('max_error for tau anti nu phi is:', max_error)
# print ('mean_error for tau anti nu  phi is:', mean_error)
# 
# 
# split big_pred into tau_pred and anti_pred so I don't have to change the indexing too much
split_the_pred = np.split(big_pred, 2)
print ("split_the_pred is:", split_the_pred)
pred = split_the_pred[0]
anti_pred = split_the_pred[1]
print ("pred is:", pred)
print ("pred.shape is:", pred.shape)
print ("anti_pred is:", anti_pred)
print ("anti_pred.shape is:", anti_pred.shape)
print("tau_features_test.shape is:", tau_features_test.shape)


#change pred to being pt ETA phi as opposed to pt THETA phi so we can use the SetPtEtaPhiM method
def arr_get_eta(theta_value):
    theta = -np.log(0.5*theta_value)
    return theta
    
pred[:,1] = arr_get_eta(pred[:,1])
print("dog! pred is:", pred)
tau_features_test[:,1] = arr_get_eta(tau_features_test[:,1])
tau_features_test[:,4] = arr_get_eta(tau_features_test[:,4])
tau_features_test[:,7] = arr_get_eta(tau_features_test[:,7])

anti_pred[:,1] = arr_get_eta(anti_pred[:,1])
antitau_features_test[:,1]=arr_get_eta(antitau_features_test[:,1])
antitau_features_test[:,4]=arr_get_eta(antitau_features_test[:,4])
antitau_features_test[:,7]=arr_get_eta(antitau_features_test[:,7])
### 


print("Hungarian Horntail! Mary testing!")
#v = Math.Root.LorentzVector()

for event in range(pred.shape[0]):
    tau_lorentz_no_neutrino = TLorentzVector()
#    firedCount = 0
    for index in range(0, tau_features_test.shape[1], 3):
        lorentz = TLorentzVector()
        lorentz.SetPtEtaPhiM(
        (tau_features_test[event][index]),
        (tau_features_test[event][index + 1]),
        (tau_features_test[event][index + 2]),
        (0.139)
        )
        print( (-np.log(0.5*(tau_features_test[event][index + 1]))))
        tau_lorentz_no_neutrino += lorentz
        #print ("I fired")
        #firedCount += 1
    tofill['tau_pt_no_neutrino'] = tau_lorentz_no_neutrino.Pt()
    print("tau_lorentz_neutrino.Px()", tau_lorentz_no_neutrino.Px())
    print("tau_lorentz_no_neutrino.Py()", tau_lorentz_no_neutrino.Py())
    print("tau_lorentz_no_neutrino.Pz()", tau_lorentz_no_neutrino.Pz())
    print("tau_lorentz_no_neutrino.E()", tau_lorentz_no_neutrino.E())
    tofill['tau_eta_no_neutrino'] = tau_lorentz_no_neutrino.Eta()
    tofill['tau_phi_no_neutrino'] = tau_lorentz_no_neutrino.Phi()
    tofill['tau_mass_no_neutrino'] = tau_lorentz_no_neutrino.M()
    print ("tau_mass_no_neutrino", tau_lorentz_no_neutrino.M())
    print("tau_eta_no_neutrino", tau_lorentz_no_neutrino.Eta())
    tau_lorentz = TLorentzVector()
    tau_lorentz.SetPxPyPzE(
        tau_lorentz_no_neutrino.Px(),
        tau_lorentz_no_neutrino.Py(),
        tau_lorentz_no_neutrino.Pz(),
        tau_lorentz_no_neutrino.E(),
    )
    print ("tau_lorentz.Px()", tau_lorentz.Px())
    print("tau_lorentz.Py()", tau_lorentz.Py())
    print("tau_lorentz_Pz()", tau_lorentz.Pz())
    for index in range(0, pred.shape[1], 3):
        lorentz = TLorentzVector()
        lorentz.SetPtEtaPhiM(
        (pred[event][index]),
        (pred[event][index + 1]),
        (pred[event][index + 2]),
        (0)
        )
        
        
        
        print("(pred[event][index + 1])", (pred[event][index + 1]))
        tau_lorentz += lorentz
        print("pred[event][index] should be pt",(pred[event][index]))
        print("(pred[event][index + 2]) should be phi",  (pred[event][index + 2]))
        print ("test M!",tau_lorentz.M())
        print("test!", tau_lorentz.Pt())
        print("test Eta", tau_lorentz.Eta())
        print("test Phi", tau_lorentz.Phi())

    tofill['tau_pt'] = tau_lorentz.Pt()
    tofill['tau_eta'] = tau_lorentz.Eta()
    tofill['tau_phi'] = tau_lorentz.Phi()
    tofill['tau_mass'] = tau_lorentz.M()
    print ("tau_lorentz.M()", tau_lorentz.M())

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
    antitau_lorentz.SetPxPyPzE(
        antitau_lorentz_no_neutrino.Px(),
        antitau_lorentz_no_neutrino.Py(),
        antitau_lorentz_no_neutrino.Pz(),
        antitau_lorentz_no_neutrino.E(),
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

  #   upsilon_lorentz = tau_lorentz + antitau_lorentz
#     upsilon_lorentz_no_neutrino = tau_lorentz_no_neutrino + antitau_lorentz_no_neutrino
# 
#     tofill['upsilon_pt_no_neutrino'] = upsilon_lorentz_no_neutrino.Pt()
#     tofill['upsilon_eta_no_neutrino'] = upsilon_lorentz_no_neutrino.Eta()
#     tofill['upsilon_phi_no_neutrino'] = upsilon_lorentz_no_neutrino.Phi()
#     tofill['upsilon_mass_no_neutrino'] = upsilon_lorentz_no_neutrino.M()
#     tofill['upsilon_pt'] = upsilon_lorentz.Pt()
#     tofill['upsilon_eta'] = upsilon_lorentz.Eta()
#     tofill['upsilon_phi'] = upsilon_lorentz.Phi()
#     tofill['upsilon_mass'] = upsilon_lorentz.M()

    masses.Fill(array('f', tofill.values()))
#    print ("firedCount is:", firedCount)
file_out.cd()
masses.Write()
file_out.Close()

