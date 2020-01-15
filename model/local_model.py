import tensorflow as tf
import numpy as np
import uproot
from array import array

#check whether you are using CPUs or GPUs
from tensorflow.python.client import device_lib
print('Available devices are', device_lib.list_local_devices())
print('#######')

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
] # change feature names, keep the order of pt, THETA, phi
tau_label_names = [
    b'toUse_local_neu_lv_pt',
    b'toUse_local_neu_lv_theta',
    b'toUse_local_neu_lv_phi',
] #ditto for label names

antitau_feature_names = [
    b'toUse_local_pi_p_lv1_pt',
    b'toUse_local_pi_p_lv1_theta',
    b'toUse_local_pi_p_lv1_phi',
    b'toUse_local_pi_p_lv2_pt',
    b'toUse_local_pi_p_lv2_theta',
    b'toUse_local_pi_p_lv2_phi',
    b'toUse_local_pi_p_lv3_pt',
    b'toUse_local_pi_p_lv3_theta',
    b'toUse_local_pi_p_lv3_phi',
] #change feature names, keep the order of pt, THETA, phi
antitau_label_names = [
    b'toUse_local_antineu_lv_pt',
    b'toUse_local_antineu_lv_theta',
    b'toUse_local_antineu_lv_phi',
]#ditto for label names

#file = uproot.open('cartesian_upsilon_taus.root')['tree']
#file = uproot.open('momentum_vector_data100k.root')['tree']
#file = uproot.open('cartesian_upsilon_taus_ALL.root')['tree']
#file = uproot.open('cartesian_upsilon_taus_15GeV_Mary_ALL.root')['tree']
#file = uproot.open('cartesian_upsilon_tausGS_88_plus_91.root')['tree'] #ok this is probably too big, need to figure out how to get this to work
#file = uproot.open('cartesian_upsilon_taus_GSwBP77.root')['tree'] #still too big...
#file  = uproot.open('cartesian_upsilon_taus_GSwBP6.root')['tree'] #3204 GS events
file  = uproot.open('CUT_nomMass_from_Otto.root')['tree'] #plug in my file

tau_features = []
tau_labels = []
antitau_features = []
antitau_labels = []

for name in tau_feature_names:
     #get rid of the cosine/sine part
    tau_features.append(file.array(name))

for name in tau_label_names:
    tau_labels.append(file.array(name)) #get rid of the cosine/sine part

for name in antitau_feature_names:
    #get rid of the cosine/sine part
    antitau_features.append(file.array(name))

for name in antitau_label_names:
    antitau_labels.append(file.array(name)) #get rid of the cosine/sine part
    
#print "tau_features before transpose", tau_features #list of lists
tau_features = np.transpose(np.array(tau_features))
#print "tau_features:", tau_features
print ("tau_features.shape", tau_features.shape) #shape is N events for the rows, 9 columns

#total_tau_pt = tau_features[:, 0] + tau_features[:, 4] + tau_features[:, 8]
#comment out the lines below if you do NOT want normalized pT
#tau_features[:, 0] = tau_features[:, 0] / total_tau_pt
#tau_features[:, 4] = tau_features[:, 4] / total_tau_pt
#tau_features[:, 8] = tau_features[:, 8] / total_tau_pt

tau_features_train = tau_features[0: int(0.9 * tau_features.shape[0]), :]
print ("tau_features_train.shape", tau_features_train.shape)

tau_features_test = tau_features[int(0.9 * tau_features.shape[0]):, :]
print ("tau_features_test.shape", tau_features_test.shape)

tau_labels = np.transpose(np.array(tau_labels))
print ("tau_labels.shape", tau_labels.shape)

tau_labels_train = tau_labels[0: int(0.9 * tau_labels.shape[0]), :]
print ("tau_labels_train.shape", tau_labels_train.shape)
tau_labels_test = tau_labels[int(0.9 * tau_labels.shape[0]):, :]
print ("tau_labels_test.shape", tau_labels_test.shape)

antitau_features = np.transpose(np.array(antitau_features))

#total_antitau_pt = antitau_features[:, 0] + antitau_features[:, 4] + antitau_features[:, 8]
#Comment out the lines below if you do NOT want normalized pT
#antitau_features[:, 0] = antitau_features[:, 0] / total_antitau_pt
#antitau_features[:, 4] = antitau_features[:, 4] / total_antitau_pt
#antitau_features[:, 8] = antitau_features[:, 8] / total_antitau_pt

antitau_features_train = antitau_features[0: int(0.9 * antitau_features.shape[0]), :]
print ("antitau_features_train.shape", antitau_features_train.shape)
antitau_features_test = antitau_features[int(0.9 * antitau_features.shape[0]):, :]
print ("antitau_features_test.shape", antitau_features_test.shape)

antitau_labels = np.transpose(np.array(antitau_labels))
antitau_labels_train = antitau_labels[0: int(0.9 * antitau_labels.shape[0]), :]
print ("antitau_labels_train.shape", antitau_labels_train.shape)
antitau_labels_test = antitau_labels[int(0.9 * antitau_labels.shape[0]):, :]
print ("antitau_labels_test.shape", antitau_labels_test.shape)

#here we are going to want to make a big_features_train, big_features_test, big_labels_train, and big_labels_test 
#by concatenating the tau and antitau features train, tau and anti tau features test, 
#tau and antitau labels test, and tau and anti tau labels test. Do concatenation row wise. This way the first half will be tau data, the second half antitau
#and we will know the row ordering is event 0-N for the first half, then event 0-N for the second half aka if we had 3 events
# we would have rows 0,1,2 with the tau data from events 1,2,3 and then rows 3,4,5 for the antitau data from events 1,2,3

big_features_train = np.concatenate((tau_features_train,antitau_features_train)) # for some reason you need a double parentheses, which I tend to forget
print ("big_features_train.shape is:", big_features_train.shape)
big_features_test = np.concatenate((tau_features_test, antitau_features_test))
print ("big_features_test.shape is:", big_features_test.shape)

big_labels_train = np.concatenate((tau_labels_train, antitau_labels_train))
print ("big_labels_train.shape is:", big_labels_train.shape)

big_labels_test = np.concatenate((tau_labels_test, antitau_labels_test))
print ("big_labels_test.shape is:", big_labels_test.shape)

def create_model():
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Dense(640, activation=tf.keras.activations.relu, input_shape=(9,)) #change input shape to 9, was 12 before
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
        tf.keras.layers.Dense(3) #only change to architecture will be to make this output 3 not 4
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=0.001, decay=0.0001),
        loss=tf.keras.losses.mean_squared_error
    )
    return model


# tau_model = create_model()
# antitau_model = create_model()

big_model = create_model()
big_model.fit(
      big_features_train,
      big_labels_train,
      batch_size=20,
      epochs=400,
     validation_data=(big_features_test, big_labels_test)
 )
 
big_model.evaluate(
    big_features_test,
    big_labels_test,
     batch_size=10
 )
 
 
 
 
 
 
# antitau_model.fit(
#     antitau_features_train,
#     antitau_labels_train,
#     batch_size=20,
#     epochs=400,
#     validation_data=(antitau_features_test, antitau_labels_test)
# )
# antitau_model.evaluate(
#     antitau_features_test,
#     antitau_labels_test,
#     batch_size=10
# )
# 
# pred = tau_model.predict(
#     tau_features_test
# )
# anti_pred = antitau_model.predict(
#     antitau_features_test
# )
# 
big_model.save('big_local_model_14Jan2020.hdf5')
# antitau_model.save('antitau_model_reproduce_WSO_my_own_no_norm.hdf5')
print ('big_model summary is:', big_model.summary())
# print ('antitau_model summary is:', antitau_model.summary())
