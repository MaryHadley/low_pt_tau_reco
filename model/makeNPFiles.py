import tensorflow as tf
import numpy as np
import uproot
from array import array

#random statement

#check whether you are using CPUs or GPUs
from tensorflow.python.client import device_lib
print('Available devices are', device_lib.list_local_devices())
print('#######')

tau_feature_names = [
    b'toUse_local_pi_m_lv1_pt',
    b'toUse_local_pi_m_lv1_theta',
#    b'toUse_local_pi_m_lv1_phi', #eliminate this branch because it is always pi for every entry by construction
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
#    b'toUse_local_pi_p_lv1_phi', #eliminate this branch because it is always pi for every entry by construction
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
#file  = uproot.open('CUT_nomMass_from_Otto.root')['tree'] #plug in my file
file  = uproot.open('fout_5March2020.root')['tree']

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

# split_the_train_features = np.split(big_features_train, 18)
# split_the_train_labels = np.split(big_labels_train, 18)
# 
# 
# for element in  enumerate(split_the_train_features):
#     print (element)
#     print("###")
# 
# for element in enumerate(split_the_train_labels):
#     print("POODLE! This is a label")
#     print(element)
#     print("###") 
# 
# count = 0
# for i in (split_the_train_features):
#     count +=1
#     print("count is:", count)
#     np.save("features_example_" +"%s" %str(count), split_the_train_features[count-1])
# 
# 
# #Checking, this seems to do what I want    
# #blah = np.load("feautures_train_example_1.npy")
# #print(blah)
# 
# 
# 
# count = 0
# for i in (split_the_train_labels):
#     count += 1
#     print("count is:", count)
#     np.save("labels_example_" +"%s" %str(count), split_the_train_labels[count-1]) 
#  
#  
# #blah1 = np.load("labels_train_example_1.npy")
# #print(blah1)
# 
# split_the_test_features = np.split(big_features_test, 2)
# split_the_test_labels = np.split(big_labels_test, 2)
# 
# for element in enumerate(split_the_test_features):
#     print("CAMEL!")
#     print(element)
#     print("###")
#     
# for element in enumerate(split_the_test_labels):
#     print("HORSE")
#     print(element)
#     print("###")
# 
# count = 0
# for i in (split_the_test_features):
#     count +=1
#     print("count is:", count)
#     np.save("features_test_example_" + "%s" %str(count), split_the_test_features[count-1])
# 
# count = 0
# for i in (split_the_test_labels):
#     count +=1
#     print("count is:", count)
#     np.save("labels_test_example_" + "%s" %str(count), split_the_test_labels[count-1])
# 
# blah2 = np.load("features_test_example_1.npy")
# print(blah2)
# 
# blah3 = np.load("labels_test_example_1.npy")
# print(blah3)
# 
# #for element in enumerate (split_the_train_features + split_the_test_features):
# #    print (element)

print("HIPPO!")
print(big_features_train)
print("HIPPO 2!")
print(big_features_test)

print("HIPPO 3!")
all_features = np.concatenate((big_features_train, big_features_test))
print(all_features)
print(all_features.shape)

list_to_make_np_files_features = np.split(all_features, 20)
print("list_to_make_np_files_features is:", list_to_make_np_files_features)

count = 0
for i in list_to_make_np_files_features:
    count += 1
    print("count is:", count)
    np.save("features_example_" + "%s" %str(count), list_to_make_np_files_features[count-1])

blah = np.load("features_example_1.npy")
print(blah)
# all_labels = np.concatenate((big_labels_train, big_labels_test))
# print(all_labels)
# print(all_labels.shape)

print("TIGER")
print(big_labels_train)
print(big_labels_train.shape)

print("TIGER 2")
print(big_labels_test)
print(big_labels_test.shape)

all_labels = np.concatenate((big_labels_train, big_labels_test))
print("TIGER 3!")
print(all_labels)
print(all_labels.shape)

list_to_make_np_files_labels = np.split(all_labels, 20)
print("list_to_make_np_files_labels is:", list_to_make_np_files_labels)

count = 0
for i in list_to_make_np_files_labels:
    count += 1
    print("count is:", count)
    np.save("labels_example_"  + "%s" %str(count), list_to_make_np_files_labels[count-1])
    
blah1 = np.load("labels_example_1.npy")
print(blah1)
    

