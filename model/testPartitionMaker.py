import os, sys, glob

# inputFeaturesDir = '/afs/cern.ch/work/m/mhadley/public/noCMSSWLowPtTauReco/model/data_features'
# inputFeaturesFileName = 'features_'
# inputFeaturesFileList =  glob.glob('%s*.npy' %(inputFeaturesFileName))
# print ('inputFeaturesFilesList is:', inputFeaturesFileList)
# 
# # # inputLabelsDir =
# # # inputLabelsFileName =
# # # inputLabelsFileList =


# features_train_example_list = []
# 
# for i in range(1,19):
#     print (i)
#     print("features_train_example_%s" %str(i))
#     features_train_example_list.append("features_train_example_%s" %str(i))
#     
# print(features_train_example_list)

partition_train_ID_list = []
for i in range(1,19):
    print(i)
    partition_train_ID_list.append("%s" %str(i))

print(partition_train_ID_list)

partition_validate_ID_list = []
for i in range(19,21):
    print(i)
    partition_validate_ID_list.append("%s" %str(i))

print(partition_validate_ID_list)