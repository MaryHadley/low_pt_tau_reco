#Overview


Codes to preprocess CMS data of X --> tau tau --> ((3 charged pion plus >= 0 neutral pions plus a neutrino) times 2, one for each tau) to get it into a format that can be fed to a DNN, the DNN model code itself, and code to test the model performance.

#Directories (Preprocessing, Model)

#Preprocessing

Preprocessing has been done in CMSSW_10_2_15 using the default python 2 version that comes with that release.

Four versions of preprocessing code exist:

1) a preprocessing code written by Willem and Shray (and Otto) that runs on the miniaod data tier. Matches gen level info to reco level info --> LowPtTauMatcher.py, to run:
python LowPtTauMatcher.py inputFiles=your_input_file.root suffix=some_useful_tag_to_append_to_the_output_file, e.g. perhaps the name of the root file you are preprocessing
  
2) a preprocessing code to run on the miniaod data tier but that uses only the gen level info from these files (therefore no matching of gen level info to reco info) --> GenSimLowPtTauMatcher_noCheckForNeutralPiInDecayChainButPiPtCutIncl_hackedToUseJustGenInfoFromMA.py, usage: python GenSimLowPtTauMatcher_noCheckForNeutralPiInDecayChainButPiPtCutIncl_hackedToUseJustGenInfoFromMA.py inputFiles=your_input_file.root suffix=some_useful_tag_to_append_to_the_output_file, e.g. perhaps the name of the root file you are preprocessing

  
3) a preprocessing code to run on the Gen Sim level data tier (and therefore by definition can only use gen level info) -->GenSimLowPtTauMatcher_noCheckForNeutralPiInDecayChainButPiPtCutIncl.py, to run: python GenSimLowPtTauMatcher_noCheckForNeutralPiInDecayChainButPiPtCutIncl.py inputFiles=your_input_file.root suffix=some_useful_tag_to_append_to_the_output_file, e.g. perhaps the name of the root file you are preprocessing

4) a preprocessing code that converts all the four vector information into a local frame (UPGRADE DOCUMENTATION TO FULLY DESCRIBE THIS LATER)...this code currently  UNDER CONSTRUCTION
  
Note that if you use the preprocessing codes that only use the gen information (so options 2 or 3 here), you will need to fiddle with the piPtCut setting depending on what mass point of the X parent sample you are using X --> tau tau --> ((3 charged pion plus >= 0 neutral pions plus a neutrino) times 2, one for each tau). For X = nominal mass of the upsilon particle (9.46 GeV), a piPtCut of 0.35 GeV seems to work. For X = 15 GeV, a piPtCut of 0.7 GeV is what is currently being used.

Note that we have verified that training the model on gen level only info and then applying these gen-trained models to full miniaod test data does not result in a significant performance degredation...this is only sort of true, see email that I sent to Greg for more details, TO BE RETURNED TO AT A LATER DATE.
  
 #Model
 
 Documentation in progress...



#Original README from Willem and Shray, Summer 2019

# Model performance

For information on the model's performance and purpose, please see the following slides: https://docs.google.com/presentation/d/1V6UwJSQ6zp3HWdvUhwIrohQ6-u5zRTiAnlsvEhLabRU/edit?usp=sharing

# Using model

If you wish to use the model without modifying it, two trained models are available for use (tau_model.hdf5 and antitau_model.hdf5). They are available for download at ```/afs/cern.ch/user/w/wspeckma/public/low_pt_tau_reconstruction/tau_model.hdf5``` and ```/afs/cern.ch/user/w/wspeckma/public/low_pt_tau_reconstruction/antitau_model.hdf5```. 

The following code loads the models and predicts the neutrino and antineutrino properties:

```python
import numpy as np
import tensorflow as tf

tau_model = tf.keras.models.load_model('tau_model.hdf5')
antitau_model = tf.keras.models.load_model('antitau_model.hdf5')

tau_pred = tau_model.predict(tau_features)
antitau_pred = antitau_model.predict(antitau_features)

```

Where tau_features and antitau_features are numpy arrays with the format: 

```python
[
  [pion1_pt, pion1_eta, np.sin(pion1_phi), np.cos(pion1_phi), ..., ...],
  ...
]
```

With the pion features repeated once for each of the three pions, such that the input arrays are of the shape ```python [no. events, 12]```. 

tau_pred and antitau_pred will then be numpy arrays of the form:

```python
[
  [pt, eta, sin(phi), cos(phi)],
  ...
]
```
If you prefer to have radians rather than sine and cosine, you can use the following code: 

```python
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
    
tau_pred[:, 2] = arr_get_angle(tau_pred[:, 2], tau_pred[:, 3])
tau_pred = tau_pred[:, 0: 3]

antitau_pred[:, 2] = arr_get_angle(antitau_pred[:, 2], antitau_pred[:, 3])
antitau_pred = antitau_pred[:, 0: 3]
```
This will ensure tau_pred and antitau_pred will be numpy arrays of the form:


```python
[
  [pt, eta, phi],
  ...
]
```

# Modifying the model

If you'd like to train the model yourself, modify it or adapt it for different purposes, the model file and the histogram generation file are included. The data used for the training of the model is available at ```/afs/cern.ch/user/w/wspeckma/public/low_pt_tau_reconstruction/momentum_vector_data100k.root```. 


