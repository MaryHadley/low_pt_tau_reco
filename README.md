# Model performance

For information on the model's performance and purpose, please see the following slides: https://docs.google.com/presentation/d/1V6UwJSQ6zp3HWdvUhwIrohQ6-u5zRTiAnlsvEhLabRU/edit?usp=sharing

# Using model

If you wish to use the model without modifying it, two trained models are available for use (tau_model.hdf5 and antitau_model.hdf5). They are available for download at ```/afs/cern.ch/user/w/wspeckma/public/low_pt_tau_reconstruction```. 

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



