import tensorflow as tf
import numpy as np
import uproot
from array import array
#from tensorflow.python.keras.utils.data_utils import Sequence
from my_classes import DataGenerator


partition = {'train': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18'], 'validation': ['19', '20']}
#print(partition)

params = {'dim': 8,
          'batch_size': 1,
          'num_of_labels': 3,
          'shuffle': True}

def create_model():
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Dense(100, activation=tf.keras.activations.relu, input_shape=(8,)) #change input shape to 8, was 12 before
    )
    model.add(
        tf.keras.layers.Dropout(0.1) #try lower dropout for input layer
    )
    
    #Hidden layer 1
    model.add(
        tf.keras.layers.Dense(100, activation=tf.keras.activations.relu)
    )
    model.add(
        tf.keras.layers.Dropout(0.5)
    )
    
    #Hidden layer 2
    model.add(
        tf.keras.layers.Dense(100, activation=tf.keras.activations.relu)
    )
    model.add(
        tf.keras.layers.Dropout(0.5)
    )
    
    #Hidden layer 3
    model.add(
        tf.keras.layers.Dense(100, activation=tf.keras.activations.relu)
    )
    model.add(
        tf.keras.layers.Dropout(0.5)
    )
    
    #Hidden layer 4 
    model.add(
        tf.keras.layers.Dense(100, activation=tf.keras.activations.relu)
    )
    model.add(
        tf.keras.layers.Dropout(0.5)
    )
    
    #Hidden layer 5
    model.add(
        tf.keras.layers.Dense(100, activation=tf.keras.activations.relu)
    )
    model.add(
        tf.keras.layers.Dropout(0.5)
    )

    model.add(
        tf.keras.layers.Dense(3) #only change to architecture will be to make this output 3 not 4
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=0.001, decay=0.0001),
        loss=tf.keras.losses.mean_squared_error
    )
    return model



big_model = create_model()


training_generator = DataGenerator(partition['train'],  **params)
validation_generator = DataGenerator(partition['validation'], **params)

big_model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True
                  )


# tau_model = create_model()
# antitau_model = create_model()

# big_model = create_model()
# big_model.fit(
#       big_features_train,
#       big_labels_train,
#       batch_size=20,
#       epochs=400,
#      validation_data=(big_features_test, big_labels_test)
#  )
#  
# big_model.evaluate(
#     big_features_test,
#     big_labels_test,
#      batch_size=10
#  )
 
 
 
 

# big_model.save('new_arch_big_local_model_noPiPtCut_nomMass_fromOtto_constantInputFeatureEliminated_2March2020.hdf5')
# # antitau_model.save('antitau_model_reproduce_WSO_my_own_no_norm.hdf5')
# print ('big_model summary is:', big_model.summary())
# # print ('antitau_model summary is:', antitau_model.summary())
