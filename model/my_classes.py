#Creating the class DataGenerator, which will allow us to feed our data in managable bites 
#to our training so that we can train on 100ks/millions of examples without having the training fail as a result of memory issues
#Example here: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly and thanks also to Bjorn for input

import numpy as np
import tensorflow 
import keras
from tensorflow.python.keras.utils.data_utils import Sequence

class DataGenerator(Sequence):
    """Generates manageable size chunks of data for keras."""
    def __init__(self, list_IDs, batch_size, dim, num_of_labels, shuffle=True): #I do not think I need labels #I may want to feed some place holder/default values in here
        """Initialize the class object."""
        self.num_of_labels = num_of_labels
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.on_epoch_end()
    
    def __len__(self):
        """Define the number of batches per epoch, return this number as an int."""
        return int(np.floor(len(self.list_IDs)/self.batch_size))
        
    def __getitem__ (self, index):
        """Get the indices of one single batch of data, returns list of form [index_start_for_this_batch, index_start_for_this_batch +1, ...index_stop_for_this_batch -1, index_stop_for_this_batch]."""
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        #Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        
        
        #Generate data
        
        X,Y = self.__data_generation(list_IDs_temp)
        
        return X,Y
        
    def on_epoch_end(self):
        """Updates indexes after each epoch."""
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def __data_generation(self, list_IDs_temp):
        """Generates chunks of data containing batch size samples."""
        #Initialize X,Y
        X = np.empty([self.batch_size, self.dim])
        Y = np.empty([self.batch_size, self.num_of_labels]) #I think I do not need to define the dtype because I am doing the Y part the same way as the X part is done aka with np.load #checked that indeed pT, theta, phi are floats, good
        
        #Generate data
        for i, ID in enumerate(list_IDs_temp):
            #Store the features
            X[i,] = np.load('toUse_data_features/features_example_' + ID + '.npy') 
            #Store labels
            Y[i,] = np.load('toUse_data_labels/labels_example_' + ID + '.npy')
            
        return X, Y 
    
    
