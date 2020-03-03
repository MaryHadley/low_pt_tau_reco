#Creating the class DataGenerator, which will allow us to feed our data in managable bites to our training so that we can train on 100ks/millions of examples without having the training fail as a result of memory issues
#Example here: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly and thanks also to Bjorn for input

import numpy as np
import keras

class DataGenerator(keras.utils.Sequence):
    """Generates manageable size chunks of data for keras."""
    def __init__(self, list_IDs, labels, batch_size, dim, num_of_labels, shuffle=True): #I may want to feed some place holder/default values in here
        """Initialize the class object."""
        self.num_of_labels = num_of_labels
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.shuttle = shuffle
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
        
        X,y = self.__data_generation(list_IDs_temp)
        
        return X,y 
        
    def on_epoch_end(self):
        """Updates indexes after each epoch."""
        self.indexes = np.arrange(len(self_list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def __data_generation(self, list_IDs_temp):
        """Generates chunks of data containing batch size samples."""
        #Initialize X,y
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), *self.num_of_labels, dtype=float) #check that pT, theta, phi are indeed floats not doubles
        
        #Generate data
        for i, ID in enumerate(list_IDs_temp):
            #Store the sample
            X[i,] = np.load('data/' + ID + '.npy') #This assumes you are storing the data in some folder called data that is inside the folder in which you are running your training code (aka you have some folder with your training code, within that folder is your folder called data where you have put all your data)
        
            #Store labels #not sure about this part, asking John
            y[i,] = self.labels[ID] #might have to update because I have three labels...I think this is a good guess though
        
        return X, y #Again, might need to tweak because I have three labels...
        
    
    
