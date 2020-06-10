###Acknowledgements###

#This code is closely based off an example from Thea Aarrestad 
#That code can be found at: https://github.com/thaarres/Quantized_CNN/blob/master/hyperParamScan.py
#Thank you very much to Thea for all her help with this!
#She gives credit to the following by writing:
## "This code was adapted from
# https://github.com/jmduarte/JEDInet-code
# and takes advantage of the libraries at
# https://github.com/SheffieldML/GPy
# https://github.com/SheffieldML/GPyOpt
# https://github.com/jmduarte/JEDInet-code
# and takes advantage of the libraries at
# https://github.com/SheffieldML/GPy
# https://github.com/SheffieldML/GPyOpt"


import sys
import numpy as np
import tensorflow
import matplotlib; matplotlib.use('PDF') 
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.layers import Concatenate, BatchNormalization, Activation
from tensorflow.keras.utils import plot_model
from tensorflow.keras import regularizers #might not need this
from tensorflow.keras import backend as K #might not need this
from tensorflow.keras import metrics
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from tensorflow.keras.utils import to_categorical
import GPyOpt
import GPy
#import setGPU

from my_classes import DataGenerator 
from local_model_testGenerator_7March2020 import partition, params # I think this should make the definition of params and partition available to this file 
########################################
training_generator = DataGenerator(partition['train'], **params)
validation_generator = DataGenerator(partition['validation'], **params)
print(training_generator.num_of_labels)
print (training_generator.dim)
# myModel class

class myModel():
    def __init__(self, optmizer_index, activation_index, DNN_layers, DNN_neurons, dropout, batch_size, epochs, n_labels=3): 
    
        self.activation = ['relu', 'selu'] #currently using relu, threw in selu because Thea had it, but may want to replace with something else after more thinking
        self.optimizer = ['adam', 'nadam','adadelta'] #currently using adam, also just plucked these from Thea, might want to change options after some more thinking 
        self.optimizer_index = optmizer_index
        self.activation_index = activation_index
        self.DNN_layers = DNN_layers
        self.DNN_neurons = DNN_neurons
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs
        #self.n_labels = n_labels
#        n_labels = 3     
        training_generator = DataGenerator(partition['train'], **params) #this should return X, Y where X is for the features in the training sample, Y for the labels in the training sample
        validation_generator = DataGenerator(partition['validation'], **params) # this should return X,Y where X is for features in the test sample, Y is for the labels in the training sample
        #Do I need the self? --> I think what I did below is reasonable 
        
        self.generator = training_generator
        self.validation_data = validation_generator
        #Also copied from training script
        #X_train, Y_train = training_generator #Is this ok? # this is giving me a too many values to unpack error 
        #X_test, Y_test = validation_generator  #Same comment Ok, maybe I do not need this and it is just making things screwy...
        self.n_labels = (training_generator.num_of_labels) # should be 3
        self.input_shape = (training_generator.dim,) #should be 8
       # print(self.n_labels)
       # print (self.input_shape)
        #self.n_labels = 3
        #self.input_shape = (8,)
        #self.__y_test  = Y_test
        #self.__y_train = Y_train
        #self.__x_test  = X_test
        #self.__x_train = X_train
        
        
        ###### End block of skepticism ####
    
        self.__model = self.build()
    
    #model
    def build(self):
        
        myInput = Input(shape=(self.input_shape)) #this might be right?
        #myInput = Input(shape=(8))
        #Maybe there should be something about Sequential here?? e.g. x = Sequential()? # Maybe not, looking at: https://keras.io/guides/sequential_model/
       # x = Sequential()
#        x = Dense(input_shape=self.input_shape)(myInput) # should this actually be Sequential as opposed to Dense??
        x=BatchNormalization()(myInput)
        x = Dropout(self.dropout)(x)
        
        for i in range(1,self.DNN_layers): #be careful with numbering here and how you define DNN_layers, you want to make sure you get all of them here except for the input layer and the output layer
            x = Dense(self.DNN_neurons, activation=self.activation[self.activation_index], name='dense_%i' %i)(x) 
            x=BatchNormalization()(x)
            x = Dropout(self.dropout)(x)
            
        output = Dense(self.n_labels, name = 'final_output')(x) #Did I correctly tell it to use no activation function here or should I say activation function = '' or something like that
        #Looks like not based on: https://keras.io/api/layers/core_layers/dense/, looks like it defaults to None 
        #was right that I do NOT want an activation function for the last layer, see: https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/
        #Do I want kernel_initialization stuff...
        model = Model(myInput, outputs = output)
        model.summary()
        model.compile(optimizer=self.optimizer[self.optimizer_index], 
                      loss='mean_squared_error', metrics=['acc']) #Change this to my custom loss function when I have written it in 
        return model               
             
    #fit model
    def model_fit(self):
        self.__model.fit_generator(self.generator, self.validation_data, epochs=self.epochs, verbose=0, 
                                   callbacks = [EarlyStopping(monitor='val_loss', patience=5, verbose=1),
                                                           ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0), 
                                                           TerminateOnNaN()]) 
   
    # evaluate  model
    def model_evaluate(self):
        self.model_fit()
        evaluation = self.__model.evaluate_generator(self.validation_data, batch_size=self.batch_size, verbose=0) 
        return evaluation 
        
####################################################

#Runner function for the model

def run_model(optmizer_index, DNN_layers, activation_index, DNN_neurons, dropout, batch_size, epochs, labels):
    
    _model = myModel(optmizer_index, DNN_layers, activation_index, DNN_neurons, dropout, batch_size, epochs, labels)
    
    model_evaluation = _model.model_evaluate()
    return model_evaluation

n_epochs = 500 #Might want to play with this, but let's start here (is there a reason you took this as fixed as opposed to trying to optimize it?)
n_labels = 3

#Bayesian Optimization
# the bounds dict should be in order of continuous type and then discrete type #QUESTION, you note the order should be continuous then discrete but it does not look like yours follows that...hmm


bounds = [{'name': 'optmizer_index',        'type': 'discrete',   'domain': (0, 1, 2)},
          {'name': 'activation_index',  'type': 'discrete',   'domain': (0, 1)},
          {'name': 'DNN_neurons',           'type': 'discrete',   'domain': (28, 64, 128, 256)},
          {'name': 'DNN_layers',            'type': 'discrete',   'domain': (1, 2, 3, 4, 5)}, #Does up to 5 seem reasonable?
          {'name': 'dropout',               'type': 'continuous', 'domain': (0.0, 0.4)},
          {'name': 'batch_size',            'type': 'discrete',   'domain': (32, 50, 200, 500,1000)}] #just copied what Thea had, maybe should tweak?
          
          
# function to optimize model
def f(x):
    print ("x parameters are") 
    print(x)
    evaluation = run_model(optmizer_index       = int(x[:,0]), 
                           activation_index     = int(x[:,1]), 
                           DNN_neurons          = int(x[:,2]), 
                           DNN_layers           = int(x[:,3]),
                           dropout              = float(x[:,4]),
                           batch_size           = int(x[:,5]),
                           epochs = n_epochs,
                           labels = n_labels
                           )
    print("LOSS:\t{0} \t ACCURACY:\t{1}".format(evaluation[0], evaluation[1]))
    print(evaluation)
    return evaluation[0]

if __name__ == "__main__":
  opt_model = GPyOpt.methods.BayesianOptimization(f=f, domain=bounds)
  opt_model.run_optimization(max_iter=10000,report_file='bayOpt.txt')
  opt_model.plot_acquisition('bayOpt_acqu.pdf')
  opt_model.plot_convergence('bayOpt_conv.pdf')

  print("DONE")
  print("x:",opt_model.x_opt)
  print("y:",opt_model.fx_opt)
  
  # print optimized model
  print("""
  Optimized Parameters:
  \t{0}:\t{1}
  \t{2}:\t{3}
  \t{4}:\t{5}
  \t{6}:\t{7}
  \t{8}:\t{9}
  \t{10}:\t{11}
  \t{12}:\t{13}
  """.format(bounds[0]["name"],opt_model.x_opt[0],
             bounds[1]["name"],opt_model.x_opt[1],
             bounds[2]["name"],opt_model.x_opt[2],
             bounds[3]["name"],opt_model.x_opt[3],
             bounds[4]["name"],opt_model.x_opt[4],
             bounds[5]["name"],opt_model.x_opt[5],
            
             ))
  print("optimized loss: {0}".format(opt_model.fx_opt))
