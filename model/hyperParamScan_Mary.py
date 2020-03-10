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
import setGPU

from my_classes import DataGenerator 
from local_model_testGenerator_7March2020 import partition, params # I think this should make the definition of params and partition available to this file 
########################################

# myModel class

class myModel():
    def __init__(self, optmizer_index, activation_index, DNN_layers, DNN_neurons, dropout, batch_size, epochs, batch_norm_index, n_labels): #this last part with batch norm index might be incorrect
    
        self.activation = ['relu', 'selu'] #currently using relu, threw in selu because Thea had it, but may want to replace with something else after more thinking
        self.optimizer = ['adam', 'nadam','adadelta'] #currently using adam, also just plucked these from Thea, might want to change options after some more thinking 
        self.optimizer_index = optmizer_index
        self.activation_index = activation_index
        self.DNN_layers = DNN_layers
        self.DNN_neurons = DNN_neurons
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.batch_norm_index = batch_norm_index
        self.n_labels = n_labels
    
        ### Skeptical about this being right, sorry, I clearly did not quite get it when we met, sorry to be slow ####
        training_generator = DataGenerator(partition['train'], **params) #this should return X, Y where X is for the features in the training sample, Y for the labels in the training sample
        validation_generator = DataGenerator(partition['validation'], **params) # this should return X,Y where X is for features in the test sample, Y is for the labels in the training sample
        
        #now need to tell it something about the input shape...will FIX THIS when I figure out if I'm giving features_train, features_test and labels_train, labels_test correctly
        ###### End block of skepticism ####
    
        self.__model = self.build()
    
    #model
    def build(self):
        #Maybe need to specify the input the way Thea has inputImage so that it knows the input shape at the start is 8, will do this when I clarify the part I'm skeptical about above
        x = Dense(input_shape=self.input_shape)(<FIX ME, WILL BE HOWEVER I SPECIFY THE FIRST INPUT IN THE LINE ABOVE>)
        #Need to figure out how to turn the batch norm on/off, could do an if statement...like define some self.batch_norm_yes_no = ['1', '0'] #or could go simpler and just always use batch norm
        x=Activation(self.activation[self.activation_index])(x)
        x = Dropout(self.dropout)(x)
        
        for i in range(1,self.DNN_layers): #be careful with numbering here and how you define DNN_layers, you want to make sure you get all of them here except for the input layer and the output layer
            x = Dense(input_shape=self.input_shape)(x) 
            #Need to figure out how to turn the batch norm on/off, also could go simpler and just always have it, same comment as above 
            x = Activation(self.activation[self.activation_index])(x) 
            x = Dropout(self.dropout)(x)
            
        output = Dense(self.n_labels)(x)
        
        model = Model(inputs=<FIX ME>, outputs =output)
        model.summary()
        model.compile(optimizer=self.optimizer[self.optimizer_index], 
                      loss='mean_squared_error', metrics=['acc']) #we know this loss function is not optimal, maybe want to make it a bound...
             
    #fit model
    def model_fit(self):
        self.__model.fit(self.__x_train, self.__y_train, epochs=self.epochs, 
                                   batch_size= self.batch_size, validation_data=[self.__x_test, self.__y_test],verbose=0, 
                                   callbacks = [EarlyStopping(monitor='val_loss', patience=5, verbose=1),
                                                           ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0), 
                                                           TerminateOnNaN()]) #will need to FIX the x and y train when I figure out the great circle of skepticism from above
   
    # evaluate  model
    def model_evaluate(self):
        self.model_fit()
        evaluation = self.__model.evaluate(self.__x_test, self.__y_test, batch_size=self.batch_size, verbose=0) #same comment about the x and y test that I made about the x and y train above FIX
        return evaluation 
        
####################################################

#Runner function for the model

def run_model(optmizer_index, DNN_layers, activation_index, batch_norm_index, DNN_neurons, dropout, batch_size, epochs): #note to self: double check that I got everything in here I need,think I got them and did not put in repeats but always good to double check. Maybe just define an order and stick to it!
    
    _model = myModel(optmizer_index, DNN_layers, activation_index, batch_norm_index, DNN_neurons, dropout, batch_size, epochs)
    
    model_evaluation = _model.model_evaluate()
    return model_evaluation

n_epochs = 500 #Might want to play with this, but let's start here (is there a reason you took this as fixed as opposed to trying to optimize it?)


#Bayesian Optimization
# the bounds dict should be in order of continuous type and then discrete type #QUESTION, you note the order should be continuous then discrete but it does not look like yours follows that...hmm
#might want to try something to mess with the not so great MSE loss function...?

bounds = [{'name': 'optmizer_index',        'type': 'discrete',   'domain': (0, 1, 2)},
          {'name': 'activation_index',  'type': 'discrete',   'domain': (0, 1)},
          {'name': 'batch_norm_index',  'type':  'discrete',   'domain': (0,1)}, #keep this if I figure out how to put it in in a reasonable way                                                                },
          {'name': 'DNN_neurons',           'type': 'discrete',   'domain': (28, 64, 128, 256)},
          {'name': 'DNN_layers',            'type': 'discrete',   'domain': (1, 2, 3, 4, 5)}, #Does up to 5 seem reasonable?
          {'name': 'dropout',               'type': 'continuous', 'domain': (0.0, 0.4)},
          {'name': 'batch_size',            'type': 'discrete',   'domain': (32, 50, 200, 500,1000)}] #just copied what Thea had, maybe should tweak?
          
          
# function to optimize model
def f(x):
    print "x parameters are" #this might need to be in () because python3...
    print(x)
    evaluation = run_model(optmizer_index       = int(x[:,0]), 
                           activation_index     = int(x[:,1]), 
                           batch_norm_index     = int(x[:,2]),
                           DNN_neurons          = int(x[:,3]), 
                           DNN_layers           = int(x[:,4]),
                           dropout              = float(x[:,5]),
                           batch_size           = int(x[:,6]),
                           epochs = n_epochs)
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
             bounds[6]["name"],opt_model.x_opt[6],
             ))
  print("optimized loss: {0}".format(opt_model.fx_opt))