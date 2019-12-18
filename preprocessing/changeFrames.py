import ROOT
import sys
from DataFormats.FWLite import Events, Handle
from collections import OrderedDict
from array import array
from ROOT import TLorentzVector
import math
import argparse
#from argparse import ArgumentParser
import os
import uproot
import numpy as np
import sys
sys.stdout.flush()


#Functions to rotate 3 vectors from global to local coordinates and to unrotate them from the local frame bacak to the global frame

def rotateToVisTauMomPointsInEtaEqualsZero(tau_orig_theta, tau_orig_phi, orig_four_vec_to_rotate):
    rotMatrix = np.array([[np.sin(tau_orig_phi), -np.cos(tau_orig_phi), 0], [((np.sin(tau_orig_theta))*(np.cos(tau_orig_phi))), ((np.sin(tau_orig_theta))*(np.sin(tau_orig_phi))), np.cos(tau_orig_theta)], [-((np.cos(tau_orig_theta))*(np.cos(tau_orig_phi))), -((np.cos(tau_orig_theta))*(np.sin(tau_orig_phi))), np.sin(tau_orig_theta)]])
    
    # protection to make sure things that really are zero get set to 0 and not 10^-17 or something
    for element in np.nditer(rotMatrix, op_flags=['readwrite']):
        if element < 1.**(-10):
            element[...] = 0
            
    tmp_Px = orig_four_vec_to_rotate.Px()
    tmp_Py = orig_four_vec_to_rotate.Py()
    tmp_Pz = orig_four_vec_to_rotate.Pz()
    tmp_E = orig_four_vec_to_rotate.E() #there is apparently not a convenient SetPxPyPzM method, so we will use SetPxPyPzE down below
   
    tmp_PxPyPz_vec_to_mult = [[tmp_Px], [tmp_Py], [tmp_Pz]]
   
    tmp_rotated_PxPyPz_vec = np.dot(rotMatrix, tmp_PxPyPz_vec_to_mult) #matrix multiplication of rotation matrix times original vector
    
    tmp_rotated_Px = tmp_rotated_PxPyPz_vec[0] #numpy vectors start labelling at 0, so 0th element is the first entry
    tmp_rotated_Py = tmp_rotated_PxPyPz_vec[1]
    tmp_rotated_Pz = tmp_rotated_PxPyPz_vec[2]
    
    local_4vec = ROOT.TLorentzVector()
    local_4vec.SetPxPyPzE(tmp_rotated_Px, tmp_rotated_Py, tmp_rotated_Pz, tmp_E)
    
    return local_4vec

##### test #####

# v = TLorentzVector()
# v.SetPxPyPzE(0,1,0,1)
# print "Px,Py,Pz,E,M:", v.Px(), v.Py(), v.Pz(), v.E(), v.M()
# print "tau_orig_theta, tau_orig_phi:", v.Theta(), v.Phi()
# tau_orig_theta_test = v.Theta()
# tau_orig_phi_test = v.Phi()
# 
#  
# toPrint = rotateToVisTauMomPointsInEtaEqualsZero(tau_orig_theta_test, tau_orig_phi_test, v)
# # 
# print toPrint
# 
# newPx = toPrint.Px()
# newPy = toPrint.Py()
# newPz = toPrint.Pz()
# newE = toPrint.E()
# newM = toPrint.M()
# 
# print "new Px, Py, Pz, E, M:", newPx, newPy, newPz, newE, newM