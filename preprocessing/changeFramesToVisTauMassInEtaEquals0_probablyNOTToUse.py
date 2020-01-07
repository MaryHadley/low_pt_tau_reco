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


#Functions to rotate 4 vectors from global to local coordinates and to unrotate them from the local frame bacak to the global frame. Takes a four vector, returns a four vector.

def rotateToVisTauMomPointsInEtaEqualsZero(tau_orig_theta, tau_orig_phi, orig_four_vec_to_rotate): #tau_orig here is the visible tau 
    rotMatrix = np.array([
    
     [np.sin(tau_orig_phi), -np.cos(tau_orig_phi), 0],
     [((np.sin(tau_orig_theta))*(np.cos(tau_orig_phi))), ((np.sin(tau_orig_theta))*(np.sin(tau_orig_phi))), np.cos(tau_orig_theta)], 
     [-((np.cos(tau_orig_theta))*(np.cos(tau_orig_phi))), -((np.cos(tau_orig_theta))*(np.sin(tau_orig_phi))), np.sin(tau_orig_theta)]
     
     ])
#    print "rotMatrix before is:", rotMatrix

 # protection to make sure things that really are zero get set to 0 and not 10^-17 or something 
    for element in np.nditer(rotMatrix, op_flags=['readwrite']):
        if abs(element) < 10.**(-10):
            element[...] = 0
#    print "rotMatrix after is:", rotMatrix        
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
    
    
def unrotateFromVisTauMomPointsInEtaEqualsZero(tau_orig_theta, tau_orig_phi, rot_four_vec_to_unrotate):  #recall that the inverse of a rotation matrix is its transpose, so unrotMatrix is just the transpose of rotMatrix
    unrotMatrix = np.array([
    
     [np.sin(tau_orig_phi),  ((np.sin(tau_orig_theta))*(np.cos(tau_orig_phi))),  -((np.cos(tau_orig_theta))*(np.cos(tau_orig_phi)))],
     [-np.cos(tau_orig_phi), ((np.sin(tau_orig_theta))*(np.sin(tau_orig_phi))),  -((np.cos(tau_orig_theta))*(np.sin(tau_orig_phi)))], 
     [0,                         np.cos(tau_orig_theta),                                  np.sin(tau_orig_theta)]                                                                                                                        
    
    ])
    
    # protection to make sure things that really are zero get set to 0 and not 10^-17 or something
    
    for unElement in np.nditer(unrotMatrix, op_flags=['readwrite']):
        if abs(unElement) <  10.**(-10):
            unElement[...] = 0
    
    tmp_Px = rot_four_vec_to_unrotate.Px()
    tmp_Py = rot_four_vec_to_unrotate.Py()
    tmp_Pz = rot_four_vec_to_unrotate.Pz()
    tmp_E  = rot_four_vec_to_unrotate.E()  #there is apparently not a convenient SetPxPyPzM method, so we will use SetPxPyPzE down below

    tmp_PxPyPz_vec_to_mult = [[tmp_Px], [tmp_Py], [tmp_Pz]]
    
    tmp_unrotated_PxPyPz_vec = np.dot(unrotMatrix, tmp_PxPyPz_vec_to_mult) #matrix multiplication of unrotMatrix times the rotated vector...the rotated vector is rotMatrix times original vector, therefore what we are doing here is unrotMatrix * rotMatrix * orig vector, which is identity matrix times orig vector, which gives us back orig vector
    
    tmp_unrotated_Px = tmp_unrotated_PxPyPz_vec[0]
    tmp_unrotated_Py = tmp_unrotated_PxPyPz_vec[1]
    tmp_unrotated_Pz = tmp_unrotated_PxPyPz_vec[2]
    
    Global_4vec = ROOT.TLorentzVector()
    Global_4vec.SetPxPyPzE(tmp_unrotated_Px, tmp_unrotated_Py, tmp_unrotated_Pz, tmp_E)
    
    return Global_4vec



    
##### test #####

v = TLorentzVector()
v.SetPxPyPzE(-3.6740152498,-2.79192430698,  21.6557548444, 22.1777103583)
print "Px,Py,Pz,E,M:", v.Px(), v.Py(), v.Pz(), v.E(), v.M()
print "tau_orig_theta, tau_orig_phi:", v.Theta(), v.Phi()
tau_orig_theta_test = v.Theta()
tau_orig_phi_test = v.Phi()



 
toPrint = rotateToVisTauMomPointsInEtaEqualsZero(tau_orig_theta_test, tau_orig_phi_test, v)
# 
print toPrint

newPx = toPrint.Px()
newPy = toPrint.Py()
newPz = toPrint.Pz()
newE = toPrint.E()
newM = toPrint.M()
newTheta = toPrint.Theta()
newPhi = toPrint.Phi()
newEta = toPrint.Eta()

print "new Px, Py, Pz, E, M, Theta, Phi, Eta:", newPx, newPy, newPz, newE, newM, newTheta, newPhi, newEta

v2 = TLorentzVector()
v2.SetPxPyPzE(8.881784197e-16,22.1419273613,0.0,22.1777103583)

toPrint2 = unrotateFromVisTauMomPointsInEtaEqualsZero(tau_orig_theta_test, tau_orig_phi_test, v2)

newPx2 = toPrint2.Px()
newPy2 = toPrint2.Py()
newPz2 = toPrint2.Pz()
newE2 =   toPrint2.E()
newM2 = toPrint2.M()
newTheta2 = toPrint2.Theta()
newPhi2   = toPrint2.Phi()
newEta2 = toPrint2.Eta()
print "new Px2, Py2, Pz2, E, M, Theta2, Phi2, Eta2:", newPx2, newPy2, newPz2, newE2, newM2, newTheta2, newPhi2, newEta2
