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

#################

#Functions to rotate 4 vectors from global (aka lab frame) coordinates to local coordinates and to unrotate them from the local frame back to the global (aka lab) frame. They take angle (theta and/or phi information) and a four vector and return a four vector.

#################

# Function to rotate the visible tau momentum (defined as the vector sum of the momenta vectors associated with the three charged pions in the decay) to point along the Z axis (aka theta = 0). 
#Takes the original theta and phi values associated with the visible tau momentum and the original visible tau momentum four vector, returns the four vector rotated so the visible tau momentum points along the Z axis.
#My naming could be clearer, as this function can also take the visible tau original theta and phi and the pion and neutrino original four vectors. If it is given this info, it rotates these four vectors to live in the frame in which the visible tau momentum points along the Z axis.

def rotateToVisTauMomPointsAlongZAxis(tau_orig_theta, tau_orig_phi, orig_four_vec_to_rotate): #tau_orig here is the visible tau 
    rotMatrix = np.array([
    
     [np.sin(tau_orig_phi), -np.cos(tau_orig_phi), 0],
     [((np.cos(tau_orig_theta))*(np.cos(tau_orig_phi))), ((np.cos(tau_orig_theta))*(np.sin(tau_orig_phi))), -(np.sin(tau_orig_theta))], 
     [((np.sin(tau_orig_theta))*(np.cos(tau_orig_phi))), (np.sin(tau_orig_theta))*(np.sin(tau_orig_phi)), np.cos(tau_orig_theta)]
     
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


#####################
 
#By applying the rotateToVisibleTauMomPointsAlongZAxis function, we move from the lab frame to the frame in which the visible tau momentum points in the z axis (as the function name implies). 
#Now that we have gotten to this frame, we want to do another rotation, this one in just the xy plane (in which the sum of the momenta of the three charged pi is by definition 0).
#We want to set the phi associated with the pion with the largest pT to be pi (equivalent to -pi) aka we want to have the largest pT pion point along the negative x axis. We will call the highest pT pion in this frame pion 1 (this is basically a definition).  
#Below we define a function to rotate the pion and neutrino vectors to this frame, the rotateToLeadPtPiInVisTauMomPointsAlongZFramePointsAlongNegX function
#This function takes the phi of the lead pT pion in the visible tau momentum points along z frame and the four vector of the particle of interest in the visible tau momentum points along Z axis frame.


def rotateToLeadPtPiInVisTauMomPointsAlongZFramePointsAlongNegX(lead_pt_pi_in_VisTauMomPointsAlongZFrame_phi, four_vec_in_VisTauMomPointsAlongZFrame_to_rotate):
    rotToLeadPtPiPointsAlongNegXMatrix = np.array([
    
    [-(np.cos(lead_pt_pi_in_VisTauMomPointsAlongZFrame_phi)), -(np.sin(lead_pt_pi_in_VisTauMomPointsAlongZFrame_phi)), 0],
    [np.sin(lead_pt_pi_in_VisTauMomPointsAlongZFrame_phi), -(np.cos(lead_pt_pi_in_VisTauMomPointsAlongZFrame_phi)), 0],
    [0, 0, 1],
    ])
    
    #protection to make sure things that really are zero get set to 0 and not 10^-17 or something
    
    for element in np.nditer(rotToLeadPtPiPointsAlongNegXMatrix, op_flags=['readwrite']):
        if abs(element) < 10.**(-10):
            element[...] = 0
    
    tmp_Px =  four_vec_in_VisTauMomPointsAlongZFrame_to_rotate.Px()
    tmp_Py = four_vec_in_VisTauMomPointsAlongZFrame_to_rotate.Py()  
    tmp_Pz = four_vec_in_VisTauMomPointsAlongZFrame_to_rotate.Pz()   
    tmp_E = four_vec_in_VisTauMomPointsAlongZFrame_to_rotate.E()  
    
    tmp_PxPyPz_vec_in_VisTauMomPointsAlongZFrame_to_mult = [[tmp_Px], [tmp_Py], [tmp_Pz]]
    
    tmp_rotated_vec_in_VisTauMomPointsAlongZFrame = np.dot(rotToLeadPtPiPointsAlongNegXMatrix, tmp_PxPyPz_vec_in_VisTauMomPointsAlongZFrame_to_mult)
    
    tmp_rotated_vec_in_VisTauMomPointsAlongZFrame_Px = tmp_rotated_vec_in_VisTauMomPointsAlongZFrame[0]
    tmp_rotated_vec_in_VisTauMomPointsAlongZFrame_Py = tmp_rotated_vec_in_VisTauMomPointsAlongZFrame[1]
    tmp_rotated_vec_in_VisTauMomPointsAlongZFrame_Pz = tmp_rotated_vec_in_VisTauMomPointsAlongZFrame[2]
    
    final_local_4vec = ROOT.TLorentzVector()
    final_local_4vec.SetPxPyPzE(tmp_rotated_vec_in_VisTauMomPointsAlongZFrame_Px,tmp_rotated_vec_in_VisTauMomPointsAlongZFrame_Py, tmp_rotated_vec_in_VisTauMomPointsAlongZFrame_Pz, tmp_E)
    
    return final_local_4vec 
    

#########

def unrotateFromLeadPtPiInVisTauMomPointsAlongZFramePointsAlongNegX(lead_pt_pi_in_VisTauMomPointsAlongZFrame_phi, final_rotated_4vec_to_unrotate):
    pass 
#Function to unrotate the visible tau momentum (defined as the vector sum of the momenta vectors associated with the three charged pions in the decay) from pointing along the Z axis (aka theta = 0) and bring it back to the original lab frame
#Takes the original theta and phi values associated with the visible tau momentum four vector and the rotated visible tau momentum four vector, returns the original tau momentum four vector in the lab frame.
#My naming could be clearer, as this function can also take the visible tau original theta and phi and the pion and neutrino rotated four vectors. If it is given this info, it unrotates these four vectors and brings them back to living in the original lab frame.

def unrotateFromVisTauMomPointsAlongZAxis(tau_orig_theta, tau_orig_phi, rot_four_vec_to_unrotate):  #recall that the inverse of a rotation matrix is its transpose, so unrotMatrix is just the transpose of rotMatrix
    unrotMatrix = np.array([
    
     [np.sin(tau_orig_phi),  ((np.cos(tau_orig_theta))*(np.cos(tau_orig_phi))),  ((np.sin(tau_orig_theta))*(np.cos(tau_orig_phi)))],
     [-np.cos(tau_orig_phi), ((np.cos(tau_orig_theta))*(np.sin(tau_orig_phi))),  ((np.sin(tau_orig_theta))*(np.sin(tau_orig_phi)))], 
     [0,                         -(np.sin(tau_orig_theta)),                                  np.cos(tau_orig_theta)]                                                                                                                        
    
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
#v.SetPxPyPzE(0,0,1,1)
print "Px,Py,Pz,E,M:", v.Px(), v.Py(), v.Pz(), v.E(), v.M()
print "tau_orig_theta, tau_orig_phi:", v.Theta(), v.Phi()
tau_orig_theta_test = v.Theta()
tau_orig_phi_test = v.Phi()



 
toPrint = rotateToVisTauMomPointsAlongZAxis(tau_orig_theta_test, tau_orig_phi_test, v)
 
print toPrint

newPx = toPrint.Px()
newPy = toPrint.Py()
newPz = toPrint.Pz()
newE = toPrint.E()
newM = toPrint.M()
newTheta = toPrint.Theta()
#newPhi = toPrint.Phi()
#newEta = toPrint.Eta()

print "new Px, Py, Pz, E, M, Theta:", newPx, newPy, newPz, newE, newM, newTheta
# 
# v2 = TLorentzVector()
# v2.SetPxPyPzE(8.881784197e-16,0.0,22.1419273613,22.1777103583)
# 
# toPrint2 = unrotateFromVisTauMomPointsAlongZAxis(tau_orig_theta_test, tau_orig_phi_test, v2)
# 
# newPx2 = toPrint2.Px()
# newPy2 = toPrint2.Py()
# newPz2 = toPrint2.Pz()
# newE2 =   toPrint2.E()
# newM2 = toPrint2.M()
# newTheta2 = toPrint2.Theta()
# # # newPhi2   = toPrint2.Phi()
# # # newEta2 = toPrint2.Eta()
# print "new Px2, Py2, Pz2, E, M, Theta2:", newPx2, newPy2, newPz2, newE2, newM2, newTheta2