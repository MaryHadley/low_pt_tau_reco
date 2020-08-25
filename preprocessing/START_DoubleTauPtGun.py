import ROOT
import sys
from DataFormats.FWLite import Events, Handle
from collections import OrderedDict
from array import array
from ROOT import TLorentzVector
import math
import argparse
import os
import uproot
import numpy as np
import sys
sys.stdout.flush()
from frameChangingUtilities import *

#######################
#####Functions ###
#####################

#Needed function to check the ancestry of particles, inspired by something written Otto Lau based on the CMSSW workbook, with tweaks courtesy of Riju Dasgupta 
def isAncestor(a,p):
    if not p: 
        return False
    
    if a == p: 
        return True
    
    for i in xrange(0, p.numberOfMothers()):
        if isAncestor(a,p.mother(i)): 
            return True
        
    return False

######
    
#Function to find good tau at gen level

def findGoodGenTau(sign):
    flagFoundGoodGenTau = False
    nuList, neuPiList, photList, pi2List, pi1List = [], [], [], [], []
    for tau in gen_taus[sign]:
        flagFoundGoodGenTau = False
        foundNu = 0 
        found2P = 0
        found1P = 0
    
#        if isAncestor(upsilon, tau.mother(0)): #no upsilons in this case!
        for nu in gen_neus[sign]:
            if isAncestor(tau, nu.mother(0)):
                foundNu +=1
                nuList.append(nu)
            
        for pi in gen_pionn:
            if isAncestor(tau, pi.mother(0)):
                neuPiList.append(pi)
                    
                for phot in gen_photons:
                    if isAncestor(pi, phot.mother(0)):
                        photList.append(phot)
           
        for pi in gen_pi2s[sign]:
            if isAncestor(tau, pi.mother(0)):
                found2P += 1
                pi2List.append(pi)
            
        for pi in gen_pi1s[sign]:
            if isAncestor(tau, pi.mother(0)):
                found1P += 1
                pi1List.append(pi)
            
        if foundNu == 1 and found2P == 2 and found1P == 1:
            flagFoundGoodGenTau = True
            #if flagFoundGoodGenTau is True:
                #print "Found good tau with pdgID sign %s" %sign 
                
            return tau, nuList, neuPiList, photList, pi2List, pi1List
    
    return None, nuList, neuPiList, photList, pi2List, pi1List
        

####

###Distance metric as suggested by Markus Seidel 

#N.B. While the distance metric itself -- the myDist -- will always be non-negative by construction, the deltaR and deltaPt can be either negative or positive
def distMetric(genPi, recPi): #takes the gen pi object and the reco pi, returns the distance metric, the deltaR, and the deltaPt, which is the percent difference in pT between the gen and rec object (normalization is the gen pi pT)
    genPi_lv = TLorentzVector()
    genPi_lv.SetPtEtaPhiM(genPi.pt(), genPi.eta(), genPi.phi(), piMass)
    
    recPi_lv = TLorentzVector()
    recPi_lv.SetPtEtaPhiM(recPi.pt(), recPi.eta(), recPi.phi(), piMass)
    
    deltaR = genPi_lv.DeltaR(recPi_lv)
    deltaPt = (genPi_lv.Pt() - recPi_lv.Pt())/genPi_lv.Pt()
    myDist = math.sqrt(deltaR**2 + deltaPt**2)
    return myDist, deltaR, deltaPt
###

## Pi1 matching inspired by discussions with Markus Seidel and Riju Dasgupta 

#N.B. the min_dist returned will always be non-negative by construction, while deltaR and deltaPt can be neither positive or negative 
def matchPi1(sign):
    if len(rec_pi1s[sign]) == 0:
        return None, -float('inf'), -float('inf'), -float('inf')
    dist_list = []
    deltaR_list = []
    deltaPt_list = []
    
    genPi1 = goodEvent_gen_pi1s[sign][0] #Will be one element by construction
   
    for recPi1 in rec_pi1s[sign]: 
        my_dist, my_deltaR, my_deltaPt = distMetric(genPi1, recPi1)
        dist_list.append(my_dist)
        deltaR_list.append(my_deltaR)
        deltaPt_list.append(my_deltaPt)
   
    myPi1DistDict = OrderedDict() 
    for i, distEl in enumerate(dist_list):
        IndexKey = str(i)
        myPi1DistDict[IndexKey] = distEl
    
    myPi1DeltaRDict = OrderedDict()
    for j, deltaREl in enumerate(deltaR_list):
        IndexKey_j = str(j)
        myPi1DeltaRDict[IndexKey_j] = deltaREl 
    
    myPi1DeltaPtDict = OrderedDict()
    for k, deltaPtEl in enumerate(deltaPt_list):
        IndexKey_k = str(k)
        myPi1DeltaPtDict[IndexKey_k] = deltaPtEl
    
    
    min_dist = min(myPi1DistDict.values())
    
    if min_dist > distCutOff_Pi1: #need to define/tune distCutOff_Pi1
        return None, -float('inf'), -float('inf'), -float('inf')
    
    else:
        IndexToSave_List = [key for key in myPi1DistDict if myPi1DistDict[key] == min_dist]
        IndexToSave = IndexToSave_List[0] #If by some miracle there was more than rec pion that was precisely the same distance, just take the first one. Using OrderedDict so this should be stable. Might be overkill...
        return IndexToSave, min_dist, myPi1DeltaRDict[IndexToSave], myPi1DeltaPtDict[IndexToSave]

####

#Pi2 matching inspired by discussion with Markus Seidel and Riju Dasgupta 
##N.B. and WARNING! Again, the min_dist returned will always be non-negative by construction. 
#In this case, since we are constructing the returned values pertaining to the deltaRs and deltaPts as the sqrt of the sum in quadrature of the two deltaR (deltaPt) terms, 
#these returned values will also always be non-negative by construction. This is different from what info gets returned regarding deltaR and deltaPt from the matchPi1 function.


def matchPi2(sign):
    if len(rec_pi2s[sign]) < 2:
        return None, -float('inf'), -float('inf'), -float('inf'), None, None, None, None 
    
    dist_list_for_first_gen_pi2 = []
    dist_list_for_second_gen_pi2 = []
    
    deltaR_list_for_first_gen_pi2 = []
    deltaR_list_for_second_gen_pi2 = []
    
    deltaPt_list_for_first_gen_pi2= []
    deltaPt_list_for_second_gen_pi2 = []
    
    first_gen_pi2 = goodEvent_gen_pi2s[sign][0]
    second_gen_pi2 = goodEvent_gen_pi2s[sign][1]
    
    for recPi2 in rec_pi2s[sign]:
        dist_first_gen_pi2_to_rec_pi, deltaR_first_genpi2_to_rec_pi, deltaPt_first_genpi2_to_rec_pi = distMetric(first_gen_pi2, recPi2)
        dist_list_for_first_gen_pi2.append(dist_first_gen_pi2_to_rec_pi)
        deltaR_list_for_first_gen_pi2.append(deltaR_first_genpi2_to_rec_pi)
        deltaPt_list_for_first_gen_pi2.append(deltaPt_first_genpi2_to_rec_pi)
        
        dist_second_gen_pi2_to_rec_pi, deltaR_second_genpi2_to_rec_pi, deltaPt_second_genpi2_to_rec_pi = distMetric(second_gen_pi2, recPi2)
        dist_list_for_second_gen_pi2.append(dist_second_gen_pi2_to_rec_pi)
        deltaR_list_for_second_gen_pi2.append(deltaR_second_genpi2_to_rec_pi)
        deltaPt_list_for_second_gen_pi2.append(deltaPt_second_genpi2_to_rec_pi)
    
    #print deltaR_list_for_first_gen_pi2
    myPi2DistDict = OrderedDict()   
    for i, x1 in enumerate(dist_list_for_first_gen_pi2):
        for j, x2 in enumerate(dist_list_for_second_gen_pi2):
            if i == j: continue 
            sum = x1 + x2
            IndPairKey = str(i) + '_' + str(j)
            myPi2DistDict[IndPairKey] = sum
    
    #N.B.: myPi2DeltaRDict and myPi2DeltaPtDict are dictionaries of values equal to the sqrt of the sum in quadrature of the deltaRs for the first and second gen pions or the deltaPts for the first and second gen pions, respectively 
    myPi2DeltaRDict = OrderedDict()
    for i, y1 in enumerate(deltaR_list_for_first_gen_pi2):
        for j, y2 in enumerate(deltaR_list_for_second_gen_pi2):
            if i== j: continue
            sqrtdeltaRSumInQuad = math.sqrt(y1**2 + y2**2)
            IndPairKey_deltaR = str(i) + '_' + str(j)
            myPi2DeltaRDict[IndPairKey_deltaR] = sqrtdeltaRSumInQuad
    
    myPi2DeltaPtDict = OrderedDict()
    for i, z1 in enumerate(deltaPt_list_for_first_gen_pi2):
        for j, z2 in enumerate(deltaPt_list_for_second_gen_pi2):
            if i == j: continue
            sqrtdeltaPtSumInQuad = math.sqrt(z1**2 + z2**2)
            IndKeyPair_deltaPt = str(i) + '_' + str(j)
            myPi2DeltaPtDict[IndKeyPair_deltaPt] = sqrtdeltaPtSumInQuad
    
    minDist = min(myPi2DistDict.values())
    
    if minDist > distCutOff_Pi2: #need to define/tune distCufOff_Pi2
       return None, -float('inf'), -float('inf'), -float('inf'), None, None, None, None 
    
    IndPairToSplit_list = [key for key in myPi2DistDict if myPi2DistDict[key] == minDist]
    
    
    
    IndPairToSplit_almost_formatted = IndPairToSplit_list[0] #If by some miracle there was more than rec pion that was precisely the same distance, just take the first one. Using OrderedDict so this should be stable. Might be overkill...
    list_inds_to_save =  IndPairToSplit_almost_formatted.split('_')
#    print "deltaR_list_for_first_gen_pi2 is WOOF:", deltaR_list_for_first_gen_pi2
    return (list_inds_to_save), (minDist), (myPi2DeltaRDict[IndPairToSplit_almost_formatted]), (myPi2DeltaPtDict[IndPairToSplit_almost_formatted]), (deltaR_list_for_first_gen_pi2), deltaR_list_for_second_gen_pi2, deltaPt_list_for_first_gen_pi2, deltaPt_list_for_second_gen_pi2
    
 ######       
    
    
    

        
#CMSSW variable parsing options
from FWCore.ParameterSet.VarParsing import VarParsing
options = VarParsing ('python')
options.register('suffix',
                        '',
                        VarParsing.multiplicity.singleton,
                        VarParsing.varType.string,
                        'suffix to append to out file name')
                        
options.register('excludeTausWithNeutralPiInDecayChain',
                   '',
                   VarParsing.multiplicity.singleton,
                   VarParsing.varType.int,
                   'Decide whether to exclude taus that have neutral pions in the decay chain or not. Use 1 for exclude is True, 0 for exclude is False'
                   
                   )

options.register('tuneCutParameters',
                  0, #default is 0, or off
                  VarParsing.multiplicity.singleton,
                   VarParsing.varType.int,
                   'Decide whether to set distCutOff_Pi1 and distCutOff_Pi2 to float inf as one wants to when making plots to tune these parameters. Usually one will want this off, this is a sort of DEBUG mode'
                   )
options.parseArguments()

print options

#Getting the collections from CMSSW
handlePruned  = Handle ("std::vector<reco::GenParticle>")
labelPruned = ("prunedGenParticles")

handleReco = Handle ("std::vector<pat::PackedCandidate>")
recoLabel = ("packedPFCandidates")

lostLabel = ("lostTracks")

handleMET = Handle ("std::vector<pat::MET>")
labelMET = ("slimmedMETs")

#Branches 
taum_branches =[
'pi_minus1_pt',
'pi_minus1_eta',
'pi_minus1_phi',
'pi_minus1_theta',
'pi_minus2_pt',
'pi_minus2_eta',
'pi_minus2_phi',
'pi_minus2_theta',
'pi_minus3_pt',
'pi_minus3_eta',
'pi_minus3_phi',
'pi_minus3_theta',
'taum_pt',
'taum_eta',
'taum_phi',
'taum_theta',
'taum_mass'
]

taup_branches =[
 'pi_plus1_pt',
 'pi_plus1_eta',
 'pi_plus1_phi',
 'pi_plus1_theta',
 'pi_plus2_pt',
 'pi_plus2_eta',
 'pi_plus2_phi',
 'pi_plus2_theta',
 'pi_plus3_pt',
 'pi_plus3_eta',
 'pi_plus3_phi',
 'pi_plus3_theta',
 'taup_pt',
 'taup_eta',
 'taup_phi',
 'taup_theta',
 'taup_mass'
]

branches = taum_branches + taup_branches

branches.append('upsilon_mass')
branches.append("upsilon_pt")
branches.append("upsilon_eta")
branches.append("upsilon_phi")
branches.append("upsilon_theta")
branches.append('neutrino_pt')
branches.append('neutrino_phi')
branches.append('neutrino_eta')
branches.append('neutrino_theta')
branches.append('antineutrino_pt')
branches.append('antineutrino_phi')
branches.append('antineutrino_eta')
branches.append("antineutrino_theta")

#branches.append('vis_taum_eta')

#things to save to do the rotations and unrotations
branches.append('orig_vis_taum_phi') 
branches.append('orig_vis_taum_theta')
branches.append("orig_vis_taup_phi")
branches.append("orig_vis_taup_theta")

#branches.append('local_pi_m_lv1_phi')
#branches.append('local_pi_m_lv1_eta')
branches.append('local_pi_m_lv1_pt')
branches.append('local_pi_m_lv2_pt')
branches.append('local_pi_m_lv3_pt')
branches.append('local_pi_m_lv1_mass')
branches.append("local_pi_m_lv2_mass")
branches.append("local_pi_m_lv3_mass")

branches.append("local_pi_p_lv1_pt")
branches.append("local_pi_p_lv2_pt")
branches.append("local_pi_p_lv3_pt")

branches.append('local_taum_lv_mass')
branches.append("local_taup_lv_mass")

#these local branches are all actually sanity check branches and strictly speaking do not need to be filled, they were just useful while I was writing the code to do intermediate tests

#another two things to save to do the rotations and unrotations
branches.append("initial_leadPt_pi_m_in_AllInZFrame_phi")
branches.append("initial_leadPt_pi_p_in_AllInZFrame_phi")
####

#things to use with the DNN in the toUse frame: for every pion and neutrino, we need toUse_pt, toUse_theta, toUse_phi
branches.append("toUse_local_taum_lv_mass")
branches.append("toUse_local_taup_lv_mass")
#toUse phi info
branches.append("toUse_local_pi_m_lv1_phi") #always pi
branches.append("toUse_local_pi_p_lv1_phi") #always pi

#toUse pT info
branches.append("toUse_local_pi_m_lv1_pt")
branches.append("toUse_local_pi_m_lv2_pt")
branches.append("toUse_local_pi_m_lv3_pt")
branches.append("toUse_local_neu_lv_pt")
#new label
branches.append("toUse_local_neu_lv_pt_norm_by_tauMass")

branches.append("toUse_local_pi_p_lv1_pt")
branches.append("toUse_local_pi_p_lv2_pt")
branches.append("toUse_local_pi_p_lv3_pt")
branches.append("toUse_local_antineu_lv_pt")
#new label
branches.append("toUse_local_antineu_lv_pt_norm_by_tauMass")

#toUse theta info

branches.append("toUse_local_pi_m_lv1_theta")
branches.append("toUse_local_pi_m_lv2_theta")
branches.append("toUse_local_pi_m_lv3_theta")
branches.append("toUse_local_neu_lv_theta")

branches.append("toUse_local_pi_p_lv1_theta")
branches.append("toUse_local_pi_p_lv2_theta")
branches.append("toUse_local_pi_p_lv3_theta")
branches.append("toUse_local_antineu_lv_theta")

branches.append("toUse_local_pi_m_lv2_phi")
branches.append("toUse_local_pi_m_lv3_phi")

branches.append("toUse_local_pi_p_lv2_phi")
branches.append("toUse_local_pi_p_lv3_phi")

branches.append("toUse_local_neu_lv_phi") # will not apply the get_toUse_local_phi to this because we do not know  these nu/antinu phis should be within [-pi/2, pi/2]
branches.append("toUse_local_antineu_lv_phi")

branches.append("check1_mass")
branches.append("check2_mass")
#also sanity check branches

# branches.append("naive_upsilon_lv_mass")
# branches.append("global_naive_upsilon_lv_mass")

branches.append("check_upsilon_mass")
#also sanity check branch

# branches.append("tau_true_mom_mag")
# branches.append("naive_tau_mom_mag")
# branches.append("antitau_true_mom_mag")
# branches.append("naive_antitau_mom_mag")
# 
# branches.append("diff_true_minus_naive_antitau_mom_mag")
# branches.append("diff_true_minus_naive_tau_mom_mag")

#Jan idea 
# branches.append("vis_ditau_px")
# branches.append("vis_ditau_py")
# branches.append("vis_ditau_pz")
# 
# branches.append("true_ditau_px")
# branches.append("true_ditau_py")
# branches.append("true_ditau_pz")
# 
# branches.append("SFx")
# branches.append("SFy")
# branches.append("SFz")

#branches to help tune parameters
branches.append("candMatchPi1Info_tau_pdgID_plus_dist")
branches.append("candMatchPi1Info_tau_pdgID_minus_dist")
branches.append('candMatchPi2Info_tau_pdgID_plus_dist')
branches.append('candMatchPi2Info_tau_pdgID_minus_dist')
branches.append('candMatchPi1Info_tau_pdgID_plus_deltaR')
branches.append('candMatchPi1Info_tau_pdgID_minus_deltaR')
branches.append('candMatchPi1Info_tau_pdgID_plus_deltaPt')
branches.append('candMatchPi1Info_tau_pdgID_minus_deltaPt')
branches.append('candMatchPi2Info_tau_pdgID_plus_deltaR')
branches.append('candMatchPi2Info_tau_pdgID_minus_deltaR')
branches.append('candMatchPi2Info_tau_pdgID_plus_deltaPt') 
branches.append('candMatchPi2Info_tau_pdgID_minus_deltaPt') 

branches.append('taum_charge') #should be -1, charge of the tau
branches.append('taup_charge') # should be +1, charge of antitau 

#all 3 pi associated with either the tau plus or the tau minus in lab frame sorted from highest to lowest pt
branches.append('sorted_in_pt_pi_lv1_pt_p')
branches.append('sorted_in_pt_pi_lv2_pt_p')
branches.append('sorted_in_pt_pi_lv3_pt_p')
branches.append('sorted_in_pt_pi_lv1_pt_m')
branches.append('sorted_in_pt_pi_lv2_pt_m')
branches.append('sorted_in_pt_pi_lv3_pt_m')

branches.append('sorted_in_pt_pi_lv1_pt_norm_by_tauMass_p')
branches.append('sorted_in_pt_pi_lv2_pt_norm_by_tauMass_p')
branches.append('sorted_in_pt_pi_lv3_pt_norm_by_tauMass_p')
branches.append('sorted_in_pt_pi_lv1_pt_norm_by_tauMass_m')
branches.append('sorted_in_pt_pi_lv2_pt_norm_by_tauMass_m')
branches.append('sorted_in_pt_pi_lv3_pt_norm_by_tauMass_m')

branches.append('sorted_in_pt_pi_lv1_eta_p')
branches.append('sorted_in_pt_pi_lv2_eta_p')
branches.append('sorted_in_pt_pi_lv3_eta_p')
branches.append('sorted_in_pt_pi_lv1_eta_m')
branches.append('sorted_in_pt_pi_lv2_eta_m')
branches.append('sorted_in_pt_pi_lv3_eta_m')


branches.append('sorted_in_pt_pi_lv1_theta_p')
branches.append('sorted_in_pt_pi_lv2_theta_p')
branches.append('sorted_in_pt_pi_lv3_theta_p')
branches.append('sorted_in_pt_pi_lv1_theta_m')
branches.append('sorted_in_pt_pi_lv2_theta_m')
branches.append('sorted_in_pt_pi_lv3_theta_m')

branches.append('sorted_in_pt_pi_lv1_phi_p')
branches.append('sorted_in_pt_pi_lv2_phi_p')
branches.append('sorted_in_pt_pi_lv3_phi_p')
branches.append('sorted_in_pt_pi_lv1_phi_m')
branches.append('sorted_in_pt_pi_lv2_phi_m')
branches.append('sorted_in_pt_pi_lv3_phi_m')

#Stuff to try with interaction net
branches.append("sorted_in_pt_pi_lv1_px_p")
branches.append("sorted_in_pt_pi_lv1_py_p")
branches.append("sorted_in_pt_pi_lv1_pz_p")
branches.append("sorted_in_pt_pi_lv1_E_p")
branches.append("sorted_in_pt_pi_lv1_mass_p")

branches.append("sorted_in_pt_pi_lv2_px_p")
branches.append("sorted_in_pt_pi_lv2_py_p")
branches.append("sorted_in_pt_pi_lv2_pz_p")
branches.append("sorted_in_pt_pi_lv2_E_p")
branches.append("sorted_in_pt_pi_lv2_mass_p")

branches.append("sorted_in_pt_pi_lv3_px_p")
branches.append("sorted_in_pt_pi_lv3_py_p")
branches.append("sorted_in_pt_pi_lv3_pz_p")
branches.append("sorted_in_pt_pi_lv3_E_p")
branches.append("sorted_in_pt_pi_lv3_mass_p")

branches.append("sorted_in_pt_pi_lv1_px_m")
branches.append("sorted_in_pt_pi_lv1_py_m")
branches.append("sorted_in_pt_pi_lv1_pz_m")
branches.append("sorted_in_pt_pi_lv1_E_m")
branches.append("sorted_in_pt_pi_lv1_mass_m")

branches.append("sorted_in_pt_pi_lv2_px_m")
branches.append("sorted_in_pt_pi_lv2_py_m")
branches.append("sorted_in_pt_pi_lv2_pz_m")
branches.append("sorted_in_pt_pi_lv2_E_m")
branches.append("sorted_in_pt_pi_lv2_mass_m")

branches.append("sorted_in_pt_pi_lv3_px_m")
branches.append("sorted_in_pt_pi_lv3_py_m")
branches.append("sorted_in_pt_pi_lv3_pz_m")
branches.append("sorted_in_pt_pi_lv3_E_m")
branches.append("sorted_in_pt_pi_lv3_mass_m")





#just normalized, but no sorting in lab frame
branches.append('pi_pt_lv1_norm_by_tauMass_p')
branches.append('pi_pt_lv2_norm_by_tauMass_p')
branches.append('pi_pt_lv3_norm_by_tauMass_p')
branches.append('pi_pt_lv1_norm_by_tauMass_m')
branches.append('pi_pt_lv2_norm_by_tauMass_m')
branches.append('pi_pt_lv3_norm_by_tauMass_m')

#branches.append('len_neuPiList_tau_pdgID_plus')

#local taup and taum stuff #Recall that taup and taum in local frame already got sorted in pT in that frame, with lv1 being the highest in pT

#oUse_local_pi_p_lv stuff 
branches.append("toUse_local_pi_p_lv1_px")
branches.append("toUse_local_pi_p_lv1_py")
branches.append("toUse_local_pi_p_lv1_pz")
branches.append("toUse_local_pi_p_lv1_E")
branches.append("toUse_local_pi_p_lv1_mass")

branches.append("toUse_local_pi_p_lv2_px")
branches.append("toUse_local_pi_p_lv2_py")
branches.append("toUse_local_pi_p_lv2_pz")
branches.append("toUse_local_pi_p_lv2_E")
branches.append("toUse_local_pi_p_lv2_mass")

branches.append("toUse_local_pi_p_lv3_px")
branches.append("toUse_local_pi_p_lv3_py")
branches.append("toUse_local_pi_p_lv3_pz")
branches.append("toUse_local_pi_p_lv3_E")
branches.append("toUse_local_pi_p_lv3_mass")

#toUse_local_pi_m_lv stuff
#Ok this stuff all works if I want to fill in a loop, but I think I don't want to change the syntax of having lowercase px, py, pz, so I am not going to do it this way
# branches.append("toUse_local_pi_m_lv1_Px")
# branches.append("toUse_local_pi_m_lv1_Py")
# branches.append("toUse_local_pi_m_lv1_Pz")
# branches.append("toUse_local_pi_m_lv1_E")
# branches.append("toUse_local_pi_m_lv1_mass")
# 
# branches.append("toUse_local_pi_m_lv2_Px")
# branches.append("toUse_local_pi_m_lv2_Py")
# branches.append("toUse_local_pi_m_lv2_Pz")
# branches.append("toUse_local_pi_m_lv2_E")
# branches.append("toUse_local_pi_m_lv2_mass")
# 
# branches.append("toUse_local_pi_m_lv3_Px")
# branches.append("toUse_local_pi_m_lv3_Py")
# branches.append("toUse_local_pi_m_lv3_Pz")
# branches.append("toUse_local_pi_m_lv3_E")
# branches.append("toUse_local_pi_m_lv3_mass")

#Ok, this keeps the px, py, pz uniformly lowercase, which I might want later down the line, and using lower(), I can keep the px, py, pz naming and still use a loop. However because of wanting to avoid locals()  I eventually decided that using the loop was more effort than it was worth and I just filled the old boring naive way. 
branches.append("toUse_local_pi_m_lv1_px")
branches.append("toUse_local_pi_m_lv1_py")
branches.append("toUse_local_pi_m_lv1_pz")
branches.append("toUse_local_pi_m_lv1_E")
branches.append("toUse_local_pi_m_lv1_mass")

branches.append("toUse_local_pi_m_lv2_px")
branches.append("toUse_local_pi_m_lv2_py")
branches.append("toUse_local_pi_m_lv2_pz")
branches.append("toUse_local_pi_m_lv2_E")
branches.append("toUse_local_pi_m_lv2_mass")

branches.append("toUse_local_pi_m_lv3_px")
branches.append("toUse_local_pi_m_lv3_py")
branches.append("toUse_local_pi_m_lv3_pz")
branches.append("toUse_local_pi_m_lv3_E")
branches.append("toUse_local_pi_m_lv3_mass")

#In global frame 

#pdgID
branches.append("pi_plus_sorted_in_pt_1_pdgID")
branches.append("pi_plus_sorted_in_pt_2_pdgID")
branches.append("pi_plus_sorted_in_pt_3_pdgID")
branches.append("pi_minus_sorted_in_pt_1_pdgID")
branches.append("pi_minus_sorted_in_pt_2_pdgID")
branches.append("pi_minus_sorted_in_pt_3_pdgID")

#charge
branches.append("pi_plus_sorted_in_pt_1_charge")
branches.append("pi_plus_sorted_in_pt_2_charge")
branches.append("pi_plus_sorted_in_pt_3_charge")
branches.append("pi_minus_sorted_in_pt_1_charge")
branches.append("pi_minus_sorted_in_pt_2_charge")
branches.append("pi_minus_sorted_in_pt_3_charge")

#HCAL Fraction
branches.append("pi_plus_sorted_in_pt_1_hcalFraction")
branches.append("pi_plus_sorted_in_pt_2_hcalFraction")
branches.append("pi_plus_sorted_in_pt_3_hcalFraction")
branches.append("pi_minus_sorted_in_pt_1_hcalFraction")
branches.append("pi_minus_sorted_in_pt_2_hcalFraction")
branches.append("pi_minus_sorted_in_pt_3_hcalFraction")

#Phi at vertex
branches.append("pi_plus_sorted_in_pt_1_phiAtVtx")
branches.append("pi_plus_sorted_in_pt_2_phiAtVtx")
branches.append("pi_plus_sorted_in_pt_3_phiAtVtx")
branches.append("pi_minus_sorted_in_pt_1_phiAtVtx")
branches.append("pi_minus_sorted_in_pt_2_phiAtVtx")
branches.append("pi_minus_sorted_in_pt_3_phiAtVtx")

#diff between phiAtVtx and phi
branches.append("pi_plus_sorted_in_pt_1_phiAtVtxMinusPhi")
branches.append("pi_plus_sorted_in_pt_2_phiAtVtxMinusPhi")
branches.append("pi_plus_sorted_in_pt_3_phiAtVtxMinusPhi")
branches.append("pi_minus_sorted_in_pt_1_phiAtVtxMinusPhi")
branches.append("pi_minus_sorted_in_pt_2_phiAtVtxMinusPhi")
branches.append("pi_minus_sorted_in_pt_3_phiAtVtxMinusPhi")

#Not useful for particle gun I don't think 
# vertexRef
# branches.append("pi_plus_sorted_in_pt_1_vertexRef")
# branches.append("pi_plus_sorted_in_pt_2_vertexRef")
# branches.append("pi_plus_sorted_in_pt_3_vertexRef")
# branches.append("pi_minus_sorted_in_pt_1_vertexRef")
# branches.append("pi_minus_sorted_in_pt_2_vertexRef")
# branches.append("pi_minus_sorted_in_pt_3_vertexRef")


#In local frame

#pdgId
branches.append("pi_plus_sorted_in_local_pt_1_pdgID")
branches.append("pi_plus_sorted_in_local_pt_2_pdgID")
branches.append("pi_plus_sorted_in_local_pt_3_pdgID")
branches.append("pi_minus_sorted_in_local_pt_1_pdgID")
branches.append("pi_minus_sorted_in_local_pt_2_pdgID")
branches.append("pi_minus_sorted_in_local_pt_3_pdgID")

#charge
branches.append("pi_plus_sorted_in_local_pt_1_charge")
branches.append("pi_plus_sorted_in_local_pt_2_charge")
branches.append("pi_plus_sorted_in_local_pt_3_charge")
branches.append("pi_minus_sorted_in_local_pt_1_charge")
branches.append("pi_minus_sorted_in_local_pt_2_charge")
branches.append("pi_minus_sorted_in_local_pt_3_charge")

#HCAL Fraction
branches.append("pi_plus_sorted_in_local_pt_1_hcalFraction")
branches.append("pi_plus_sorted_in_local_pt_2_hcalFraction")
branches.append("pi_plus_sorted_in_local_pt_3_hcalFraction")
branches.append("pi_minus_sorted_in_local_pt_1_hcalFraction")
branches.append("pi_minus_sorted_in_local_pt_2_hcalFraction")
branches.append("pi_minus_sorted_in_local_pt_3_hcalFraction")

branches.append("pi_plus_sorted_in_local_pt_1_phiAtVtx")
branches.append("pi_plus_sorted_in_local_pt_2_phiAtVtx")
branches.append("pi_plus_sorted_in_local_pt_3_phiAtVtx")
branches.append("pi_minus_sorted_in_local_pt_1_phiAtVtx")
branches.append("pi_minus_sorted_in_local_pt_2_phiAtVtx")
branches.append("pi_minus_sorted_in_local_pt_3_phiAtVtx")

#diff between phiAtVtx and phi
branches.append("pi_plus_sorted_in_local_pt_1_phiAtVtxMinusPhi")
branches.append("pi_plus_sorted_in_local_pt_2_phiAtVtxMinusPhi")
branches.append("pi_plus_sorted_in_local_pt_3_phiAtVtxMinusPhi")
branches.append("pi_minus_sorted_in_local_pt_1_phiAtVtxMinusPhi")
branches.append("pi_minus_sorted_in_local_pt_2_phiAtVtxMinusPhi")
branches.append("pi_minus_sorted_in_local_pt_3_phiAtVtxMinusPhi")


#unrotation matrix elements as defined in email with scanned document from 23 aout 2020
branches.append("m00_taup")
branches.append("m01_taup")
branches.append("m02_taup")
branches.append("m10_taup")
branches.append("m11_taup")
branches.append("m12_taup")
branches.append("m20_taup")
branches.append("m21_taup")
branches.append("m22_taup")

branches.append("m00_taum")
branches.append("m01_taum")
branches.append("m02_taum")
branches.append("m10_taum")
branches.append("m11_taum")
branches.append("m12_taum")
branches.append("m20_taum")
branches.append("m21_taum")
branches.append("m22_taum")

#input vector elements

branches.append("in_00_antinu") #for use with taup
branches.append("in_10_antinu")
branches.append("in_20_antinu")

branches.append("in_00_nu") #for use with taum
branches.append("in_10_nu")
branches.append("in_20_nu")

#output vector elements

branches.append("out_00_antinu")
branches.append("out_10_antinu")
branches.append("out_20_antinu")

branches.append("out_00_nu")
branches.append("out_10_nu")
branches.append("out_20_nu")

branches.append("first_eq_antinu")
branches.append("second_eq_antinu")
branches.append("third_eq_antinu")

branches.append("first_eq_nu")
branches.append("second_eq_nu")
branches.append("third_eq_nu")




### End of long list of branches

###Efficiency histogram definitions
lowEdge_gen_tau_pt_hists_0to30pt = 0
highEdge_gen_tau_pt_hists_0to30pt = 30
nBins_gen_tau_pt_hists_0to30pt = 15

h_den_gen_tau_pt_all_pdgID_plus_0to30pt = ROOT.TH1F("h_den_gen_tau_pt_all_pdgID_plus_0to30pt", "", nBins_gen_tau_pt_hists_0to30pt, lowEdge_gen_tau_pt_hists_0to30pt, highEdge_gen_tau_pt_hists_0to30pt)
h_num_gen_tau_pt_matched_pdgID_plus_0to30pt = ROOT.TH1F("h_num_gen_tau_pt_matched_pdgID_plus_0to30pt", "", nBins_gen_tau_pt_hists_0to30pt, lowEdge_gen_tau_pt_hists_0to30pt, highEdge_gen_tau_pt_hists_0to30pt)

h_den_gen_tau_pt_all_pdgID_minus_0to30pt = ROOT.TH1F("h_den_gen_tau_pt_all_pdgID_minus_0to30pt", "", nBins_gen_tau_pt_hists_0to30pt, lowEdge_gen_tau_pt_hists_0to30pt, highEdge_gen_tau_pt_hists_0to30pt)
h_num_gen_tau_pt_matched_pdgID_minus_0to30pt = ROOT.TH1F("h_num_gen_tau_pt_matched_pdgID_minus_0to30pt", "", nBins_gen_tau_pt_hists_0to30pt, lowEdge_gen_tau_pt_hists_0to30pt, highEdge_gen_tau_pt_hists_0to30pt)

lowEdge_gen_tau_pt_hists_30to50pt = 30
highEdge_gen_tau_pt_hists_30to50pt = 50
nBins_gen_tau_pt_hists_30to50pt = 2



h_den_gen_tau_pt_all_pdgID_plus_30to50pt = ROOT.TH1F("h_den_gen_tau_pt_all_pdgID_plus_30to50pt", "", nBins_gen_tau_pt_hists_30to50pt, lowEdge_gen_tau_pt_hists_30to50pt, highEdge_gen_tau_pt_hists_30to50pt)
h_num_gen_tau_pt_matched_pdgID_plus_30to50pt = ROOT.TH1F("h_num_gen_tau_pt_matched_pdgID_plus_30to50pt", "", nBins_gen_tau_pt_hists_30to50pt, lowEdge_gen_tau_pt_hists_30to50pt, highEdge_gen_tau_pt_hists_30to50pt)

h_den_gen_tau_pt_all_pdgID_minus_30to50pt = ROOT.TH1F("h_den_gen_tau_pt_all_pdgID_minus_30to50pt", "", nBins_gen_tau_pt_hists_30to50pt, lowEdge_gen_tau_pt_hists_30to50pt, highEdge_gen_tau_pt_hists_30to50pt)
h_num_gen_tau_pt_matched_pdgID_minus_30to50pt = ROOT.TH1F("h_num_gen_tau_pt_matched_pdgID_minus_30to50pt", "", nBins_gen_tau_pt_hists_30to50pt, lowEdge_gen_tau_pt_hists_30to50pt, highEdge_gen_tau_pt_hists_30to50pt)




suffix = options.suffix
excludeTausWithNeutralPiInDecayChain = options.excludeTausWithNeutralPiInDecayChain
print 'suffix is:', suffix
print "excludeTausWithNeutralPiInDecayChain is:", excludeTausWithNeutralPiInDecayChain
tuneCutParameters = options.tuneCutParameters
print "tuneCutParameters is:", tuneCutParameters

if excludeTausWithNeutralPiInDecayChain == 1:
    print "excludeTausWithNeutralPiInDecayChain is %s. Events with taus with neutral pions in the decay chain at gen level will NOT be considered." %(excludeTausWithNeutralPiInDecayChain)
elif excludeTausWithNeutralPiInDecayChain == 0:
#    print "excludeTausWithNeutralPiInDecayChain is %s. Events with taus with neutral pions in the decay chain at gen level WILL be considered. A pi pT cut will be applied at reco level to help recover tau mass peak." %(excludeTausWithNeutralPiInDecayChain)
    raise Exception("excludeTausWithNeutralPiInDecayChain is %s. Events with taus with neutral pions in the decay chain at gen level WILL be considered. A pi pT cut will be applied at reco level to help recover tau mass peak. THIS OPTION IS NOT YET ENABLED. Currently only the mode where we exclude taus with neutral pi in the decay chain is possible. Please try again and set the excludeTausWithNeutralPiInDecayChain option to 1." %(excludeTausWithNeutralPiInDecayChain))
else:
    raise Exception("Please specify whether you want to excludeTausWithNeutralPiInDecayChain -- to do so, set excludeTausWithNeutralPiInDecayChain=1 at the command line -- or include them -- to do so, set excludeTausWithNeutralPiInDecayChain = 0 at the command line.")


if tuneCutParameters == 0:
    print "Running in the default mode (tuneCutParameters == 0), you have settled on your tunable parameters and are happy with them. Branches used to tune these parameters will be filled with DEFAULT values and will NOT be meaningful. The efficiency plots WILL be meaningful."
else:
    print "Running in mode where the distCutOffs have been set to be infinite (tuneCutParameters == 1). Branches used to tune these parameters will be filled with MEANINGFUL values. The efficiency plots will all show an efficiency of 1 by construction and so are NOT particularly meaningful."

   

file_out = ROOT.TFile('cartesian_upsilon_taus_%s.root'%(suffix), 'recreate')
file_out.cd()

ntuple = ROOT.TNtuple('tree', 'tree', ':'.join(branches))


#Some pdgID IDs

upsilon_id = 553
tau_id = 15
pion_id = 211
tau_neu_id = 16
neutral_pion_id = 111
photon_id = 22


# piPtCut settings

#piPtCut = 0.35
piPtCut = 0
#piPtCut = 0.7

#etaCut
etaCut = 2.5

#Mass constants (in GeV)
piMass = 0.13957 
nuMass = 0
tauMass = 1.777

#Distance Metric Cuts

if tuneCutParameters:
    distCutOff_Pi1 = float('inf') #for sanity checking and tuning code
    distCutOff_Pi2 = float('inf') #for sanity checking and tuning code
else:
    distCutOff_Pi1 = 0.1
    distCutOff_Pi2 = 0.2 #What I think we will go with 



# use Varparsing object
events = Events (options)
print "events are:", events

#Counters
nTot = 0
eventHasGoodGenUpsilonCount = 0
eventDoesNOTHaveGoodGenUpsilonCount = 0
eventHasGenPiOutsideEtaAcceptCount = 0
eventHasMatchedUpsilonCount = 0
tau_pdgID_plus_has_neuPiCount = 0
tau_pdgID_minus_has_neuPiCount = 0
eventHasTauWithNeutralPiInDecayChainCount = 0



for event in events:
    nTot += 1
    eventHasGoodGenUpsilon = False 
    eventHasMatchedPi1_for_tau_pdgID_plus = False
    eventHasMatchedPi1_for_tau_pdgID_minus = False 
    eventHasMatchedPi2_for_tau_pdgID_plus = False
    eventHasMatchedPi2_for_tau_pdgID_minus = False 
    eventHasMatchedTau_for_pdgID_plus = False
    eventHasMatchedTau_for_pdgID_minus = False 
    eventHasMatchedUpsilon = False 

    
    print 'Processing event: %i...'%(nTot)
    
    # Generated stuff
    event.getByLabel(labelPruned, handlePruned)
    pruned_gen_particles = handlePruned.product()

    event.getByLabel(recoLabel, handleReco)
    pf_Particles = handleReco.product()

    event.getByLabel(lostLabel, handleReco)
    lost_Particles = handleReco.product()
    
    event.getByLabel(labelMET, handleMET)
    met = handleMET.product().front()

    reco_Particles = []

    for p in pf_Particles:
        reco_Particles.append(p)
    for p in lost_Particles:
        reco_Particles.append(p)



    gen_upsilon = []
    gen_taum = []
    gen_taup = []
    gen_pionm = []
    gen_pionp = []
    gen_neu = []
    gen_anti_neu = []
    gen_pionn = []
    gen_photons = []


    matched_pionp = []
    matched_pionm = []
    matched_photonp = []
    matched_photonm = []

    lost_pions = []
    taum_has_pionn = False #Might end up getting rid of this
    taup_has_pionn = False #Might end up getting rid of this

    # Filter reco particles
    rec_pionm = [] #this is a list of rec pions with pdgID -211, pdgID -211 refers to negatively charged pions
    rec_pionp = [] #this is a list of rec pions with pdgID + 211, pdg + 211 refers to positively charged pions
#    rec_pions = []
    rec_photons = []
    
     # Tagging particles in gen particles 
     #Note that the sign conventions for what is plus and what is minus have to do with the sign on the pdgId
     #Therefore the tau, which has charge -1, is called the gen_taup, because it has a positive pdgID
     #The antitau, which has charge +1, is called the gen_taum, because it has a negative pdgID
     #Positively charged pion has pdgID + 211, called gen_pionp
     #Negatively charged pion had pdgID -211, called gen_pionm 
    for pp in pruned_gen_particles:
        if abs(pp.pdgId()) == upsilon_id:
            gen_upsilon.append(pp)
        elif pp.pdgId() == - tau_id:
            gen_taum.append(pp)
        elif pp.pdgId() == tau_id:
            gen_taup.append(pp)
        elif pp.pdgId() == - pion_id:
            gen_pionm.append(pp)
        elif pp.pdgId() == pion_id:
            gen_pionp.append(pp)
        elif pp.pdgId() == tau_neu_id:
            gen_neu.append(pp)
        elif pp.pdgId() == - tau_neu_id:
            gen_anti_neu.append(pp)
        elif pp.pdgId() == neutral_pion_id:
            gen_pionn.append(pp)
        elif pp.pdgId() == photon_id:
            gen_photons.append(pp)

        #note that these are filled once per event, so this is not all the taus associated with an upsilon necessarily, but just...all the taus in the event

        # Tagging reco particles
    for pp in reco_Particles:
        if pp.pdgId() == pion_id:
            rec_pionp.append(pp)
        elif pp.pdgId() == - pion_id:
            rec_pionm.append(pp)
        elif abs(pp.pdgId()) == photon_id:
            rec_photons.append(pp)

    for pp in lost_Particles:
        if abs(pp.pdgId()) == pion_id:
            lost_pions.append(pp) #May get rid of this lost_pions, does not get used
    
    
    gen_taus = {'+':gen_taup , '-':gen_taum    }
    gen_neus = {'+':gen_neu  , '-':gen_anti_neu}
    gen_pi1s = {'+':gen_pionp, '-':gen_pionm   }
    gen_pi2s = {'+':gen_pionm, '-':gen_pionp   }
    
    rec_pi1s = {'+': rec_pionp , '-':rec_pionm } 
    rec_pi2s = {'+': rec_pionm , '-':rec_pionp }
    
#    print "rec_pi1s", rec_pi1s
    
#    print "len(gen_upsillon) is:", len(gen_upsilon)
#    print "len(gen_taup is:", len(gen_taup)
#    print "len(gen_taum is:", len(gen_taum)
#    print "len(rec_pionp) is:", len(rec_pionp)

    
    tau_pdgID_plus, nuList_tau_pdgID_plus, neuPiList_tau_pdgID_plus, photList_tau_pdgID_plus, pi2List_tau_pdgID_plus, pi1List_tau_pdgID_plus = findGoodGenTau('+')
    tau_pdgID_minus, nuList_tau_pdgID_minus, neuPiList_tau_pdgID_minus, photList_tau_pdgID_minus, pi2List_tau_pdgID_minus, pi1List_tau_pdgID_minus = findGoodGenTau('-')
    if tau_pdgID_plus is not None and tau_pdgID_minus is not None:
            eventHasGoodGenUpsilon = True #eventHasGoodGenUpsilon in this code is just shorthand to indicate both a tau and an antitau were found in the event, obviously there are no physical upsilons in a pT gun sample
            if eventHasGoodGenUpsilon:
                eventHasGoodGenUpsilonCount += 1
                #break
    if not eventHasGoodGenUpsilon:
        eventDoesNOTHaveGoodGenUpsilonCount +=1
        continue #there is not a good gen upsilon in this event, let us not spend any further time on it!
    
    if len(neuPiList_tau_pdgID_plus) != 0:
        tau_pdgID_plus_has_neuPiCount +=1
    
    if len(neuPiList_tau_pdgID_minus) !=0:
        tau_pdgID_minus_has_neuPiCount += 1
    
    if excludeTausWithNeutralPiInDecayChain == 1:
        #len_neuPiList_tau_pdgID_plus = len(neuPiList_tau_pdgID_plus)
        if len(neuPiList_tau_pdgID_plus) != 0 or len(neuPiList_tau_pdgID_minus) !=0:
            eventHasTauWithNeutralPiInDecayChainCount += 1
            continue #eliminate events with neutral pions in the tau decay chain
    
    if excludeTausWithNeutralPiInDecayChain == 0:
        if len(neuPiList_tau_pdgID_plus) != 0 or len(neuPiList_tau_pdgID_minus) !=0:
            eventHasTauWithNeutralPiInDecayChainCount += 1
            #same as above, except we do NOT skip these events this time, so no continue
            
    if eventHasGoodGenUpsilon:
        if abs(pi1List_tau_pdgID_plus[0].eta()) > etaCut or abs(pi1List_tau_pdgID_minus[0].eta()) > etaCut or abs(pi2List_tau_pdgID_plus[0].eta()) > etaCut or abs(pi2List_tau_pdgID_plus[1].eta()) > etaCut or abs(pi2List_tau_pdgID_minus[0].eta()) > etaCut or abs(pi2List_tau_pdgID_minus[1].eta())  > etaCut:
            #print pi1List_tau_pdgID_plus[0].eta()
            #print pi1List_tau_pdgID_minus[0].eta()
#            print "TEST"
#            print pi1List_tau_pdgID_minus[0].numberOfMothers()
            eventHasGenPiOutsideEtaAcceptCount +=1
            continue #skip events where any of the gen Pis fall outside the tracker eta acceptance
        
        goodEvent_gen_tau_pdgID_plus_pt =  tau_pdgID_plus.pt()
        goodEvent_gen_tau_pdgID_minus_pt = tau_pdgID_minus.pt()
        #print "goodEvent_gen_tau_pdgID_plus_pt is:", goodEvent_gen_tau_pdgID_plus_pt
        if goodEvent_gen_tau_pdgID_plus_pt < 30.:
            h_den_gen_tau_pt_all_pdgID_plus_0to30pt.Fill(goodEvent_gen_tau_pdgID_plus_pt)
        if goodEvent_gen_tau_pdgID_plus_pt >=30.:
            h_den_gen_tau_pt_all_pdgID_plus_30to50pt.Fill(goodEvent_gen_tau_pdgID_plus_pt) 
        if goodEvent_gen_tau_pdgID_minus_pt < 30.:
            h_den_gen_tau_pt_all_pdgID_minus_0to30pt.Fill(goodEvent_gen_tau_pdgID_minus_pt)
        if goodEvent_gen_tau_pdgID_minus_pt >= 30.:
            h_den_gen_tau_pt_all_pdgID_minus_30to50pt.Fill(goodEvent_gen_tau_pdgID_minus_pt) 
            
     
        goodEvent_gen_tau_pdgID_minus_pt = tau_pdgID_minus.pt()  
        goodEvent_gen_pi1s = {'+': pi1List_tau_pdgID_plus, '-': pi1List_tau_pdgID_minus }
        goodEvent_gen_pi2s = {'+':pi2List_tau_pdgID_plus,    '-': pi2List_tau_pdgID_minus}
        
        #could ultimately put an if statement here if I enable the option where we include taus with neutral pi in decay chain and end up having two different matching functions
        #e.g: if option 1 --> candMatchPi blah = first matching function blah blah
        #e.g: if option 2 --> candidateMatchPi blah = second matching function
        #that would probably minimize the amount of rearranging of other parts of the code I would need to do, as it would just involve the 2 line matching block 
        candMatchPi1Info_tau_pdgID_plus_index, candMatchPi1Info_tau_pdgID_plus_dist, candMatchPi1Info_tau_pdgID_plus_deltaR, candMatchPi1Info_tau_pdgID_plus_deltaPt  = matchPi1('+')
        candMatchPi1Info_tau_pdgID_minus_index, candMatchPi1Info_tau_pdgID_minus_dist, candMatchPi1Info_tau_pdgID_minus_deltaR, candMatchPi1Info_tau_pdgID_minus_deltaPt  = matchPi1('-')
 #       print "candMatchPi1Info_tau_pdgID_plus_index is:", candMatchPi1Info_tau_pdgID_plus_index, candMatchPi1Info_tau_pdgID_plus_deltaR, candMatchPi1Info_tau_pdgID_plus_deltaPt
 #       print "candMatchPi1Info_tau_pdgID_plus_dist is:", candMatchPi1Info_tau_pdgID_plus_dist
        
        # deltaR_list_for_first_gen_pi2, deltaR_list_for_second_gen_pi2, deltaPt_list_for_first_gen_pi2, deltaPt_list_for_second_gen_pi2
#        candMatchPi2Info_tau_pdgID_plus_index_list, candMatchPi2Info_tau_pdgID_plus_dist, candMatchPi2Info_tau_pdgID_plus_deltaR, candMatchPi2Info_tau_pdgID_plus_deltaPt, candSqrtSumInQuadDeltaR_list_for_first_gen_pi2, candSqrtSumInQuadDeltaR_list_for_second_gen_pi2, candSqrtSumInQuadDeltaPt_list_for_first_gen_pi2, candSqrtSumInQuadDeltaPt_list_for_second_gen_pi2  = matchPi2('+')
       # print "candMatchPi2Info_tau_pdgID_plus_index_list is:", candMatchPi2Info_tau_pdgID_plus_index_list
       #  print "candMatchPi2Info_tau_pdgID_plus_dist is:", candMatchPi2Info_tau_pdgID_plus_dist
#         print  "candMatchPi2Info_tau_pdgID_plus_deltaR is:", candMatchPi2Info_tau_pdgID_plus_deltaR 
#         print "deltaR_list_for_first_gen_pi2 is:",  deltaR_list_for_first_gen_pi2
#         print " deltaR_list_for_second_gen_pi2 is:", deltaR_list_for_second_gen_pi2
#         print "candMatchPi2Info_tau_pdgID_plus_deltaPt is:",candMatchPi2Info_tau_pdgID_plus_deltaPt,
#         print  "deltaPt_list_for_first_gen_pi2 is:", deltaPt_list_for_first_gen_pi2 
#         print "deltaPt_list_for_second_gen_pi2 is:", deltaPt_list_for_second_gen_pi2
        
        if candMatchPi1Info_tau_pdgID_plus_index is not None and candMatchPi1Info_tau_pdgID_plus_dist != -float('inf') and candMatchPi1Info_tau_pdgID_plus_deltaR != -float('inf') and candMatchPi1Info_tau_pdgID_plus_deltaPt != -float('inf'):
            eventHasMatchedPi1_for_tau_pdgID_plus = True
            print "eventHasMatchedPi1_for_tau_pdgID_plus = True"
#         
        if candMatchPi1Info_tau_pdgID_minus_index is not None and candMatchPi1Info_tau_pdgID_minus_dist != -float('inf') and candMatchPi1Info_tau_pdgID_minus_deltaR != -float('inf') and candMatchPi1Info_tau_pdgID_minus_deltaPt != -float('inf'):
            eventHasMatchedPi1_for_tau_pdgID_minus = True
            print "eventHasMatchedPi1_for_tau_pdgID_minus = True"
#         
        if not eventHasMatchedPi1_for_tau_pdgID_plus or not eventHasMatchedPi1_for_tau_pdgID_minus:
            continue #skip events where we did not find a matched Pi1 for both the tau_pdgID_plus and the tau_pdgID_minus
#         
        candMatchPi2Info_tau_pdgID_plus_index_list, candMatchPi2Info_tau_pdgID_plus_dist, candMatchPi2Info_tau_pdgID_plus_deltaR, candMatchPi2Info_tau_pdgID_plus_deltaPt, candSqrtSumInQuadDeltaR_list_for_first_gen_pi2_tau_pdgID_plus, candSqrtSumInQuadDeltaR_list_for_second_gen_pi2_tau_pdgID_plus, candSqrtSumInQuadDeltaPt_list_for_first_gen_pi2_tau_pdgID_plus, candSqrtSumInQuadDeltaPt_list_for_second_gen_pi2_tau_pdgID_plus  = matchPi2('+')
        candMatchPi2Info_tau_pdgID_minus_index_list, candMatchPi2Info_tau_pdgID_minus_dist, candMatchPi2Info_tau_pdgID_minus_deltaR, candMatchPi2Info_tau_pdgID_minus_deltaPt, candSqrtSumInQuadDeltaR_list_for_first_gen_pi2_tau_pdgID_minus, candSqrtSumInQuadDeltaR_list_for_second_gen_pi2_tau_pdgID_minus, candSqrtSumInQuadDeltaPt_list_for_first_gen_pi2_tau_pdgID_minus, candSqrtSumInQuadDeltaPt_list_for_second_gen_pi2_tau_pdgID_minus  = matchPi2('-')
#         
#         
#         print "candMatchPi2Info_tau_pdgID_plus_index_list is:", candMatchPi2Info_tau_pdgID_plus_index_list
#         print "candMatchPi2Info_tau_pdgID_plus_dist is:", candMatchPi2Info_tau_pdgID_plus_dist
#         
        if candMatchPi2Info_tau_pdgID_plus_index_list is not None and candMatchPi2Info_tau_pdgID_plus_dist != -float('inf') and candMatchPi2Info_tau_pdgID_plus_deltaR != -float('inf') and candMatchPi2Info_tau_pdgID_plus_deltaPt != -float('inf') and candSqrtSumInQuadDeltaR_list_for_first_gen_pi2_tau_pdgID_plus is not None and candSqrtSumInQuadDeltaR_list_for_second_gen_pi2_tau_pdgID_plus is not None and candSqrtSumInQuadDeltaPt_list_for_first_gen_pi2_tau_pdgID_plus is not None and candSqrtSumInQuadDeltaPt_list_for_second_gen_pi2_tau_pdgID_plus is not None:
#              #print "GOT HERE"
            eventHasMatchedPi2_for_tau_pdgID_plus = True
            print "eventHasMatchedPi2_for_tau_pdgID_plus = True"
# #             
        if candMatchPi2Info_tau_pdgID_minus_index_list is not None and candMatchPi2Info_tau_pdgID_minus_dist != -float('inf') and candMatchPi2Info_tau_pdgID_minus_deltaR != -float('inf') and candMatchPi2Info_tau_pdgID_minus_deltaPt != -float('inf') and candSqrtSumInQuadDeltaR_list_for_first_gen_pi2_tau_pdgID_minus is not None and candSqrtSumInQuadDeltaR_list_for_second_gen_pi2_tau_pdgID_minus is not None and candSqrtSumInQuadDeltaPt_list_for_first_gen_pi2_tau_pdgID_minus is not None and candSqrtSumInQuadDeltaPt_list_for_second_gen_pi2_tau_pdgID_minus is not None:
            eventHasMatchedPi2_for_tau_pdgID_minus = True 
            print "eventHasMatchedPi2_for_tau_pdgID_minus = True"
# #         
        if not eventHasMatchedPi2_for_tau_pdgID_plus or not eventHasMatchedPi2_for_tau_pdgID_minus:
            continue #skip events where we did not find matched Pi2s for both tau_pdgID_plus and tau_pdgID_minus 
# #         
        if eventHasMatchedPi1_for_tau_pdgID_plus and eventHasMatchedPi2_for_tau_pdgID_plus:
            eventHasMatchedTau_for_pdgID_plus = True
            print "eventHasMatchedTau_for_pdgID_plus = True"
# #         
        if eventHasMatchedPi1_for_tau_pdgID_minus and eventHasMatchedPi2_for_tau_pdgID_minus:
            eventHasMatchedTau_for_pdgID_minus = True
            print "eventHasMatchedTau_for_pdgID_minus = True"
#         
        if eventHasMatchedTau_for_pdgID_plus and eventHasMatchedTau_for_pdgID_minus:
            eventHasMatchedUpsilon = True
# #         
        if eventHasMatchedUpsilon:
            print "Found matched upsilon!"
            eventHasMatchedUpsilonCount += 1
            if goodEvent_gen_tau_pdgID_plus_pt < 30.:
                h_num_gen_tau_pt_matched_pdgID_plus_0to30pt.Fill(goodEvent_gen_tau_pdgID_plus_pt)
            if goodEvent_gen_tau_pdgID_plus_pt >= 30.:
                h_num_gen_tau_pt_matched_pdgID_plus_30to50pt.Fill(goodEvent_gen_tau_pdgID_plus_pt) 
            if goodEvent_gen_tau_pdgID_minus_pt < 30.:
                h_num_gen_tau_pt_matched_pdgID_minus_0to30pt.Fill(goodEvent_gen_tau_pdgID_minus_pt)
            if goodEvent_gen_tau_pdgID_minus_pt >= 30.:
                h_num_gen_tau_pt_matched_pdgID_minus_30to50pt.Fill(goodEvent_gen_tau_pdgID_minus_pt) 
#              
            tofill = OrderedDict(zip(branches, [-999.] * len(branches)))
             
             #Some sanity check branches. Will set distCutOff_Pi1, distCutOff_Pi2 to float('inf') when I fill these
            if tuneCutParameters:
                #Pi1 figure of merit to cut on and the two constituent terms in the FOM
                tofill['candMatchPi1Info_tau_pdgID_plus_dist'] = candMatchPi1Info_tau_pdgID_plus_dist #FOM to cut on 
                tofill['candMatchPi1Info_tau_pdgID_minus_dist'] = candMatchPi1Info_tau_pdgID_minus_dist #FOM to cut on 
                
                
                tofill['candMatchPi1Info_tau_pdgID_plus_deltaR'] = candMatchPi1Info_tau_pdgID_plus_deltaR #a term in the FOM
                tofill['candMatchPi1Info_tau_pdgID_minus_deltaR'] = candMatchPi1Info_tau_pdgID_minus_deltaR #a term in the FOM
                tofill['candMatchPi1Info_tau_pdgID_plus_deltaPt'] = candMatchPi1Info_tau_pdgID_plus_deltaPt #a term in the FOM
                tofill['candMatchPi1Info_tau_pdgID_minus_deltaPt'] = candMatchPi1Info_tau_pdgID_minus_deltaPt #a term in the FOM
                
                #Pi2 figure of merit to cut on and sort of the constituent parameters. "Sort of" because of cross terms. Recall that FOM is of form sqrt(a^2 + b^2) + sqrt(c^2 + d^2) and the sort of constituents are of form sqrt(a^2 + c^2) and sqrt(b^2 + d^2)
                #For details and pseudocode if more explanation is needed, see email with subject "chatting about improving reco to gen matching" with Markus Seidel, 22 March 2020
                tofill['candMatchPi2Info_tau_pdgID_plus_dist']  = candMatchPi2Info_tau_pdgID_plus_dist #FOM to cut on 
                tofill['candMatchPi2Info_tau_pdgID_minus_dist'] = candMatchPi2Info_tau_pdgID_minus_dist #FOM to cut on 
                tofill['candMatchPi2Info_tau_pdgID_plus_deltaR'] = candMatchPi2Info_tau_pdgID_plus_deltaR #sort of FOM constituent 
                tofill['candMatchPi2Info_tau_pdgID_minus_deltaR'] = candMatchPi2Info_tau_pdgID_minus_deltaR #sort of FOM constituent 
                tofill['candMatchPi2Info_tau_pdgID_plus_deltaPt'] = candMatchPi2Info_tau_pdgID_plus_deltaPt #sort of FOM constituent 
                tofill['candMatchPi2Info_tau_pdgID_minus_deltaPt'] = candMatchPi2Info_tau_pdgID_minus_deltaPt #sort of FOM constituent 
                
                
             
             #NOMENCLATURE CHANGE WARNING! ATTENZIONE! ATTENTION! BE CAREFUL! ACTUALLY READ ME!! #####
             ## Now we will switch nomenclature to match what Shray had and save rewriting! A plus or minus now refers to the CHARGE of the tau, not the pdgID. Up until this point, we had used plus or minus to refer to the pdgID, so a tau was pdgID_plus blah and an antitau was pdgID_minus blah. 
             #No more however! Now we are using plus minus convention to indicate the sign of the charge.
             # e.g. taup blah blah is tau with charge +1, aka the antitau
             #and taum blah blah is tau with charge -1, aka the tau   
             #This is not ideal, but notating it (READ THIS, PEOPLE!!) is the best of an imperfect set of solutions, given that we want code that is both somewhat readable AND we do NOT want to rewrite absolutely everything from earlier iterations
             #Hence, we have a nomenclature convention change that takes effect at this point after we have found events with matched upsilons. Apologies and WARNNGS in advance to those of us who must live with it.
             
                      
             #print "candMatchPi1Info_tau_pdgID_minus_index is:", candMatchPi1Info_tau_pdgID_minus_index
             #print "type(candMatchPi1Info_tau_pdgID_minus_index) is:", type(candMatchPi1Info_tau_pdgID_minus_index)
             
            #Antitau aka pi plus stuff aka taup stuff
            pi_plus1 = rec_pi1s['-'][int(candMatchPi1Info_tau_pdgID_minus_index)]
            pi_plus2 = rec_pi2s['-'][int(candMatchPi2Info_tau_pdgID_minus_index_list[0])]
            pi_plus3 = rec_pi2s['-'][int(candMatchPi2Info_tau_pdgID_minus_index_list[1])]
            
            #variables for sorting the pi_plus objects, being very careful and making a different variable because I am tired
            pi_plus1_unsorted = rec_pi1s['-'][int(candMatchPi1Info_tau_pdgID_minus_index)]
            pi_plus2_unsorted = rec_pi2s['-'][int(candMatchPi2Info_tau_pdgID_minus_index_list[0])]
            pi_plus3_unsorted = rec_pi2s['-'][int(candMatchPi2Info_tau_pdgID_minus_index_list[1])]
            
            pi_plus_unsorted_list = [pi_plus1_unsorted, pi_plus2_unsorted, pi_plus3_unsorted]
            pi_plus_unsorted_list_pt = [pi_plus1_unsorted.pt(), pi_plus2_unsorted.pt(), pi_plus3_unsorted.pt()]
            print "24 July 2020 Addition"
            for i in pi_plus_unsorted_list_pt: print i
            print "#####"
            
            sorted_pi_plus_originalIndexList_pt =  [i[0] for i in sorted(enumerate(pi_plus_unsorted_list_pt), reverse=True, key=lambda x:x[1])]
            
            for myIndex in sorted_pi_plus_originalIndexList_pt: print myIndex 
            
            pi_plus_sorted_in_pt_1 = pi_plus_unsorted_list[sorted_pi_plus_originalIndexList_pt[0]]
            pi_plus_sorted_in_pt_2 = pi_plus_unsorted_list[sorted_pi_plus_originalIndexList_pt[1]]
            pi_plus_sorted_in_pt_3 = pi_plus_unsorted_list[sorted_pi_plus_originalIndexList_pt[2]]
            
            print "pi_plus_sorted_in_pt_1.pt():", pi_plus_sorted_in_pt_1.pt()
            print "pi_plus_sorted_in_pt_2.pt():", pi_plus_sorted_in_pt_2.pt()
            print "pi_plus_sorted_in_pt_3.pt():", pi_plus_sorted_in_pt_3.pt()
            
            print "pi_plus1.pt():", pi_plus1.pt()
            print "pi_plus2.pt():", pi_plus2.pt()
            print "pi_plus3.pt():", pi_plus3.pt()
            
            
          #   print "pi_plus1.p4() is:", pi_plus1.p4()
#             print("pi_plus1.p4().px():", pi_plus1.p4().px() )
#             print("pi_plus2.p4().px():", pi_plus2.p4().px() )
#             print("pi_plus3.p4().px():", pi_plus3.p4().px() )
#             
#             print("pi_plus1.p4().py():", pi_plus1.p4().py() )
#             print("pi_plus2.p4().py():", pi_plus2.p4().py() )
#             print("pi_plus3.p4().py():", pi_plus3.p4().py() )
#             
#             print("pi_plus1.p4().pz():", pi_plus1.p4().pz() )
#             print("pi_plus2.p4().pz():", pi_plus2.p4().pz() )
#             print("pi_plus3.p4().pz():", pi_plus3.p4().pz() )
#             
#             print("pi_plus1.p4().energy():", pi_plus1.p4().energy() )
#             print("pi_plus2.p4().energy():", pi_plus2.p4().energy() )
#             print("pi_plus3.p4().energy():", pi_plus3.p4().energy() )
            
             #print "nuList_tau_pdgID_minus is:", nuList_tau_pdgID_minus
            antineu = nuList_tau_pdgID_minus[0]
            # print "pi_plus1.pdgId() is:", pi_plus1.pdgId()
#             print "pi_plus2.pdgId() is:", pi_plus2.pdgId()
#             print "pi_plus3.pdgId() is:", pi_plus3.pdgId()
#             print "antineu.pdgId() is:", antineu.pdgId()
            taup_charge = np.sign(pi_plus1.pdgId() + pi_plus2.pdgId() + pi_plus3.pdgId())
             
            pi_p_lv1 = TLorentzVector()
            pi_p_lv2 = TLorentzVector()
            pi_p_lv3 = TLorentzVector()
            antineu_lv = TLorentzVector()
             
            pi_p_lv1.SetPtEtaPhiM(pi_plus1.pt(), pi_plus1.eta(), pi_plus1.phi(), piMass)
            pi_p_lv2.SetPtEtaPhiM(pi_plus2.pt(), pi_plus2.eta(), pi_plus2.phi(), piMass)
            pi_p_lv3.SetPtEtaPhiM(pi_plus3.pt(), pi_plus3.eta(), pi_plus3.phi(), piMass)
            antineu_lv.SetPtEtaPhiM(antineu.pt(), antineu.eta(), antineu.phi(), nuMass)
            taup_lv = pi_p_lv1 + pi_p_lv2 + pi_p_lv3 + antineu_lv
            
            print "pi_p_lv1.Px():", pi_p_lv1.Px()
            print "pi_p_lv2.Px()", pi_p_lv2.Px()
            print "pi_p_lv3.Px()", pi_p_lv3.Px()
            
            print "pi_p_lv1.Py():", pi_p_lv1.Py()
            print "pi_p_lv2.Py()", pi_p_lv2.Py()
            print "pi_p_lv3.Py()", pi_p_lv3.Py()
            
            print "pi_p_lv1.Pz():", pi_p_lv1.Pz()
            print "pi_p_lv2.Pz()", pi_p_lv2.Pz()
            print "pi_p_lv3.Pz()", pi_p_lv3.Pz()
            
            print "pi_p_lv1.E():", pi_p_lv1.E()
            print "pi_p_lv2.E()", pi_p_lv2.E()
            print "pi_p_lv3.E()", pi_p_lv3.E()
            
            #Do rotation to local frame for taup
            vis_taup_lv = pi_p_lv1 + pi_p_lv2 + pi_p_lv3
            orig_vis_taup_theta = vis_taup_lv.Theta()
            orig_vis_taup_phi   = vis_taup_lv.Phi()
            
            local_vis_taup_lv =  rotateToVisTauMomPointsAlongZAxis(orig_vis_taup_theta, orig_vis_taup_phi, vis_taup_lv)
            local_pi_p_lv1 = rotateToVisTauMomPointsAlongZAxis(orig_vis_taup_theta, orig_vis_taup_phi,pi_p_lv1)
            local_pi_p_lv2 = rotateToVisTauMomPointsAlongZAxis(orig_vis_taup_theta, orig_vis_taup_phi, pi_p_lv2)
            local_pi_p_lv3 = rotateToVisTauMomPointsAlongZAxis(orig_vis_taup_theta, orig_vis_taup_phi, pi_p_lv3)
            local_antineu_lv = rotateToVisTauMomPointsAlongZAxis(orig_vis_taup_theta, orig_vis_taup_phi, antineu_lv)
            
            local_unsortedPiPtList_p = [local_pi_p_lv1.Pt(), local_pi_p_lv2.Pt(), local_pi_p_lv3.Pt()]
            local_unsortedPi4VecList_p = [local_pi_p_lv1, local_pi_p_lv2, local_pi_p_lv3]
            print "local_unsortedPiPtList_p is:", local_unsortedPiPtList_p
            
             # idea of how to do this from: https://stackoverflow.com/questions/6422700/how-to-get-indices-of-a-sorted-array-in-python
            local_sortedPiPtOriginalIndexList_p =      [i[0] for i in sorted(enumerate(local_unsortedPiPtList_p), reverse=True, key=lambda x:x[1])]
            #print "local_sortedPiPtOriginalIndexList_p:", local_sortedPiPtOriginalIndexList_p
            print "local stuff for sanity checking 28 July 2020"
            print local_sortedPiPtOriginalIndexList_p[0] #index of the element of the vector with the biggest pT
            print local_sortedPiPtOriginalIndexList_p[1] #index of the element of the vector with the second biggest pT
            print local_sortedPiPtOriginalIndexList_p[2] #index of the element of the vector with the smallest pT
        
            local_pi_p_lv1 = local_unsortedPi4VecList_p[local_sortedPiPtOriginalIndexList_p[0]] #make the pi_m_lv1 the vector that has the biggest pT in the new frame
            local_pi_p_lv2 = local_unsortedPi4VecList_p[local_sortedPiPtOriginalIndexList_p[1]] #make the pi_m_lv2 the vector that has the second biggest pT in the new frame
            local_pi_p_lv3 = local_unsortedPi4VecList_p[local_sortedPiPtOriginalIndexList_p[2]] #make the pi_m_lv3 the vector that has the smallest pT in the new frame
            
          #   print "new local_pi_p_lv1.Pt() is:", local_pi_p_lv1.Pt()
#             print "new local_pi_p_lv2.Pt() is:", local_pi_p_lv2.Pt()
#             print "new local_pi_p_lv3.Pt() is:", local_pi_p_lv3.Pt()
        
            #pi_plus_unsorted_list has the pis in the original indices 0,1,2 
            #these we transformed the four vectors from lab to local, keeping the indices the same, in the sense that the pi object associated with local lortenz vector 1 was still the 0th indexed object
            pi_plus_sorted_in_local_pt_1 = pi_plus_unsorted_list[local_sortedPiPtOriginalIndexList_p[0]]
            pi_plus_sorted_in_local_pt_2 = pi_plus_unsorted_list[local_sortedPiPtOriginalIndexList_p[1]]
            pi_plus_sorted_in_local_pt_3 = pi_plus_unsorted_list[local_sortedPiPtOriginalIndexList_p[2]]
            
            #express check for event 4993 of my test file (aka i know what to expect for this test file)
            print "pi_plus_sorted_in_local_pt_1.charge():", pi_plus_sorted_in_local_pt_1.charge()
            print "pi_plus1_unsorted.charge():", pi_plus1_unsorted.charge()
            
            print "pi_plus_sorted_in_local_pt_2.charge():", pi_plus_sorted_in_local_pt_2.charge()
            print "pi_plus2_unsorted.charge():", pi_plus2_unsorted.charge()
            
            print "pi_plus_sorted_in_local_pt_3.charge():", pi_plus_sorted_in_local_pt_3.charge()
            print "pi_plus3_unsorted.charge():", pi_plus3_unsorted.charge()
            
            if pi_plus_sorted_in_local_pt_1.hcalFraction() != 0 and local_sortedPiPtOriginalIndexList_p[0] ==0:
                 print "PISHEE"
                 print pi_plus_sorted_in_local_pt_1.hcalFraction()
                 print pi_plus1_unsorted.hcalFraction()
            
        
            local_pi_p_lv1_pt = local_pi_p_lv1.Pt()
            local_pi_p_lv2_pt = local_pi_p_lv2.Pt()
            local_pi_p_lv3_pt = local_pi_p_lv3.Pt()
            
            local_pi_p_lv1_pt = local_pi_p_lv1.Pt()
            local_pi_p_lv2_pt = local_pi_p_lv2.Pt()
            local_pi_p_lv3_pt = local_pi_p_lv3.Pt()
            
            local_taup_lv = local_pi_p_lv1 + local_pi_p_lv2 + local_pi_p_lv3 + local_antineu_lv
            local_taup_lv_mass = local_taup_lv.M()
            
            #now we are in the so-called local frame, the frame in which the visible tau momentum points along z. 
           #But we are not quite where we want to be yet, we still need to rotate so the lead pT pi in the local, vis tau mom points along Z frame points along neg x and everyone else lives in this world as well
           #We will call this good frame that we want to get to the toUse_local blah blah
        
            initial_leadPt_pi_p_in_AllInZFrame_phi = local_pi_p_lv1.Phi() # we will need this to do the unrotation
            
            toUse_local_pi_p_lv1 = rotateToLeadPtPiInVisTauMomPointsAlongZFramePointsAlongNegX(initial_leadPt_pi_p_in_AllInZFrame_phi, local_pi_p_lv1)
            toUse_local_pi_p_lv2 = rotateToLeadPtPiInVisTauMomPointsAlongZFramePointsAlongNegX(initial_leadPt_pi_p_in_AllInZFrame_phi, local_pi_p_lv2)
            toUse_local_pi_p_lv3 = rotateToLeadPtPiInVisTauMomPointsAlongZFramePointsAlongNegX(initial_leadPt_pi_p_in_AllInZFrame_phi, local_pi_p_lv3)
            toUse_local_antineu_lv = rotateToLeadPtPiInVisTauMomPointsAlongZFramePointsAlongNegX(initial_leadPt_pi_p_in_AllInZFrame_phi, local_antineu_lv)
            
            toUse_local_pi_p_lv1_phi = toUse_local_pi_p_lv1.Phi()
        
            toUse_local_pi_p_lv1_pt = toUse_local_pi_p_lv1.Pt()
            toUse_local_pi_p_lv2_pt = toUse_local_pi_p_lv2.Pt()
            toUse_local_pi_p_lv3_pt = toUse_local_pi_p_lv3.Pt()
            toUse_local_antineu_lv_pt = toUse_local_antineu_lv.Pt()
            toUse_local_antineu_lv_pt_norm_by_tauMass = toUse_local_antineu_lv_pt/tauMass
        
            toUse_local_pi_p_lv1_theta = toUse_local_pi_p_lv1.Theta()
            toUse_local_pi_p_lv2_theta = toUse_local_pi_p_lv2.Theta()
            toUse_local_pi_p_lv3_theta = toUse_local_pi_p_lv3.Theta()
            toUse_local_antineu_lv_theta = toUse_local_antineu_lv.Theta()
        
            toUse_local_pi_p_lv2_phi = get_toUse_local_phi(toUse_local_pi_p_lv2)
            toUse_local_pi_p_lv2.SetPhi(toUse_local_pi_p_lv2_phi)
        
            toUse_local_pi_p_lv3_phi = get_toUse_local_phi(toUse_local_pi_p_lv3)
            toUse_local_pi_p_lv3.SetPhi(toUse_local_pi_p_lv3_phi)
        
            toUse_local_antineu_lv_phi = toUse_local_antineu_lv.Phi() # do not apply the get_toUse_local_phi function here because we do NOT know that the antinu phi should be with [-pi/2, pi/2]
            
            toUse_local_vis_taup_lv = toUse_local_pi_p_lv1 + toUse_local_pi_p_lv2 + toUse_local_pi_p_lv3
            toUse_local_taup_lv = toUse_local_pi_p_lv1 + toUse_local_pi_p_lv2 + toUse_local_pi_p_lv3 + toUse_local_antineu_lv
            
            toUse_local_taup_lv_mass = toUse_local_taup_lv.M()
        
            check3 =  unrotateFromLeadPtPiInVisTauMomPointsAlongZFramePointsAlongNegX(initial_leadPt_pi_p_in_AllInZFrame_phi,toUse_local_taup_lv)
            check3_mass = check3.M()
            check4 = unrotateFromVisTauMomPointsAlongZAxis(orig_vis_taup_theta,orig_vis_taup_phi, check3)
            check4_mass = check4.M()
            
            
            
            print "orig_vis_taup_theta:", orig_vis_taup_theta
            print "orig_vis_taup_phi:", orig_vis_taup_phi
            print "initial_leadPt_pi_p_in_AllInZFrame_phi:",initial_leadPt_pi_p_in_AllInZFrame_phi
            m00_taup = (-np.sin(orig_vis_taup_phi) * np.cos(initial_leadPt_pi_p_in_AllInZFrame_phi)) - (np.cos(orig_vis_taup_theta)*np.cos(orig_vis_taup_phi) * np.sin(initial_leadPt_pi_p_in_AllInZFrame_phi))
            m01_taup = (np.sin(orig_vis_taup_phi) * np.sin(initial_leadPt_pi_p_in_AllInZFrame_phi)) - (np.cos(orig_vis_taup_theta)*np.cos(orig_vis_taup_phi) * np.cos(initial_leadPt_pi_p_in_AllInZFrame_phi))
            m02_taup = np.sin(orig_vis_taup_theta) * np.cos(orig_vis_taup_phi)
            m10_taup = (np.cos(orig_vis_taup_phi) * np.cos(initial_leadPt_pi_p_in_AllInZFrame_phi)) - (np.cos(orig_vis_taup_theta)*np.sin(orig_vis_taup_phi) * np.sin(initial_leadPt_pi_p_in_AllInZFrame_phi))
            m11_taup = (-np.cos(orig_vis_taup_phi) * np.sin(initial_leadPt_pi_p_in_AllInZFrame_phi)) - (np.cos(orig_vis_taup_theta) * np.sin(orig_vis_taup_phi) * np.cos(initial_leadPt_pi_p_in_AllInZFrame_phi))
            m12_taup = np.sin(orig_vis_taup_theta) * np.sin(orig_vis_taup_phi)
            m20_taup = np.sin(orig_vis_taup_theta) * np.sin(initial_leadPt_pi_p_in_AllInZFrame_phi)
            m21_taup = np.sin(orig_vis_taup_theta) * np.cos(initial_leadPt_pi_p_in_AllInZFrame_phi)
            m22_taup = np.cos(orig_vis_taup_theta)
            
            print "m00_taup:", m00_taup
            print "m01_taup:", m01_taup
            print "m02_taup:", m02_taup
            print "m10_taup:", m10_taup
            print "m11_taup:", m11_taup
            print "m12_taup:", m12_taup
            print "m20_taup:", m20_taup
            print "m21_taup:", m21_taup
            print "m22_taup:", m22_taup
            
            in_00_antinu = toUse_local_antineu_lv_pt * np. cos(toUse_local_antineu_lv_phi)
            in_10_antinu = toUse_local_antineu_lv_pt * np.sin(toUse_local_antineu_lv_phi)
            in_20_antinu = toUse_local_antineu_lv_pt * (1/np.tan(toUse_local_antineu_lv_theta))
            
            print "in_00_antinu:", in_00_antinu
            print "in_10_antinu:", in_10_antinu
            print "in_20_antinu:", in_20_antinu
            
            
            out_00_antinu = antineu_lv.Pt() * np.cos(antineu_lv.Phi())
            out_10_antinu = antineu_lv.Pt() * np.sin(antineu_lv.Phi())
            out_20_antinu = antineu_lv.Pt() * (1/np.tan(antineu_lv.Theta()))
            
            print "out_00_antinu:", out_00_antinu
            print "out_10_antinu:", out_10_antinu
            print "out_20_antinu:", out_20_antinu
            
            first_eq_antinu = (m00_taup * in_00_antinu) + (m01_taup * in_10_antinu) +(m02_taup * in_20_antinu)
            second_eq_antinu = (m10_taup * in_00_antinu) + (m11_taup * in_10_antinu) + (m12_taup * in_20_antinu)
            third_eq_antinu = (m20_taup * in_00_antinu) + (m21_taup * in_10_antinu) + (m22_taup * in_20_antinu)


            
           

            
        
            
            #Tau aka pi minus stuff aka taum stuff
             
            pi_minus1 = rec_pi1s['+'][int(candMatchPi1Info_tau_pdgID_plus_index)]
            pi_minus2 = rec_pi2s['+'][int(candMatchPi2Info_tau_pdgID_plus_index_list[0])]
            pi_minus3 = rec_pi2s['+'][int(candMatchPi2Info_tau_pdgID_plus_index_list[1])]
            nu = nuList_tau_pdgID_plus[0]
            # print "pi_minus1.pdgId() is:", pi_minus1.pdgId()
#             print "pi_minus2.pdgId() is:", pi_minus2.pdgId()
#             print "pi_minus3.pdgId() is:", pi_minus3.pdgId()
#             print "nu.pdgId() is:", nu.pdgId()

            #variables for sorting the pi_plus objects, being very careful and making a different variable because I am tired
            pi_minus1_unsorted = rec_pi1s['+'][int(candMatchPi1Info_tau_pdgID_plus_index)]
            pi_minus2_unsorted = rec_pi2s['+'][int(candMatchPi2Info_tau_pdgID_plus_index_list[0])]
            pi_minus3_unsorted = rec_pi2s['+'][int(candMatchPi2Info_tau_pdgID_plus_index_list[1])]
            
            pi_minus_unsorted_list = [pi_minus1_unsorted, pi_minus2_unsorted, pi_minus3_unsorted]
            pi_minus_unsorted_list_pt = [pi_minus1_unsorted.pt(), pi_minus2_unsorted.pt(), pi_minus3_unsorted.pt()]
            print "24 July 2020 Addition"
            for i in pi_minus_unsorted_list_pt: print i
            print "#####"
            
            sorted_pi_minus_originalIndexList_pt =  [i[0] for i in sorted(enumerate(pi_minus_unsorted_list_pt), reverse=True, key=lambda x:x[1])]
            for myIndex in sorted_pi_minus_originalIndexList_pt: print myIndex 
            
            pi_minus_sorted_in_pt_1 = pi_minus_unsorted_list[sorted_pi_minus_originalIndexList_pt[0]]
            pi_minus_sorted_in_pt_2 = pi_minus_unsorted_list[sorted_pi_minus_originalIndexList_pt[1]]
            pi_minus_sorted_in_pt_3 = pi_minus_unsorted_list[sorted_pi_minus_originalIndexList_pt[2]]
            
            print "pi_minus_sorted_in_pt_1.pt():", pi_minus_sorted_in_pt_1.pt()
            print "pi_minus_sorted_in_pt_2.pt():", pi_minus_sorted_in_pt_2.pt()
            print "pi_minus_sorted_in_pt_3.pt():", pi_minus_sorted_in_pt_3.pt()
            
            print "pi_minus1.pt():", pi_minus1.pt()
            print "pi_minus2.pt():", pi_minus2.pt()
            print "pi_minus3.pt():", pi_minus3.pt()
            
            taum_charge = np.sign(pi_minus1.pdgId() + pi_minus2.pdgId() + pi_minus3.pdgId())
            
             
            pi_m_lv1 = TLorentzVector()
            pi_m_lv2 = TLorentzVector()
            pi_m_lv3 = TLorentzVector()
            neu_lv   = TLorentzVector()
             
            pi_m_lv1.SetPtEtaPhiM(pi_minus1.pt(), pi_minus1.eta(), pi_minus1.phi(), piMass)
            pi_m_lv2.SetPtEtaPhiM(pi_minus2.pt(), pi_minus2.eta(), pi_minus2.phi(), piMass)
            pi_m_lv3.SetPtEtaPhiM(pi_minus3.pt(), pi_minus3.eta(), pi_minus3.phi(), piMass)
            neu_lv.SetPtEtaPhiM(nu.pt(), nu.eta(), nu.phi(), nuMass)
            taum_lv = pi_m_lv1 + pi_m_lv2 + pi_m_lv3 + neu_lv
            
            vis_taum_lv = pi_m_lv1 + pi_m_lv2 + pi_m_lv3
            
            orig_vis_taum_theta = vis_taum_lv.Theta() #this is the theta before any rotation has been done, we need to save this
            orig_vis_taum_phi   = vis_taum_lv.Phi() #this is the phi before any rotation has been done, we need to save this
            
            local_vis_taum_lv = rotateToVisTauMomPointsAlongZAxis(orig_vis_taum_theta, orig_vis_taum_phi, vis_taum_lv)
            local_pi_m_lv1 = rotateToVisTauMomPointsAlongZAxis(orig_vis_taum_theta, orig_vis_taum_phi, pi_m_lv1)
            local_pi_m_lv2 = rotateToVisTauMomPointsAlongZAxis(orig_vis_taum_theta, orig_vis_taum_phi, pi_m_lv2)
            local_pi_m_lv3 = rotateToVisTauMomPointsAlongZAxis(orig_vis_taum_theta, orig_vis_taum_phi, pi_m_lv3)
            local_neu_lv   = rotateToVisTauMomPointsAlongZAxis(orig_vis_taum_theta, orig_vis_taum_phi, neu_lv)
            
            local_unsortedPiPtList_m = [local_pi_m_lv1.Pt(), local_pi_m_lv2.Pt(), local_pi_m_lv3.Pt()]
            local_unsortedPi4VecList_m = [local_pi_m_lv1, local_pi_m_lv2, local_pi_m_lv3]
            #print "local_unsortedPiPtList_m is:", local_unsortedPiPtList_m
#            print "local_unsortedPi4VecList_m is:", local_unsortedPi4VecList_m
            
            # idea of how to do this from: https://stackoverflow.com/questions/6422700/how-to-get-indices-of-a-sorted-array-in-python
            local_sortedPiPtOriginalIndexList_m =      [i[0] for i in sorted(enumerate(local_unsortedPiPtList_m), reverse=True, key=lambda x:x[1])]
            #print "local_sortedPiPtOriginalIndexList_m:", local_sortedPiPtOriginalIndexList_m
            
          #   print local_sortedPiPtOriginalIndexList_m[0] #index of the element of the vector with the biggest pT
#             print local_sortedPiPtOriginalIndexList_m[1] #index of the element of the vector with the second biggest pT
#             print local_sortedPiPtOriginalIndexList_m[2] #index of the element of the vector with the smallest pT
            
            local_pi_m_lv1 = local_unsortedPi4VecList_m[local_sortedPiPtOriginalIndexList_m[0]] #make the pi_m_lv1 the vector that has the biggest pT in the new frame
            local_pi_m_lv2 = local_unsortedPi4VecList_m[local_sortedPiPtOriginalIndexList_m[1]] #make the pi_m_lv2 the vector that has the second biggest pT in the new frame
            local_pi_m_lv3 = local_unsortedPi4VecList_m[local_sortedPiPtOriginalIndexList_m[2]] #make the pi_m_lv3 the vector that has the smallest pT in the new frame 
            
            #pi_minus_unsorted_list has the pis in the original indices 0,1,2 
            #these we transformed the four vectors from lab to local, keeping the indices the same, in the sense that the pi object associated with local lortenz vector 1 was still the 0th indexed object
            
            pi_minus_sorted_in_local_pt_1 = pi_minus_unsorted_list[local_sortedPiPtOriginalIndexList_m[0]]
            pi_minus_sorted_in_local_pt_2 = pi_minus_unsorted_list[local_sortedPiPtOriginalIndexList_m[1]]
            pi_minus_sorted_in_local_pt_3 = pi_minus_unsorted_list[local_sortedPiPtOriginalIndexList_m[2]]
            
            if pi_minus_sorted_in_local_pt_1.hcalFraction() != 0 and local_sortedPiPtOriginalIndexList_m[0] ==0:
                 print "PISHEE2"
                 print pi_minus_sorted_in_local_pt_1.hcalFraction()
                 print pi_minus1_unsorted.hcalFraction()
          #   print "new local_pi_m_lv1.Pt() is:", local_pi_m_lv1.Pt()
#             print "new local_pi_m_lv2.Pt() is:", local_pi_m_lv2.Pt()
#             print "new local_pi_m_lv3.Pt() is:", local_pi_m_lv3.Pt()
            
            local_taum_lv = local_pi_m_lv1 + local_pi_m_lv2 + local_pi_m_lv3 + local_neu_lv
            
            local_taum_lv_mass = local_taum_lv.M()
        
            local_pi_m_lv1_pt = local_pi_m_lv1.Pt()
            local_pi_m_lv2_pt = local_pi_m_lv2.Pt()
            local_pi_m_lv3_pt = local_pi_m_lv3.Pt()
            local_pi_m_lv1_eta = local_pi_m_lv1.Eta()
            local_pi_m_lv1_phi = local_pi_m_lv1.Phi()
            local_pi_m_lv1_mass = local_pi_m_lv1.M()
            local_pi_m_lv2_mass = local_pi_m_lv2.M()
            local_pi_m_lv3_mass = local_pi_m_lv3.M()
            
            #now we are in the so-called local frame, the frame in which the visible tau momentum points along z. 
           #But we are not quite where we want to be yet, we still need to rotate so the lead pT pi in the local, vis tau mom points along Z frame points along neg x and everyone else lives in this world as well
            #We will call this good frame that we want to get to the toUse_local blah blah
            
            initial_leadPt_pi_m_in_AllInZFrame_phi = local_pi_m_lv1_phi # we will need this to do the unrotation
            toUse_local_pi_m_lv1 = rotateToLeadPtPiInVisTauMomPointsAlongZFramePointsAlongNegX(initial_leadPt_pi_m_in_AllInZFrame_phi,local_pi_m_lv1)
            
            toUse_local_pi_m_lv1_phi = toUse_local_pi_m_lv1.Phi()
            toUse_local_pi_m_lv2 = rotateToLeadPtPiInVisTauMomPointsAlongZFramePointsAlongNegX(initial_leadPt_pi_m_in_AllInZFrame_phi,local_pi_m_lv2)
            toUse_local_pi_m_lv3 = rotateToLeadPtPiInVisTauMomPointsAlongZFramePointsAlongNegX(initial_leadPt_pi_m_in_AllInZFrame_phi,local_pi_m_lv3)
            toUse_local_neu_lv =  rotateToLeadPtPiInVisTauMomPointsAlongZFramePointsAlongNegX(initial_leadPt_pi_m_in_AllInZFrame_phi,local_neu_lv)
            
            toUse_local_pi_m_lv1_pt =  toUse_local_pi_m_lv1.Pt()
            toUse_local_pi_m_lv2_pt =  toUse_local_pi_m_lv2.Pt()
            toUse_local_pi_m_lv3_pt =  toUse_local_pi_m_lv3.Pt()
            toUse_local_neu_lv_pt =    toUse_local_neu_lv.Pt()
            toUse_local_neu_lv_pt_norm_by_tauMass = toUse_local_neu_lv_pt/tauMass
            
            toUse_local_pi_m_lv1_theta = toUse_local_pi_m_lv1.Theta()
            toUse_local_pi_m_lv2_theta = toUse_local_pi_m_lv2.Theta()
            toUse_local_pi_m_lv3_theta = toUse_local_pi_m_lv3.Theta()
            toUse_local_neu_lv_theta = toUse_local_neu_lv.Theta()
            
            toUse_local_pi_m_lv2_phi = get_toUse_local_phi(toUse_local_pi_m_lv2)
            toUse_local_pi_m_lv2.SetPhi(toUse_local_pi_m_lv2_phi)
            toUse_local_pi_m_lv3_phi = get_toUse_local_phi(toUse_local_pi_m_lv3)
            toUse_local_pi_m_lv3.SetPhi(toUse_local_pi_m_lv3_phi)
            
            toUse_local_neu_lv_phi = toUse_local_neu_lv.Phi() # do not apply the get_toUse_local_phi function here because we do NOT know that the nu phi should be with [-pi/2, pi/2]
            
            toUse_local_taum_lv =  toUse_local_pi_m_lv1 + toUse_local_pi_m_lv2 + toUse_local_pi_m_lv3 + toUse_local_neu_lv
            toUse_local_taum_lv_mass = toUse_local_taum_lv.M()
            check1 =  unrotateFromLeadPtPiInVisTauMomPointsAlongZFramePointsAlongNegX(initial_leadPt_pi_m_in_AllInZFrame_phi,toUse_local_taum_lv)
            check1_mass = check1.M()
            check2 = unrotateFromVisTauMomPointsAlongZAxis(orig_vis_taum_theta,orig_vis_taum_phi, check1)
            check2_mass = check2.M()
            
            ####
            #upsilon_lv = taup_lv + taum_lv
            #check_upsilon_lv = check2 + check4
            
             
             #print"dir(nu) is:", dir(nu)
             #print "#################"
             #print "dir(pi_plus1) is:", dir(pi_plus1)
             #print "pi_plus1.numberOfMothers() is:", pi_plus1.numberOfMothers()
            
             
            print "orig_vis_taum_theta:", orig_vis_taum_theta
            print "orig_vis_taum_phi:", orig_vis_taum_phi
            print "initial_leadPt_pi_m_in_AllInZFrame_phi:",initial_leadPt_pi_m_in_AllInZFrame_phi
            m00_taum = (-np.sin(orig_vis_taum_phi) * np.cos(initial_leadPt_pi_m_in_AllInZFrame_phi)) - (np.cos(orig_vis_taum_theta)*np.cos(orig_vis_taum_phi) * np.sin(initial_leadPt_pi_m_in_AllInZFrame_phi))
            m01_taum = (np.sin(orig_vis_taum_phi) * np.sin(initial_leadPt_pi_m_in_AllInZFrame_phi)) - (np.cos(orig_vis_taum_theta)*np.cos(orig_vis_taum_phi) * np.cos(initial_leadPt_pi_m_in_AllInZFrame_phi))
            m02_taum = np.sin(orig_vis_taum_theta) * np.cos(orig_vis_taum_phi)
            m10_taum = (np.cos(orig_vis_taum_phi) * np.cos(initial_leadPt_pi_m_in_AllInZFrame_phi)) - (np.cos(orig_vis_taum_theta)*np.sin(orig_vis_taum_phi) * np.sin(initial_leadPt_pi_m_in_AllInZFrame_phi))
            m11_taum = (-np.cos(orig_vis_taum_phi) * np.sin(initial_leadPt_pi_m_in_AllInZFrame_phi)) - (np.cos(orig_vis_taum_theta) * np.sin(orig_vis_taum_phi) * np.cos(initial_leadPt_pi_m_in_AllInZFrame_phi))
            m12_taum = np.sin(orig_vis_taum_theta) * np.sin(orig_vis_taum_phi)
            m20_taum = np.sin(orig_vis_taum_theta) * np.sin(initial_leadPt_pi_m_in_AllInZFrame_phi)
            m21_taum = np.sin(orig_vis_taum_theta) * np.cos(initial_leadPt_pi_m_in_AllInZFrame_phi)
            m22_taum = np.cos(orig_vis_taum_theta)
            
            print "m00_taum:", m00_taum
            print "m01_taum:", m01_taum
            print "m02_taum:", m02_taum
            print "m10_taum:", m10_taum
            print "m11_taum:", m11_taum
            print "m12_taum:", m12_taum
            print "m20_taum:", m20_taum
            print "m21_taum:", m21_taum
            print "m22_taum:", m22_taum 
            
            in_00_nu = toUse_local_neu_lv_pt * np. cos(toUse_local_neu_lv_phi)
            in_10_nu = toUse_local_neu_lv_pt * np.sin(toUse_local_neu_lv_phi)
            in_20_nu = toUse_local_neu_lv_pt * (1/np.tan(toUse_local_neu_lv_theta))
            
            print "in_00_nu:", in_00_nu
            print "in_10_nu:", in_10_nu
            print "in_20_nu:", in_20_nu
            
            
            out_00_nu = neu_lv.Pt() * np.cos(neu_lv.Phi())
            out_10_nu = neu_lv.Pt() * np.sin(neu_lv.Phi())
            out_20_nu = neu_lv.Pt() * (1/np.tan(neu_lv.Theta()))
            
            print "out_00_nu:", out_00_nu
            print "out_10_nu:", out_10_nu
            print "out_20_nu:", out_20_nu
            
            first_eq_nu = (m00_taum * in_00_nu) + (m01_taum * in_10_nu) +(m02_taum * in_20_nu)
            second_eq_nu = (m10_taum * in_00_nu) + (m11_taum * in_10_nu) + (m12_taum * in_20_nu)
            third_eq_nu = (m20_taum * in_00_nu) + (m21_taum * in_10_nu) + (m22_taum * in_20_nu)
             
             
             
             
             
             #Put things into tofill
            tofill['taup_mass'] = taup_lv.M()
            tofill['taup_pt'] = taup_lv.Pt()
            tofill['taup_eta'] = taup_lv.Eta()
            tofill['taup_phi'] = taup_lv.Phi()
            tofill['taup_theta'] = taup_lv.Theta()
            tofill['taum_mass'] = taum_lv.M()
            tofill['taum_pt'] = taum_lv.Pt()
            tofill['taum_eta'] = taum_lv.Eta()
            tofill['taum_phi'] = taum_lv.Phi()
            tofill['taum_theta'] = taum_lv.Theta()
            
           #  tofill['upsilon_mass'] = upsilon_lv.M()
#             tofill['upsilon_pt'] = upsilon_lv.Pt()
#             tofill['upsilon_phi'] = upsilon_lv.Phi()
#             tofill['upsilon_eta'] = upsilon_lv.Eta()
#             tofill['upsilon_theta'] = upsilon_lv.Theta()
            
            tofill['pi_minus1_pt'] = pi_m_lv1.Pt()
            tofill['pi_minus1_eta'] = pi_m_lv1.Eta()
            tofill['pi_minus1_phi'] = pi_m_lv1.Phi()
            tofill['pi_minus1_theta'] = pi_m_lv1.Theta()
            tofill['pi_minus2_pt'] = pi_m_lv2.Pt()
            tofill['pi_minus2_eta'] = pi_m_lv2.Eta()
            tofill['pi_minus2_phi'] = pi_m_lv2.Phi()
            tofill['pi_minus2_theta'] = pi_m_lv2.Theta()
            tofill['pi_minus3_pt'] = pi_m_lv3.Pt()
            tofill['pi_minus3_eta'] = pi_m_lv3.Eta()
            tofill['pi_minus3_phi'] = pi_m_lv3.Phi()
            tofill['pi_minus3_theta'] = pi_m_lv3.Theta()
            
            tofill['pi_plus1_pt'] = pi_p_lv1.Pt()
            tofill['pi_plus1_eta'] = pi_p_lv1.Eta()
            tofill['pi_plus1_phi'] = pi_p_lv1.Phi()
            tofill['pi_plus1_theta'] = pi_p_lv1.Theta()
            tofill['pi_plus2_pt'] = pi_p_lv2.Pt()
            tofill['pi_plus2_eta'] = pi_p_lv2.Eta()
            tofill['pi_plus2_phi'] = pi_p_lv2.Phi()
            tofill['pi_plus2_theta'] = pi_p_lv2.Theta()
            tofill['pi_plus3_pt'] = pi_p_lv3.Pt()
            tofill['pi_plus3_eta'] = pi_p_lv3.Eta()
            tofill['pi_plus3_phi'] = pi_p_lv3.Phi()
            tofill['pi_plus3_theta'] = pi_p_lv3.Theta()
            
            
            #Lab frame, just normalized, no sorting 
            tofill['pi_pt_lv1_norm_by_tauMass_p'] = (pi_p_lv1.Pt()) * (1/tauMass)
            tofill['pi_pt_lv2_norm_by_tauMass_p'] = (pi_p_lv2.Pt()) * (1/tauMass)
            tofill['pi_pt_lv3_norm_by_tauMass_p'] = (pi_p_lv3.Pt()) * (1/tauMass)
            tofill['pi_pt_lv1_norm_by_tauMass_m'] = (pi_m_lv1.Pt()) * (1/tauMass)
            tofill['pi_pt_lv2_norm_by_tauMass_m'] = (pi_m_lv2.Pt()) * (1/tauMass)
            tofill['pi_pt_lv3_norm_by_tauMass_m'] = (pi_m_lv3.Pt()) * (1/tauMass)
            
            
            tofill['neutrino_pt'] = neu_lv.Pt()
            tofill['neutrino_phi'] = neu_lv.Phi()
            tofill['neutrino_eta'] = neu_lv.Eta()
            tofill['neutrino_theta'] = neu_lv.Theta()
            
            tofill['antineutrino_pt'] = antineu_lv.Pt()
            tofill['antineutrino_phi'] = antineu_lv.Phi()
            tofill['antineutrino_eta'] = antineu_lv.Eta()
            tofill['antineutrino_theta'] = antineu_lv.Theta()
            
            tofill['taup_charge'] = taup_charge
            tofill['taum_charge'] = taum_charge 
            
            
            #original info to save for unrotation
            tofill['orig_vis_taum_phi'] = orig_vis_taum_phi
            tofill['orig_vis_taum_theta'] = orig_vis_taum_theta 
            tofill['orig_vis_taup_phi'] = orig_vis_taup_phi
            tofill['orig_vis_taup_theta'] = orig_vis_taup_theta
           
            tofill["initial_leadPt_pi_m_in_AllInZFrame_phi"] =  initial_leadPt_pi_m_in_AllInZFrame_phi
            tofill["initial_leadPt_pi_p_in_AllInZFrame_phi"] =  initial_leadPt_pi_p_in_AllInZFrame_phi
            
             ####toUse local stuff ###
            tofill['toUse_local_taup_lv_mass'] = toUse_local_taup_lv_mass
            tofill["toUse_local_taum_lv_mass"] = toUse_local_taum_lv_mass
            tofill['toUse_local_pi_p_lv1_phi'] = toUse_local_pi_p_lv1_phi #always pi by construction
            tofill["toUse_local_pi_m_lv1_phi"] = toUse_local_pi_m_lv1_phi #always pi by construction 
            
            
            
            #toUse pT stuff
            
            tofill["toUse_local_pi_p_lv1_pt"] = toUse_local_pi_p_lv1_pt 
            tofill["toUse_local_pi_p_lv2_pt"] = toUse_local_pi_p_lv2_pt 
            tofill["toUse_local_pi_p_lv3_pt"] = toUse_local_pi_p_lv3_pt 
            tofill["toUse_local_antineu_lv_pt"] =  toUse_local_antineu_lv_pt
            
            tofill["toUse_local_pi_m_lv1_pt"] = toUse_local_pi_m_lv1_pt
            tofill["toUse_local_pi_m_lv2_pt"] = toUse_local_pi_m_lv2_pt
            tofill["toUse_local_pi_m_lv3_pt"] = toUse_local_pi_m_lv3_pt
            tofill["toUse_local_neu_lv_pt"]   = toUse_local_neu_lv_pt
            
            #toUse theta stuff
            tofill["toUse_local_pi_p_lv1_theta"] = toUse_local_pi_p_lv1_theta
            tofill["toUse_local_pi_p_lv2_theta"] = toUse_local_pi_p_lv2_theta
            tofill["toUse_local_pi_p_lv3_theta"] = toUse_local_pi_p_lv3_theta
            tofill["toUse_local_antineu_lv_theta"] = toUse_local_antineu_lv_theta
            
            tofill["toUse_local_pi_m_lv1_theta"] = toUse_local_pi_m_lv1_theta
            tofill["toUse_local_pi_m_lv2_theta"] = toUse_local_pi_m_lv2_theta
            tofill["toUse_local_pi_m_lv3_theta"] = toUse_local_pi_m_lv3_theta
            tofill["toUse_local_neu_lv_theta"] =   toUse_local_neu_lv_theta
            
            #toUse phi stuff
            tofill["toUse_local_pi_p_lv2_phi"] = toUse_local_pi_p_lv2_phi
            tofill["toUse_local_pi_p_lv3_phi"] = toUse_local_pi_p_lv3_phi
            tofill["toUse_local_antineu_lv_phi"] = toUse_local_antineu_lv_phi
            
            tofill["toUse_local_pi_m_lv2_phi"] = toUse_local_pi_m_lv2_phi
            tofill["toUse_local_pi_m_lv3_phi"] = toUse_local_pi_m_lv3_phi
            tofill["toUse_local_neu_lv_phi"] = toUse_local_neu_lv_phi
            
            #New labels
            
            tofill["toUse_local_neu_lv_pt_norm_by_tauMass"] = toUse_local_neu_lv_pt_norm_by_tauMass
            tofill["toUse_local_antineu_lv_pt_norm_by_tauMass"] = toUse_local_antineu_lv_pt_norm_by_tauMass
            
             
             #Sanity check
            tofill['local_taup_lv_mass'] = local_taup_lv_mass
#            tofill['len_neuPiList_tau_pdgID_plus'] = len_neuPiList_tau_pdgID_plus
            tofill['local_pi_p_lv1_pt'] = local_pi_p_lv1_pt
            tofill["check1_mass"] = check1_mass #check1_mass and check2_mass give back the taum mass, as they should
            tofill["check2_mass"] = check2_mass #so unrotation works out!
            tofill['check3_mass'] = check3_mass
            tofill['check4_mass'] = check4_mass
             
             
             #some pT ordered global variable stuff
            unsorted_PiPtList_p = [pi_p_lv1.Pt(), pi_p_lv2.Pt(), pi_p_lv3.Pt()]
            print "11 July Addition Check"
            for myPt_p in unsorted_PiPtList_p: print myPt_p
            unsorted_PiPt4VecList_p = [pi_p_lv1, pi_p_lv2, pi_p_lv3]
             
            sorted_PiPtOriginalIndexList_p =  [i[0] for i in sorted(enumerate(unsorted_PiPtList_p), reverse=True, key=lambda x:x[1])]
            for myIndex in sorted_PiPtOriginalIndexList_p: print myIndex
            
            sorted_in_pt_pi_lv1_p = unsorted_PiPt4VecList_p[sorted_PiPtOriginalIndexList_p[0]]
            sorted_in_pt_pi_lv2_p = unsorted_PiPt4VecList_p[sorted_PiPtOriginalIndexList_p[1]]
            sorted_in_pt_pi_lv3_p = unsorted_PiPt4VecList_p[sorted_PiPtOriginalIndexList_p[2]]
            
            
            unsorted_PiPtList_m = [pi_m_lv1.Pt(), pi_m_lv2.Pt(), pi_m_lv3.Pt()]
            print "11 July Addition Check"
            for myPt_m in unsorted_PiPtList_p: print myPt_m
            unsorted_PiPt4VecList_m = [pi_m_lv1, pi_m_lv2, pi_m_lv3]
             
            sorted_PiPtOriginalIndexList_m =  [i[0] for i in sorted(enumerate(unsorted_PiPtList_m), reverse=True, key=lambda x:x[1])]
            for myIndex in sorted_PiPtOriginalIndexList_m: print myIndex
            
            sorted_in_pt_pi_lv1_m = unsorted_PiPt4VecList_m[sorted_PiPtOriginalIndexList_m[0]]
            sorted_in_pt_pi_lv2_m = unsorted_PiPt4VecList_m[sorted_PiPtOriginalIndexList_m[1]]
            sorted_in_pt_pi_lv3_m = unsorted_PiPt4VecList_m[sorted_PiPtOriginalIndexList_m[2]]
            
            
            
            # sorted_in_pt_all_pi_lv4 = unsorted_PiPt4VecList[sorted_PiPtOriginalIndexList[3]]
#             sorted_in_pt_all_pi_lv5 = unsorted_PiPt4VecList[sorted_PiPtOriginalIndexList[4]]
#             sorted_in_pt_all_pi_lv6 = unsorted_PiPt4VecList[sorted_PiPtOriginalIndexList[5]]
   #          
#             print sorted_in_pt_all_pi_lv1.Pt()
#             print sorted_in_pt_all_pi_lv2.Pt()
#             print sorted_in_pt_all_pi_lv3.Pt()
#             print sorted_in_pt_all_pi_lv4.Pt()
#             print sorted_in_pt_all_pi_lv5.Pt()
#             print sorted_in_pt_all_pi_lv6.Pt()
#             
#             for myPt in unsorted_PiPtList: print myPt
#             
            tofill['sorted_in_pt_pi_lv1_pt_p'] = sorted_in_pt_pi_lv1_p.Pt()
            tofill['sorted_in_pt_pi_lv2_pt_p'] = sorted_in_pt_pi_lv2_p.Pt()
            tofill['sorted_in_pt_pi_lv3_pt_p'] = sorted_in_pt_pi_lv3_p.Pt()
            tofill['sorted_in_pt_pi_lv1_pt_m'] = sorted_in_pt_pi_lv1_m.Pt()
            tofill['sorted_in_pt_pi_lv2_pt_m'] = sorted_in_pt_pi_lv2_m.Pt()
            tofill['sorted_in_pt_pi_lv3_pt_m'] = sorted_in_pt_pi_lv3_m.Pt()
            
            tofill['sorted_in_pt_pi_lv1_eta_p'] = sorted_in_pt_pi_lv1_p.Eta()
            tofill['sorted_in_pt_pi_lv2_eta_p'] = sorted_in_pt_pi_lv2_p.Eta()
            tofill['sorted_in_pt_pi_lv3_eta_p'] = sorted_in_pt_pi_lv3_p.Eta()
            tofill['sorted_in_pt_pi_lv1_eta_m'] = sorted_in_pt_pi_lv1_m.Eta()
            tofill['sorted_in_pt_pi_lv2_eta_m'] = sorted_in_pt_pi_lv2_m.Eta()
            tofill['sorted_in_pt_pi_lv3_eta_m'] = sorted_in_pt_pi_lv3_m.Eta()
            
            tofill['sorted_in_pt_pi_lv1_theta_p'] = sorted_in_pt_pi_lv1_p.Theta()
            tofill['sorted_in_pt_pi_lv2_theta_p'] = sorted_in_pt_pi_lv2_p.Theta()
            tofill['sorted_in_pt_pi_lv3_theta_p'] = sorted_in_pt_pi_lv3_p.Theta()
            tofill['sorted_in_pt_pi_lv1_theta_m'] = sorted_in_pt_pi_lv1_m.Theta()
            tofill['sorted_in_pt_pi_lv2_theta_m'] = sorted_in_pt_pi_lv2_m.Theta()
            tofill['sorted_in_pt_pi_lv3_theta_m'] = sorted_in_pt_pi_lv3_m.Theta()
            
            tofill['sorted_in_pt_pi_lv1_phi_p'] = sorted_in_pt_pi_lv1_p.Phi()
            tofill['sorted_in_pt_pi_lv2_phi_p'] = sorted_in_pt_pi_lv2_p.Phi()
            tofill['sorted_in_pt_pi_lv3_phi_p'] = sorted_in_pt_pi_lv3_p.Phi()
            tofill['sorted_in_pt_pi_lv1_phi_m'] = sorted_in_pt_pi_lv1_m.Phi()
            tofill['sorted_in_pt_pi_lv2_phi_m'] = sorted_in_pt_pi_lv2_m.Phi()
            tofill['sorted_in_pt_pi_lv3_phi_m'] = sorted_in_pt_pi_lv3_m.Phi()
            
            tofill['sorted_in_pt_pi_lv1_pt_norm_by_tauMass_p'] = (sorted_in_pt_pi_lv1_p.Pt()) * (1/tauMass)
            tofill['sorted_in_pt_pi_lv2_pt_norm_by_tauMass_p'] = (sorted_in_pt_pi_lv2_p.Pt()) * (1/tauMass)
            tofill['sorted_in_pt_pi_lv3_pt_norm_by_tauMass_p'] = (sorted_in_pt_pi_lv3_p.Pt()) * (1/tauMass)
            tofill['sorted_in_pt_pi_lv1_pt_norm_by_tauMass_m'] = (sorted_in_pt_pi_lv1_m.Pt()) * (1/tauMass)
            tofill['sorted_in_pt_pi_lv2_pt_norm_by_tauMass_m'] = (sorted_in_pt_pi_lv2_m.Pt()) * (1/tauMass)
            tofill['sorted_in_pt_pi_lv3_pt_norm_by_tauMass_m'] = (sorted_in_pt_pi_lv3_m.Pt()) * (1/tauMass)
            
#            print "sorted_in_pt_pi_lv1_p.p4():", sorted_in_pt_pi_lv1_p.p4()
            
            print "sorted_in_pt_pi_lv1_p.Px()", sorted_in_pt_pi_lv1_p.Px()
            print "sorted_in_pt_pi_lv1_p.Py()",  sorted_in_pt_pi_lv1_p.Py()
            print "sorted_in_pt_pi_lv1_p.Pz()", sorted_in_pt_pi_lv1_p.Pz()
            print "sorted_in_pt_pi_lv1_p.E()", sorted_in_pt_pi_lv1_p.E()
            
            #Px,Py,Pz, M, E associated with tau + in global frame 
            tofill["sorted_in_pt_pi_lv1_px_p"] = sorted_in_pt_pi_lv1_p.Px()
            tofill["sorted_in_pt_pi_lv1_py_p"] = sorted_in_pt_pi_lv1_p.Py()
            tofill["sorted_in_pt_pi_lv1_pz_p"] = sorted_in_pt_pi_lv1_p.Pz()
            tofill["sorted_in_pt_pi_lv1_E_p"] = sorted_in_pt_pi_lv1_p.E()
            tofill["sorted_in_pt_pi_lv1_mass_p"] = sorted_in_pt_pi_lv1_p.M()
            
            tofill["sorted_in_pt_pi_lv2_px_p"] = sorted_in_pt_pi_lv2_p.Px()
            tofill["sorted_in_pt_pi_lv2_py_p"] = sorted_in_pt_pi_lv2_p.Py()
            tofill["sorted_in_pt_pi_lv2_pz_p"] = sorted_in_pt_pi_lv2_p.Pz()
            tofill["sorted_in_pt_pi_lv2_E_p"] = sorted_in_pt_pi_lv2_p.E()
            tofill["sorted_in_pt_pi_lv2_mass_p"] = sorted_in_pt_pi_lv2_p.M()
            
            tofill["sorted_in_pt_pi_lv3_px_p"] = sorted_in_pt_pi_lv3_p.Px()
            tofill["sorted_in_pt_pi_lv3_py_p"] = sorted_in_pt_pi_lv3_p.Py()
            tofill["sorted_in_pt_pi_lv3_pz_p"] = sorted_in_pt_pi_lv3_p.Pz()
            tofill["sorted_in_pt_pi_lv3_E_p"] = sorted_in_pt_pi_lv3_p.E()
            tofill["sorted_in_pt_pi_lv3_mass_p"] = sorted_in_pt_pi_lv3_p.M()
            
            #Px, Py, Pz, M, E associated with tau - in global frame 
            tofill["sorted_in_pt_pi_lv1_px_m"] = sorted_in_pt_pi_lv1_m.Px()
            tofill["sorted_in_pt_pi_lv1_py_m"] = sorted_in_pt_pi_lv1_m.Py()
            tofill["sorted_in_pt_pi_lv1_pz_m"] = sorted_in_pt_pi_lv1_m.Pz()
            tofill["sorted_in_pt_pi_lv1_E_m"] = sorted_in_pt_pi_lv1_m.E()
            tofill["sorted_in_pt_pi_lv1_mass_m"] = sorted_in_pt_pi_lv1_m.M()
            
            tofill["sorted_in_pt_pi_lv2_px_m"] = sorted_in_pt_pi_lv2_m.Px()
            tofill["sorted_in_pt_pi_lv2_py_m"] = sorted_in_pt_pi_lv2_m.Py()
            tofill["sorted_in_pt_pi_lv2_pz_m"] = sorted_in_pt_pi_lv2_m.Pz()
            tofill["sorted_in_pt_pi_lv2_E_m"] = sorted_in_pt_pi_lv2_m.E()
            tofill["sorted_in_pt_pi_lv2_mass_m"] = sorted_in_pt_pi_lv2_m.M()
            
            tofill["sorted_in_pt_pi_lv3_px_m"] = sorted_in_pt_pi_lv3_m.Px()
            tofill["sorted_in_pt_pi_lv3_py_m"] = sorted_in_pt_pi_lv3_m.Py()
            tofill["sorted_in_pt_pi_lv3_pz_m"] = sorted_in_pt_pi_lv3_m.Pz()
            tofill["sorted_in_pt_pi_lv3_E_m"] = sorted_in_pt_pi_lv3_m.E()
            tofill["sorted_in_pt_pi_lv3_mass_m"] = sorted_in_pt_pi_lv3_m.M()
            
  #           tofill["pi_plus_sorted_in_pt_1_hcalFraction"] = sorted_in_pt_pi_l
#             tofill["pi_plus_sorted_in_pt_2_hcalFraction"]
#             tofill["pi_plus_sorted_in_pt_3_hcalFraction"]
#             tofill["pi_minus_sorted_in_pt_1_hcalFraction"]
#             tofill["pi_minus_sorted_in_pt_2_hcalFraction"]
#             tofill["pi_minus_sorted_in_pt_3_hcalFraction"]
#             
            
            print  tofill["sorted_in_pt_pi_lv3_mass_m"]
            
            #toUse_local_pi_p_lv Px, Py, Pz, E, M
            #some playing for my edification about locals()
#             for thing in ('px', 'py', 'pz'): print thing

            #Thanks to Riju I got this! although it ended up not to be the way I wanted to go. Leaving for posterity. Also yes, ended up not using the 1,2,3 stuff... 
#             for thing, otherThing in zip(['Px', 'Py', 'Pz'], [1,2,3]): 
#                 print tofill['toUse_local_pi_m_lv3_%s' %(thing.lower())]
#                 tofill['toUse_local_pi_m_lv3_%s' %(thing)] = getattr(toUse_local_pi_m_lv3, "%s" %(thing))() #need the extra ()
#                 print  getattr(toUse_local_pi_m_lv3, "%s" %(thing))
#                 print  getattr(toUse_local_pi_m_lv3, "%s" %(thing))()
#                 
#                 #print tofill['toUse_local_pi_m_lv3_%s' %(thing.lower())]
# #                print 
# #             print locals()["tofill"]["toUse_local_pi_m_lv3_pz"]
# #             #for i in ('Px', 'Py', 'Pz'):
#             #    locals() ["tofill["var_%s' %(i"]]['var_%s' %(i)] = i
            #toUse_local_pi_p stuff
            tofill["toUse_local_pi_p_lv1_px"] = toUse_local_pi_p_lv1.Px()
            tofill["toUse_local_pi_p_lv1_py"] = toUse_local_pi_p_lv1.Py()
            tofill["toUse_local_pi_p_lv1_pz"] = toUse_local_pi_p_lv1.Pz()
            tofill["toUse_local_pi_p_lv1_E"] = toUse_local_pi_p_lv1.E()
            tofill["toUse_local_pi_p_lv1_mass"] = toUse_local_pi_p_lv1.M()
            
            tofill["toUse_local_pi_p_lv2_px"] = toUse_local_pi_p_lv2.Px()
            tofill["toUse_local_pi_p_lv2_py"] = toUse_local_pi_p_lv2.Py()
            tofill["toUse_local_pi_p_lv2_pz"] = toUse_local_pi_p_lv2.Pz()
            tofill["toUse_local_pi_p_lv2_E"] = toUse_local_pi_p_lv2.E()
            tofill["toUse_local_pi_p_lv2_mass"] = toUse_local_pi_p_lv2.M()
            
            tofill["toUse_local_pi_p_lv3_px"] = toUse_local_pi_p_lv3.Px()
            tofill["toUse_local_pi_p_lv3_py"] = toUse_local_pi_p_lv3.Py()
            tofill["toUse_local_pi_p_lv3_pz"] = toUse_local_pi_p_lv3.Pz()
            tofill["toUse_local_pi_p_lv3_E"] = toUse_local_pi_p_lv3.E()
            tofill["toUse_local_pi_p_lv3_mass"] = toUse_local_pi_p_lv3.M()
            
            #toUse_local_pi_m stuff 
            tofill["toUse_local_pi_m_lv1_px"] = toUse_local_pi_m_lv1.Px()
            tofill["toUse_local_pi_m_lv1_py"] = toUse_local_pi_m_lv1.Py()
            tofill["toUse_local_pi_m_lv1_pz"] = toUse_local_pi_m_lv1.Pz()
            tofill["toUse_local_pi_m_lv1_E"] = toUse_local_pi_m_lv1.E()
            tofill["toUse_local_pi_m_lv1_mass"] = toUse_local_pi_m_lv1.M()
            
            tofill["toUse_local_pi_m_lv2_px"] = toUse_local_pi_m_lv2.Px()
            tofill["toUse_local_pi_m_lv2_py"] = toUse_local_pi_m_lv2.Py()
            tofill["toUse_local_pi_m_lv2_pz"] = toUse_local_pi_m_lv2.Pz()
            tofill["toUse_local_pi_m_lv2_E"] = toUse_local_pi_m_lv2.E()
            tofill["toUse_local_pi_m_lv2_mass"] = toUse_local_pi_m_lv2.M()
            
            tofill["toUse_local_pi_m_lv3_px"] = toUse_local_pi_m_lv3.Px()
            tofill["toUse_local_pi_m_lv3_py"] = toUse_local_pi_m_lv3.Py()
            tofill["toUse_local_pi_m_lv3_pz"] = toUse_local_pi_m_lv3.Pz()
            tofill["toUse_local_pi_m_lv3_E"] = toUse_local_pi_m_lv3.E()
            tofill["toUse_local_pi_m_lv3_mass"] = toUse_local_pi_m_lv3.M()
            
            
            tofill["pi_plus_sorted_in_pt_1_pdgID"] = pi_plus_sorted_in_pt_1.pdgId()
            tofill["pi_plus_sorted_in_pt_2_pdgID"] = pi_plus_sorted_in_pt_2.pdgId()
            tofill["pi_plus_sorted_in_pt_3_pdgID"] = pi_plus_sorted_in_pt_3.pdgId()
            tofill["pi_minus_sorted_in_pt_1_pdgID"] = pi_minus_sorted_in_pt_1.pdgId()
            tofill["pi_minus_sorted_in_pt_2_pdgID"] = pi_minus_sorted_in_pt_2.pdgId()
            tofill["pi_minus_sorted_in_pt_3_pdgID"] = pi_minus_sorted_in_pt_3.pdgId()
            
            tofill["pi_plus_sorted_in_pt_1_charge"] = pi_plus_sorted_in_pt_1.charge()
            tofill["pi_plus_sorted_in_pt_2_charge"] = pi_plus_sorted_in_pt_2.charge()
            tofill["pi_plus_sorted_in_pt_3_charge"] = pi_plus_sorted_in_pt_3.charge()
            tofill["pi_minus_sorted_in_pt_1_charge"] = pi_minus_sorted_in_pt_1.charge()
            tofill["pi_minus_sorted_in_pt_2_charge"] = pi_minus_sorted_in_pt_2.charge()
            tofill["pi_minus_sorted_in_pt_3_charge"] = pi_minus_sorted_in_pt_3.charge()
            
            tofill["pi_plus_sorted_in_pt_1_hcalFraction"] = pi_plus_sorted_in_pt_1.hcalFraction()
            tofill["pi_plus_sorted_in_pt_2_hcalFraction"] = pi_plus_sorted_in_pt_2.hcalFraction()
            tofill["pi_plus_sorted_in_pt_3_hcalFraction"] = pi_plus_sorted_in_pt_3.hcalFraction()
            tofill["pi_minus_sorted_in_pt_1_hcalFraction"] = pi_minus_sorted_in_pt_1.hcalFraction()
            tofill["pi_minus_sorted_in_pt_2_hcalFraction"] = pi_minus_sorted_in_pt_2.hcalFraction()
            tofill["pi_minus_sorted_in_pt_3_hcalFraction"] = pi_minus_sorted_in_pt_3.hcalFraction()
            
            tofill["pi_plus_sorted_in_pt_1_phiAtVtx"] = pi_plus_sorted_in_pt_1.phiAtVtx()
            tofill["pi_plus_sorted_in_pt_2_phiAtVtx"] = pi_plus_sorted_in_pt_2.phiAtVtx()
            tofill["pi_plus_sorted_in_pt_3_phiAtVtx"] = pi_plus_sorted_in_pt_3.phiAtVtx()
            tofill["pi_minus_sorted_in_pt_1_phiAtVtx"] = pi_minus_sorted_in_pt_1.phiAtVtx()
            tofill["pi_minus_sorted_in_pt_2_phiAtVtx"] = pi_minus_sorted_in_pt_2.phiAtVtx()
            tofill["pi_minus_sorted_in_pt_3_phiAtVtx"] = pi_minus_sorted_in_pt_3.phiAtVtx()
            
            tofill["pi_plus_sorted_in_pt_1_phiAtVtxMinusPhi"] = pi_plus_sorted_in_pt_1.phiAtVtx() - pi_plus_sorted_in_pt_1.phi()
            tofill["pi_plus_sorted_in_pt_2_phiAtVtxMinusPhi"] = pi_plus_sorted_in_pt_2.phiAtVtx() - pi_plus_sorted_in_pt_2.phi()
            tofill["pi_plus_sorted_in_pt_3_phiAtVtxMinusPh"] = pi_plus_sorted_in_pt_3.phiAtVtx() - pi_plus_sorted_in_pt_3.phi()
            tofill["pi_minus_sorted_in_pt_1_phiAtVtxMinusPhi"] = pi_minus_sorted_in_pt_1.phiAtVtx() - pi_minus_sorted_in_pt_1.phi()
            tofill["pi_minus_sorted_in_pt_2_phiAtVtxMinusPhi"] = pi_minus_sorted_in_pt_2.phiAtVtx() - pi_minus_sorted_in_pt_2.phi()
            tofill["pi_minus_sorted_in_pt_1_phiAtVtxMinusPhi"] = pi_minus_sorted_in_pt_3.phiAtVtx() - pi_minus_sorted_in_pt_3.phi()
             
            # tofill["pi_plus_sorted_in_pt_1_vertexRef"] = float(pi_plus_sorted_in_pt_1.vertexRef())
#             tofill["pi_plus_sorted_in_pt_2_vertexRef"] = float(pi_plus_sorted_in_pt_2.vertexRef())
#             tofill["pi_plus_sorted_in_pt_3_vertexRef"] = float(pi_plus_sorted_in_pt_3.vertexRef())
#             tofill["pi_minus_sorted_in_pt_1_vertexRef"] = float(pi_minus_sorted_in_pt_1.vertexRef())
#             tofill["pi_minus_sorted_in_pt_2_vertexRef"] = float(pi_minus_sorted_in_pt_2.vertexRef())
#             tofill["pi_minus_sorted_in_pt_3_vertexRef"] = float(pi_minus_sorted_in_pt_3.vertexRef())

#            print "(pi_plus_sorted_in_pt_1.vertexRef():", pi_plus_sorted_in_pt_1.vertexRef()
 #           print "(pi_plus_sorted_in_pt_1.vertexRef().px():", pi_plus_sorted_in_pt_1.vertexRef().px()
              
            #print "(*pi_plus_sorted_in_pt_1.vertexRef():", *(pi_plus_sorted_in_pt_1.vertexRef())
  #          print "pi_plus_sorted_in_pt_1.vertex().X():", pi_plus_sorted_in_pt_1.vertex().X()
  
            print "pi_plus_sorted_in_pt_1.pvAssociationQuality():",pi_plus_sorted_in_pt_1.pvAssociationQuality()
            
            #local pi object stuff
            
            tofill["pi_plus_sorted_in_local_pt_1_pdgID"] = pi_plus_sorted_in_local_pt_1.pdgId()
            tofill["pi_plus_sorted_in_local_pt_2_pdgID"] = pi_plus_sorted_in_local_pt_2.pdgId()
            tofill["pi_plus_sorted_in_local_pt_3_pdgID"] = pi_plus_sorted_in_local_pt_3.pdgId()
            tofill["pi_minus_sorted_in_local_pt_1_pdgID"] = pi_minus_sorted_in_local_pt_1.pdgId()
            tofill["pi_minus_sorted_in_local_pt_2_pdgID"] = pi_minus_sorted_in_local_pt_2.pdgId()
            tofill["pi_minus_sorted_in_local_pt_3_pdgID"] = pi_minus_sorted_in_local_pt_3.pdgId()
            
            tofill["pi_plus_sorted_in_local_pt_1_charge"] = pi_plus_sorted_in_local_pt_1.charge()
            tofill["pi_plus_sorted_in_local_pt_2_charge"] = pi_plus_sorted_in_local_pt_2.charge()
            tofill["pi_plus_sorted_in_local_pt_3_charge"] = pi_plus_sorted_in_local_pt_3.charge()
            tofill["pi_minus_sorted_in_local_pt_1_charge"] = pi_minus_sorted_in_local_pt_1.charge()
            tofill["pi_minus_sorted_in_local_pt_2_charge"] = pi_minus_sorted_in_local_pt_2.charge()
            tofill["pi_minus_sorted_in_local_pt_3_charge"] = pi_minus_sorted_in_local_pt_3.charge()
            
            tofill["pi_plus_sorted_in_local_pt_1_hcalFraction"] = pi_plus_sorted_in_local_pt_1.hcalFraction()
            tofill["pi_plus_sorted_in_local_pt_2_hcalFraction"] = pi_plus_sorted_in_local_pt_2.hcalFraction()
            tofill["pi_plus_sorted_in_local_pt_3_hcalFraction"] = pi_plus_sorted_in_local_pt_3.hcalFraction()
            tofill["pi_minus_sorted_in_local_pt_1_hcalFraction"] = pi_minus_sorted_in_local_pt_1.hcalFraction()
            tofill["pi_minus_sorted_in_local_pt_2_hcalFraction"] = pi_minus_sorted_in_local_pt_2.hcalFraction()
            tofill["pi_minus_sorted_in_local_pt_3_hcalFraction"] = pi_minus_sorted_in_local_pt_3.hcalFraction()
            
            tofill["pi_plus_sorted_in_local_pt_1_phiAtVtx"] = pi_plus_sorted_in_local_pt_1.phiAtVtx()
            tofill["pi_plus_sorted_in_local_pt_2_phiAtVtx"] = pi_plus_sorted_in_local_pt_2.phiAtVtx()
            tofill["pi_plus_sorted_in_local_pt_3_phiAtVtx"] = pi_plus_sorted_in_local_pt_3.phiAtVtx()
            tofill["pi_minus_sorted_in_local_pt_1_phiAtVtx"] = pi_minus_sorted_in_local_pt_1.phiAtVtx()
            tofill["pi_minus_sorted_in_local_pt_2_phiAtVtx"] = pi_minus_sorted_in_local_pt_2.phiAtVtx()
            tofill["pi_minus_sorted_in_local_pt_3_phiAtVtx"] = pi_minus_sorted_in_local_pt_3.phiAtVtx()
            
            tofill["pi_plus_sorted_in_local_pt_1_phiAtVtxMinusPhi"] = pi_plus_sorted_in_local_pt_1.phiAtVtx() - pi_plus_sorted_in_local_pt_1.phi()
            tofill["pi_plus_sorted_in_local_pt_2_phiAtVtxMinusPhi"] = pi_plus_sorted_in_local_pt_2.phiAtVtx() -  pi_plus_sorted_in_local_pt_2.phi()
            tofill["pi_plus_sorted_in_local_pt_3_phiAtVtxMinusPhi"] = pi_plus_sorted_in_local_pt_3.phiAtVtx() - pi_plus_sorted_in_local_pt_3.phi()
            tofill["pi_minus_sorted_in_local_pt_1_phiAtVtxMinusPhi"] = pi_minus_sorted_in_local_pt_1.phiAtVtx() - pi_minus_sorted_in_local_pt_1.phi()
            tofill["pi_minus_sorted_in_local_pt_2_phiAtVtxMinusPhi"] = pi_minus_sorted_in_local_pt_2.phiAtVtx() - pi_minus_sorted_in_local_pt_2.phi()
            tofill["pi_minus_sorted_in_local_pt_3_phiAtVtxMinusPhi"] = pi_minus_sorted_in_local_pt_3.phiAtVtx() - pi_minus_sorted_in_local_pt_3.phi()
            
            
            tofill["m00_taup"] = m00_taup
            tofill["m01_taup"] = m01_taup 
            tofill["m02_taup"] = m02_taup
            tofill["m10_taup"] = m10_taup
            tofill["m11_taup"] = m11_taup
            tofill["m12_taup"] = m12_taup
            tofill["m20_taup"] = m20_taup
            tofill["m21_taup"] = m21_taup
            tofill["m22_taup"] = m22_taup
            
            
            tofill["m00_taum"] = m00_taum
            tofill["m01_taum"] = m01_taum
            tofill["m02_taum"] = m02_taum
            tofill["m10_taum"] = m10_taum
            tofill["m11_taum"] = m11_taum
            tofill["m12_taum"] = m12_taum
            tofill["m20_taum"] = m20_taum
            tofill["m21_taum"] = m21_taum
            tofill["m22_taum"] = m22_taum
            
            
            tofill["in_00_antinu"] = in_00_antinu
            tofill["in_10_antinu"] = in_10_antinu
            tofill["in_20_antinu"] = in_20_antinu
            tofill["in_00_nu"] = in_00_nu
            tofill["in_10_nu"] = in_10_nu
            tofill["in_20_nu"] = in_20_nu

            tofill["out_00_antinu"] = out_00_antinu
            tofill["out_10_antinu"] = out_10_antinu
            tofill["out_20_antinu"] = out_20_antinu
            tofill["out_00_nu"] = out_00_nu
            tofill["out_10_nu"] = out_10_nu
            tofill["out_20_nu"] = out_20_nu
            
            tofill["first_eq_antinu"] = first_eq_antinu
            tofill["second_eq_antinu"] = second_eq_antinu
            tofill["third_eq_antinu"] = third_eq_antinu 
            
            tofill["first_eq_nu"] = first_eq_nu
            tofill["second_eq_nu"] = second_eq_nu
            tofill["third_eq_nu"] = third_eq_nu 
            
            
             
          
            
             #actually fill tree
            ntuple.Fill(array('f', tofill.values()))      
             #print "candMatchPi2Info_tau_pdgID_plus_index_list is:", candMatchPi2Info_tau_pdgID_plus_index_list
             #print "candMatchPi2Info_tau_pdgID_plus_index_list[0] is:", candMatchPi2Info_tau_pdgID_plus_index_list[0]
#             

        #test stuff to comment out later
        # print "len(pi1List_tau_pdgID_plus) is:", len(pi1List_tau_pdgID_plus)
#         #genPi1_plus_list = goodEvent_gen_pi1s['+']
#         genPi1_plus = goodEvent_gen_pi1s['+'][0]
#         print genPi1_plus.pdgId()
#         print "len(pi1List_tau_pdgID_minus) is:", len(pi1List_tau_pdgID_minus)

print "eventHasGoodGenUpsilonCount is:", eventHasGoodGenUpsilonCount 
print "eventDoesNOTHaveGoodGenUpsilonCount is:", eventDoesNOTHaveGoodGenUpsilonCount
print "eventHasGenPiOutsideEtaAcceptCount is:", eventHasGenPiOutsideEtaAcceptCount
print "eventHasMatchedUpsilonCount is:", eventHasMatchedUpsilonCount 
print "tau_pdgID_plus_has_neuPiCount is:", tau_pdgID_plus_has_neuPiCount
print "tau_pdgID_minus_has_neuPiCount is:", tau_pdgID_minus_has_neuPiCount

if excludeTausWithNeutralPiInDecayChain:
    print "number of events excluded at GEN LEVEL because eventHasTauWithNeutralPiInDecayChain and you chose to exclude these events and NOT consider them for matching to RECO is:", eventHasTauWithNeutralPiInDecayChainCount
else:
    print "number of events included at GEN LEVEL even though eventHasTauWithNeutralPiInDecayChain because you chose to keep these events and consider them for matching to RECO  is:", eventHasTauWithNeutralPiInDecayChainCount



file_out.cd()
#Write 0to30 tau pt regime efficiency histos
h_num_gen_tau_pt_matched_pdgID_plus_0to30pt.Write()
h_num_gen_tau_pt_matched_pdgID_minus_0to30pt.Write()
h_den_gen_tau_pt_all_pdgID_plus_0to30pt.Write()
h_den_gen_tau_pt_all_pdgID_minus_0to30pt.Write()

#Efficiency histograms 
#Use TEfficiency class to make efficiency histograms as described by John Hakala
if ROOT.TEfficiency().CheckConsistency(h_num_gen_tau_pt_matched_pdgID_plus_0to30pt, h_den_gen_tau_pt_all_pdgID_plus_0to30pt):
    plot_Eff_tau_pt_pdgID_plus_0to30pt = ROOT.TEfficiency(h_num_gen_tau_pt_matched_pdgID_plus_0to30pt,  h_den_gen_tau_pt_all_pdgID_plus_0to30pt)
    plot_Eff_tau_pt_pdgID_plus_0to30pt.SetName("plot_Eff_tau_pt_pdgID_plus_0to30pt")
    plot_Eff_tau_pt_pdgID_plus_0to30pt.Write()
    #print "plot_Eff_tau_pt_pdgID_plus_0to30pt is:", plot_Eff_tau_pt_pdgID_plus_0to30pt

if ROOT.TEfficiency().CheckConsistency(h_num_gen_tau_pt_matched_pdgID_minus_0to30pt, h_den_gen_tau_pt_all_pdgID_minus_0to30pt):
    plot_Eff_tau_pt_pdgID_minus_0to30pt = ROOT.TEfficiency(h_num_gen_tau_pt_matched_pdgID_minus_0to30pt, h_den_gen_tau_pt_all_pdgID_minus_0to30pt)
    plot_Eff_tau_pt_pdgID_minus_0to30pt.SetName("plot_Eff_tau_pt_pdgID_minus_0to30pt")
    plot_Eff_tau_pt_pdgID_minus_0to30pt.Write()


#Write 30to50 tau pt regime eff histos
h_den_gen_tau_pt_all_pdgID_plus_30to50pt.Write()
h_num_gen_tau_pt_matched_pdgID_plus_30to50pt.Write()
h_den_gen_tau_pt_all_pdgID_minus_30to50pt.Write()
h_num_gen_tau_pt_matched_pdgID_minus_30to50pt.Write()

if ROOT.TEfficiency().CheckConsistency(h_num_gen_tau_pt_matched_pdgID_plus_30to50pt,h_den_gen_tau_pt_all_pdgID_plus_30to50pt):
    plot_Eff_tau_pt_pdgID_plus_30to50pt = ROOT.TEfficiency(h_num_gen_tau_pt_matched_pdgID_plus_30to50pt,h_den_gen_tau_pt_all_pdgID_plus_30to50pt)
    plot_Eff_tau_pt_pdgID_plus_30to50pt.SetName("plot_Eff_tau_pt_pdgID_plus_30to50pt")
    plot_Eff_tau_pt_pdgID_plus_30to50pt.Write()

if ROOT.TEfficiency().CheckConsistency(h_num_gen_tau_pt_matched_pdgID_minus_30to50pt,h_den_gen_tau_pt_all_pdgID_minus_30to50pt):
    plot_Eff_tau_pt_pdgID_minus_30to50pt = ROOT.TEfficiency(h_num_gen_tau_pt_matched_pdgID_minus_30to50pt,h_den_gen_tau_pt_all_pdgID_minus_30to50pt)
    plot_Eff_tau_pt_pdgID_minus_30to50pt.SetName("plot_Eff_tau_pt_pdgID_minus_30to50pt")
    plot_Eff_tau_pt_pdgID_minus_30to50pt.Write()

print "h_num_gen_tau_pt_matched_pdgID_plus_0to30pt.GetBinContent(16) is:", h_num_gen_tau_pt_matched_pdgID_plus_0to30pt.GetBinContent(16)
print "h_num_gen_tau_pt_matched_pdgID_minus_0to30pt.GetBinContent(16) is:", h_num_gen_tau_pt_matched_pdgID_minus_0to30pt.GetBinContent(16)
print "h_den_gen_tau_pt_all_pdgID_plus_0to30pt.GetBinContent(16) is:", h_den_gen_tau_pt_all_pdgID_plus_0to30pt.GetBinContent(16)
print "h_den_gen_tau_pt_all_pdgID_minus_0to30pt.GetBinContent(16) is:", h_den_gen_tau_pt_all_pdgID_minus_0to30pt.GetBinContent(16)
print "h_den_gen_tau_pt_all_pdgID_plus_30to50pt.GetBinContent(3) is:", h_den_gen_tau_pt_all_pdgID_plus_30to50pt.GetBinContent(3)
print "h_num_gen_tau_pt_matched_pdgID_plus_30to50pt.GetBinContent(3) is:", h_num_gen_tau_pt_matched_pdgID_plus_30to50pt.GetBinContent(3)
print "h_den_gen_tau_pt_all_pdgID_minus_30to50pt.GetBinContent(3) is:", h_den_gen_tau_pt_all_pdgID_minus_30to50pt.GetBinContent(3)

print "h_num_gen_tau_pt_matched_pdgID_minus_30to50pt.GetBinContent(3) is:", h_num_gen_tau_pt_matched_pdgID_minus_30to50pt.GetBinContent(3)
print "h_num_gen_tau_pt_matched_pdgID_plus_0to30pt.GetEntries() is:", h_num_gen_tau_pt_matched_pdgID_plus_0to30pt.GetEntries()
print "h_den_gen_tau_pt_all_pdgID_plus_0to30pt.GetEntries() is:", h_den_gen_tau_pt_all_pdgID_plus_0to30pt.GetEntries()

ntuple.Write()
file_out.Close()  

