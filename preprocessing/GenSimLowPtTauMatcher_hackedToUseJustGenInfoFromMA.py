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

#parser = ArgumentParser()
#parser.add_argument('--suffix', default='_', help='Suffix to be added to the end of out the output file name, such as cartesian_upsilon_taus<_suffix>, where the _suffix would be some useful suffix')


#################
#ORDER OF VECTORS:
#Gen neu, gen antineu, reco pion +, reco pion -
#################

def isAncestor(a,p) :
    if a == p :
        return True
    for i in xrange(0,p.numberOfMothers()) :
        if isAncestor(a,p.mother(i)) :
            return True
    return False


def computeCosine (Vx, Vy, Vz,
                   Wx, Wy, Wz):

    Vnorm = math.sqrt(Vx*Vx + Vy*Vy + Vz*Vz)
    Wnorm = math.sqrt(Wx*Wx + Wy*Wy + Wz*Wz)
    VdotW = Vx*Wx + Vy*Wy + Vz*Wz

    if Vnorm > 0. and Wnorm > 0.:
        cosAlpha = VdotW / (Vnorm * Wnorm)
    else:
        cosAlpha = -99.

    return cosAlpha




def has_mcEvent_match(g_elec, reco_ele, delta_r_cut, charge_label):
    min_delta_r = 9.9
    delta_r = 10.

    for ii, reco in enumerate(reco_ele):
        if reco.pt() > 0.1:
            reco_vec = ROOT.TLorentzVector()
            reco_vec.SetPtEtaPhiM(reco_ele.pt(), reco_ele.eta(), reco_ele.phi(), mass_pion)

            delta_r = g_elec.DeltaR(reco_vec)
            if delta_r < min_delta_r:
                min_delta_r = delta_r
                if delta_r < delta_r_cut:
                    #return True
                    if not abs(charge_label):
                        return (delta_r, ii)
                    elif charge_label*reco.charge() >0:
                        return (delta_r, ii)

    return (False, 100)

#Note: this function never actually gets used in the code
from FWCore.ParameterSet.VarParsing import VarParsing
options = VarParsing ('python')
options.register('suffix',
                        '',
                        VarParsing.multiplicity.singleton,
                        VarParsing.varType.string,
                        'suffix to append to out file name')


options.parseArguments()
print options

handlePruned  = Handle ("std::vector<reco::GenParticle>") #same as in the full-on LowPtTauMatcher that has Reco stuff
labelPruned = ("prunedGenParticles") #for Reco, different from the pure unhacked GenSimLowPtTauMatcher, same as the regular LowPtTauMatcher

#handleReco = Handle ("std::vector<pat::PackedCandidate>")
#recoLabel = ("packedPFCandidates")

#lostLabel = ("lostTracks")
#the above three lines refer to reco stuff

handleMET = Handle ("std::vector<pat::MET>") #for Reco, different from pure unhacked GenSimLowPtTauMatcher, same as the regular LowPtTauMatcher
labelMET = ("slimmedMETs") #for Reco, different from pure unhacked GenSimLowPtTauMatcher, same as the regular LowPtTauMatcher



# Create histograms, etc.
#ROOT.gROOT.SetBatch()        # don't pop up canvases
#ROOT.gROOT.SetStyle('Plain') # white background

# taum_branches = [
# 'pi_minus1_px',
# 'pi_minus1_py',
# 'pi_minus1_pz',
# 'pi_minus2_px',
# 'pi_minus2_py',
# 'pi_minus2_pz',
# 'pi_minus3_px',
# 'pi_minus3_py',
# 'pi_minus3_pz',
# 'taum_m',
# 'taum_neu_px',
# 'taum_neu_py',
# 'taum_neu_pz',
# ]
# 
# taup_branches = [
# 'pi_plus1_px',
# 'pi_plus1_py',
# 'pi_plus1_pz',
# 'pi_plus2_px',
# 'pi_plus2_py',
# 'pi_plus2_pz',
# 'pi_plus3_px',
# 'pi_plus3_py',
# 'pi_plus3_pz',
# 'taup_m',
# 'taup_neu_px',
# 'taup_neu_py',
# 'taup_neu_pz',
# ]
# 
# branches = taum_branches + taup_branches
# branches.append('upsilon_m')
# 

taum_branches =[
'pi_minus1_pt',
'pi_minus1_eta',
'pi_minus1_phi',
#'pi_minus1_pt',
#'pi_minus1_eta',
#'pi_minus1_phi',
'pi_minus2_pt',
'pi_minus2_eta',
'pi_minus2_phi',
'pi_minus3_pt',
'pi_minus3_eta',
'pi_minus3_phi',
'taum_neu_pt',
'taum_neu_eta',
'taum_neu_phi',
'taum_mass'
]

taup_branches =[
 'pi_plus1_pt',
 'pi_plus1_eta',
 'pi_plus1_phi',
# 'pi_plus1_pt',
# 'pi_plus1_eta',
# 'pi_plus1_phi',
 'pi_plus2_pt',
 'pi_plus2_eta',
 'pi_plus2_phi',
 'pi_plus3_pt',
 'pi_plus3_eta',
 'pi_plus3_phi',
 'taup_neu_pt',
 'taup_neu_eta',
 'taup_neu_phi',
 'taup_mass'
]

# branch names can stay the same
branches = taum_branches + taup_branches
branches.append('upsilon_m')
branches.append('neutrino_pt')
branches.append('neutrino_phi')
branches.append('neutrino_eta')
branches.append('antineutrino_pt')
branches.append('antineutrino_phi')
branches.append('antineutrino_eta')

suffix = options.suffix
print 'suffix is:', suffix
file_out = ROOT.TFile('cartesian_upsilon_taus_%s.root'%(suffix), 'recreate')
file_out.cd()

taup_ntuple = ROOT.TNtuple('tree', 'tree', ':'.join(taum_branches)) #ok I think this was a mistake, because testing shows this adds the minus branches to the taup_ntuple, which does not make sense, but we get saved by the fact that we want the total anyway and only end up using ntuple, which is the combination of the two, so it is a benign mistake, no need to worry :)

#print "taup_ntuple is:", taup_ntuple
#print "branches in taup_ntuple are:", taup_ntuple.Print()

taum_ntuple = ROOT.TNtuple('tree', 'tree', ':'.join(taup_branches))
#print "taum_ntuple is:", taum_ntuple
#print "branches in taum_ntuple are:", taum_ntuple.Print()

ntuple = ROOT.TNtuple('tree', 'tree', ':'.join(branches))
#print 'ntuple is:', ntuple
#print "branches in ntuple are:", ntuple.Print()

verb = False

upsilon_id = 553
tau_id = 15
pion_id = 211
tau_neu_id = 16
neutral_pion_id = 111
photon_id = 22

# Events takes either
# - single file name
# - list of file names
# - VarParsing options

# use Varparsing object
events = Events (options)
nTot = 0
tagUpsilonCount = 0
finalBreakCount=0
ZOINKSCount = 0

for event in events:
    print 'Processing event: %i...'%(nTot)

    # Generated stuff
    event.getByLabel(labelPruned, handlePruned)
    pruned_gen_particles = handlePruned.product()

#    event.getByLabel(recoLabel, handleReco)
#   pf_Particles = handleReco.product()

#   event.getByLabel(lostLabel, handleReco)
#    lost_Particles = handleReco.product()
    
    event.getByLabel(labelMET, handleMET)
    met = handleMET.product().front()

#    reco_Particles = []

#    for p in pf_Particles:
#       reco_Particles.append(p)
#    for p in lost_Particles:
#       reco_Particles.append(p)



    gen_upsilon = []
    gen_taum = []
    gen_taup = []
    gen_pionm = []
    gen_pionp = []
    gen_neu = []
    gen_antineu = []
    gen_pionn = []
    gen_photons = []


    # matched_pionp = []
#     matched_pionm = []
#     matched_photonp = []
#     matched_photonm = []
# 
#     lost_pions = []
    taum_has_pionn = False
    taup_has_pionn = False

    # Filter reco particles
#     rec_pionm = []
#     rec_pionp = []
#     rec_pions = []
#     rec_photons = []



    
    # Tagging particles in gen particles
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
            gen_antineu.append(pp)
        elif pp.pdgId() == neutral_pion_id:
            gen_pionn.append(pp)
        elif pp.pdgId() == photon_id:
            gen_photons.append(pp)



        # Tagging reco particles
    # for pp in reco_Particles:
#         if abs(pp.pdgId()) == pion_id:
#             rec_pions.append(pp)
#         elif abs(pp.pdgId()) == photon_id:
#             rec_photons.append(pp)
# 
#     for pp in lost_Particles:
#         if abs(pp.pdgId()) == pion_id:
#             lost_pions.append(pp)

    neu_etaCheck = False
    antineu_etaCheck = False

    for pp in gen_neu:
        if abs(pp.eta()) > math.pi:
            neu_etaCheck = True
    for pp in gen_antineu:
        if abs(pp.eta()) > math.pi:
            antineu_etaCheck = True



    leps_mydecay = []
    g_taum = None
    g_taup = None
    g_taum_pions = []
    g_taup_pions = []
    g_taum_pionn = []
    g_taup_pionn = []
    g_taum_photons = []
    g_taup_photons = []
    g_tau_neum = None
    g_tau_neup = None





    found_taup = False
    found_taum = False

    found_anti_neu = False
    found_neu = False

    tag_upsilon = False
    tag_taup = False
    tag_taum = False

    # comb upsilon particles, see if decayed into t-t+
    for p in gen_upsilon:
#        print 'len(gen_upsilon) is:', len(gen_upsilon)
#        if (len(gen_upsilon)) != 1: 
#            print 'len(gen_upsilon) does NOT equal 1, len(gen_upsilon) is:', len(gen_upsilon)
#            print 'ZOINKS'
#            ZOINKSCount +=1
        found_taup = False
        found_taum = False
        #  normal tau (negative, should have 2- pions and 1+ pion)
        # comb list of antitau generated
        for pa in gen_taup:
            mother = pa.mother(0)
            if mother and isAncestor(p, mother):
                g_taup = pa
                found_taup = True
                positive_found = False
                negative_found = 0
                g_taum_pions = []
                
                # Comb list of neutrinos from tau decay
                for ne in gen_neu:
                    mother = ne.mother(0)
                    if mother and isAncestor(p, mother):
                        found_anti_neu = True
                        g_tau_neum = ne
                for pn in gen_pionn:
                    mother = pn.mother(0)
                    if mother and isAncestor(pa, mother):
                        g_taum_pionn.append(pn)
                        for ph in gen_photons:
                            mother = ph.mother(0)
                            if mother and isAncestor(pn, mother):
                                g_taum_photons.append(ph)
                for pm in gen_pionm:
                    mother = pm.mother(0)
                    if mother and isAncestor(pa, mother):
                        negative_found += 1
                        g_taum_pions.append(pm)
                for pp in gen_pionp:
                    mother = pp.mother(0)
                    if mother and isAncestor(pa, mother):
                        positive_found = True
                        g_taum_pions.append(pp)

                if found_anti_neu and positive_found and negative_found == 2:
                    tag_taum = True 
#                    if tag_taum: print "tag_taum is True!"
                    break

        #  anti tau (positive, should have 1- pion and 2+ pions)
        for pa in gen_taum:
#            print 'len (gen_taum) is:', len(gen_taum)
#            print "I entered here 1"
            mother = pa.mother(0)
            if mother and isAncestor(p, mother):
                g_taum = pa
                found_taum = True
                positive_found = 0
                negative_found = False
                g_taup_pions = []

                for ne in gen_antineu:
                    mother = ne.mother(0)
                    if mother and isAncestor(p, mother):
                        found_neu = True
                        g_tau_neup = ne

                for pn in gen_pionn:
                    mother = pn.mother(0)
                    if mother and isAncestor(pa, mother):
                        g_taup_pionn.append(pn)
                        for ph in gen_photons:
                            mother = ph.mother(0)
                            if mother and isAncestor(pn, mother):
                                g_taum_photons.append(ph)

                for pm in gen_pionm:
                    mother = pm.mother(0)
                    if mother and isAncestor(pa, mother):
                        negative_found = True
                        g_taup_pions.append(pm)
                for pp in gen_pionp:
                    mother = pp.mother(0)
                    if mother and isAncestor(pa, mother):
                        positive_found += 1
                        g_taup_pions.append(pp)

                if found_neu and positive_found == 2 and negative_found:
                    tag_taup = True
                    if tag_taup: print "tag_taup is True!"
                    #print "g_taup_pions is:", g_taup_pions
                    #for thing in g_taup_pions: print thing
                    break

        if found_taum and found_taup and len(g_taum_pions) == 3 and len(g_taup_pions) == 3 and found_neu and found_anti_neu:
            #print "This event has ", len(g_taup_photons) / 2, " taup pionn"
            #print "This event has ", len(g_taum_photons) / 2, " taum pionn"
            leps_mydecay.append(g_taum)
            leps_mydecay.append(g_taup)
            leps_mydecay += g_taum_pions
            leps_mydecay += g_taup_pions
            leps_mydecay += g_taum_pionn
            leps_mydecay += g_taup_pionn
            leps_mydecay.append(gen_neu)
            leps_mydecay.append(gen_antineu)
            tag_upsilon = True
            if tag_upsilon: print "tag_upsilon is True before neutral pion check!"
            taup_has_pionn = len(g_taup_pionn) != 0
            taum_has_pionn = len(g_taum_pionn) != 0
            if  not ((taup_has_pionn) or  (taum_has_pionn)): 
               
                print 'upsilon is good after checking for neutral pions, none found in the decay chain (though of course lots of them will be hanging around the event)!'
                break #the upsilon is good, we can leave this loop
             #sadness, the upsilon is bad
            tag_upsilon = False
            # no need for an explicit continue any more since we're now at the natural end of the loop anyway


	
	
	
   
            # tau pions
            # for genpi in g_taum_pions:
#                 min_ind = None
#                 min_deltaR = 9999
#                 matched_x = False
#                 for i in range(0, len(
#                         rec_pions)):  # check if particles correspond to one another based on delta r, charge, and delta pt
#                     recpi = rec_pions[i]
#                     gen_lv = TLorentzVector()
#                     gen_lv.SetPtEtaPhiM(genpi.pt(), genpi.eta(), genpi.phi(), 0.139)
#                     rec_lv = TLorentzVector()
#                     rec_lv.SetPtEtaPhiM(recpi.pt(), recpi.eta(), recpi.phi(), 0.139)
#                     deltaR = gen_lv.DeltaR(rec_lv)
#                     deltaPT = (rec_lv.Pt() - gen_lv.Pt()) / gen_lv.Pt()
#                     
#                     if recpi.pdgId() == genpi.pdgId() and abs(deltaR) < 0.1 and abs(deltaPT) < 0.3 and deltaR < min_deltaR and abs(genpi.eta()) < 2.5 and not recpi in matched_pionm:
#                         min_ind = i
#                         matched_x = True
#                         min_deltaR = deltaR
#                 if matched_x:
#                     matched_pionm.append(rec_pions[min_ind])
#        Get rid of matching gen to reco, as we do NOT have reco...if we did, we would be running the full-on code 
           #  if taum_has_pionn:
#                 for genph in g_taum_photons:
#                 
#                     min_ind = None
#                     min_deltaR = 9999
#                     matched_x = False
#                     for i in range(0, len(rec_photons)):
#                         recph = rec_photons[i]
#                         gen_lv = TLorentzVector()
#                         gen_lv.SetPtEtaPhiM(genph.pt(), genph.eta(), genph.phi(), 0)
#                         rec_lv = TLorentzVector()
#                         rec_lv.SetPtEtaPhiM(genph.pt(), genph.eta(), genph.phi(), 0)
#                         deltaR = gen_lv.DeltaR(rec_lv)
#                         deltaPT = (rec_lv.Pt() - gen_lv.Pt()) / gen_lv.Pt()
#                         
#                         if abs(deltaR) < 0.1 and abs(deltaPT) < 0.1 and deltaR < min_deltaR and abs(genph.eta()) < 2.5 and not recph in matched_photonm:
#                             min_ind = i
#                             matched_x = True
#                             min_deltaR = deltaR
#                     if matched_x:
#                         matched_photonm.append(rec_photons[min_ind])
#      same comment, get rid of matching to reco, we do NOT have it...if we did, we would be running the full-on code               
                    
        #antitau pions
#             for genpi in g_taup_pions:
# 
#                 min_ind = None
#                 min_deltaR = 99999
#                 matched_x = False
#                 for i in range(0, len(
#                         rec_pions)):  # check if particles correspond to one another based on delta r, charge, and delta pt
#                     recpi = rec_pions[i]
#                     gen_lv = TLorentzVector()
#                     gen_lv.SetPtEtaPhiM(genpi.pt(), genpi.eta(), genpi.phi(), 0.139)
#                     rec_lv = TLorentzVector()
#                     rec_lv.SetPtEtaPhiM(recpi.pt(), recpi.eta(), recpi.phi(), 0.139)
#                     deltaR = gen_lv.DeltaR(rec_lv)
#                     deltaPT = (rec_lv.Pt() - gen_lv.Pt()) / gen_lv.Pt()
#                     if recpi.pdgId() == genpi.pdgId() and abs(deltaR) < 0.1 and abs(deltaPT) < 0.3 and deltaR < min_deltaR and abs(genpi.eta()) < 2.5 and not recpi in matched_pionp:
#                         min_ind = i
#                         min_deltaR = deltaR
#                         matched_x = True
#                 if matched_x:
#                         matched_pionp.append(rec_pions[min_ind])
# same comment, get rid of matching to reco, we do NOT have it...if we did, we would be running the full-on code

  #           if taup_has_pionn:
#                 for genph in g_taup_photons:
#                     min_ind = None
#                     min_deltaR = 9999
#                     matched_x = False
#                     for i in range(0, len(rec_photons)):
#                         recph = rec_photons[i]
#                         gen_lv = TLorentzVector()
#                         gen_lv.SetPtEtaPhiM(genph.pt(), genph.eta(), genph.phi(), 0)
#                         rec_lv = TLorentzVector()
#                         rec_lv.SetPtEtaPhiM(genph.pt(), genph.eta(), genph.phi(), 0)
#                         deltaR = gen_lv.DeltaR(rec_lv)
#                         deltaPT = (rec_lv.Pt() - gen_lv.Pt()) / gen_lv.Pt()
#                         if abs(deltaR) < 0.1 and abs(deltaPT) < 0.3 and deltaR < min_deltaR and abs(genph.eta()) < 2.5 and not recph in matched_photonp:
#                             min_ind = i
#                             matched_x = True
#                             min_deltaR = deltaR
#                     if matched_x:
#                         matched_photonp.append(rec_photons[min_ind])

#we are are not dealing with neutral pions right now, also we have no reco to match it to anyway because we are not running the full-on code
# 
#             if len(gen_pionn) != 0:
#                 tag_upsilon = False
#                 print "failed the gen_pionn length check"
#                 print 'len(gen_pionn) is:', len(gen_pionn)
#                 print "I entered here 2" #I think this looks like it's sending you back to the pa in gen_taum and pa in gen_taup level but need to check RUBBER DUCK
#                 continue
#             
#             gen_pionn is an event level variable (are there any neutral pions in this event at all? not in the decay chain specifically but hanging around?) so this cut kills everything, I think that is why there is the tag_upsilon switch that I thought was redundant below 
#            if antineu_etaCheck or neu_etaCheck:
#                tag_upsilon = False
#                print "failed the (anti)neu eta check!"
#             
#             if len(matched_pionp) == 3 and len(matched_photonp) % 2 == 0:
#                 tag_taup = True
#             if len(matched_pionm) == 3 and len(matched_photonm) % 2 == 0:
#                 tag_taum = True
#             
#             if len(matched_pionp) + len(matched_pionm) != 6:
#                 tag_upsilon = False
#        print "i reached the final break" # RUBBER DUCK I'm breaking out of the for p in gen_upsilon loop (see L349 (or maybe3 349ish if this code gets updated and I forget to change this comment) for that loop)
#        finalBreakCount += 1
#        break

    nTot += 1
    print "I made it through %s events" %(nTot)
    #if nTot > 1: break
    gen_taup_lv = TLorentzVector()
    gen_taup_lv.SetPtEtaPhiM(gen_taum[0].pt(), gen_taum[0].eta(), gen_taum[0].phi(), 0.139)

    gen_taum_lv = TLorentzVector()
    gen_taum_lv.SetPtEtaPhiM(gen_taup[0].pt(), gen_taup[0].eta(), gen_taup[0].phi(), 0.139)


    neu_lv = TLorentzVector()
    neu_lv.SetPtEtaPhiM(gen_neu[0].pt(), gen_neu[0].eta(), gen_neu[0].phi(), 0)
        
    antineu_lv = TLorentzVector()
    antineu_lv.SetPtEtaPhiM(gen_antineu[0].pt(), gen_antineu[0].eta(), gen_antineu[0].phi(), 0)

    pi_m_lv1 = TLorentzVector()
    pi_m_lv2 = TLorentzVector()
    pi_m_lv3 = TLorentzVector()

    pi_p_lv1 = TLorentzVector()
    pi_p_lv2 = TLorentzVector()
    pi_p_lv3 = TLorentzVector()

    taup_lv = TLorentzVector()
    taum_lv = TLorentzVector()

    taum_pionn_lv = TLorentzVector()
    taup_pionn_lv = TLorentzVector()

#    if tag_taum and tag_taup:
#       tag_upsilon = True
#is this not redundant??? it wasn't for them because of their crazy cut, this was actually what saved them
    
    if tag_taum:
        print("found Tau-")
        #taum_tofill = OrderedDict(zip(taum_branches, [-99.] * len(taum_branches)))
        
        pi_m_lv1.SetPtEtaPhiM(g_taum_pions[0].pt(), g_taum_pions[0].eta(), g_taum_pions[0].phi(), 0.139)
        pi_m_lv2.SetPtEtaPhiM(g_taum_pions[1].pt(), g_taum_pions[1].eta(), g_taum_pions[1].phi(), 0.139)
        pi_m_lv3.SetPtEtaPhiM(g_taum_pions[2].pt(), g_taum_pions[2].eta(), g_taum_pions[2].phi(), 0.139)
        taum_lv = pi_m_lv1 + pi_m_lv2 + pi_m_lv3 + neu_lv
        
       #  """
#         taum_tofill['taum_neu_px'] = neu_lv.Px()
#         taum_tofill['taum_neu_py'] = neu_lv.Py()
#         taum_tofill['taum_neu_pz'] = neu_lv.Pz()
#         """
#         #Add neutral pions to tau lv
#         
#         """
#         if(len(matched_photonm) != 0):
#             print("Tagged ", len(matched_photonm) / 2, " pion neutral particles")
#             print("Photonless tau mass: ", taum_lv.M())
#         
#         
#         temp_ph_lv = TLorentzVector()
#         for ph in matched_photonm:
#             temp_ph_lv.SetPtEtaPhiM(ph.pt(), ph.eta(), ph.phi(), 0)
#             taum_lv += temp_ph_lv
#             taum_pionn_lv += temp_ph_lv
#             print "taum photon E: ", temp_ph_lv.E()
#         """
#         
#         
#         
#         #print "taum mass: ", taum_lv.M()
#         """
#         taum_tofill['pi_minus1_px'] = pi_m_lv1.Px()
#         taum_tofill['pi_minus1_py'] = pi_m_lv1.Py()
#         taum_tofill['pi_minus1_pz'] = pi_m_lv1.Pz()
#         taum_tofill['pi_minus2_px'] = pi_m_lv2.Px()
#         taum_tofill['pi_minus2_py'] = pi_m_lv2.Py()
#         taum_tofill['pi_minus2_pz'] = pi_m_lv2.Pz()
#         taum_tofill['pi_minus3_px'] = pi_m_lv3.Px()
#         taum_tofill['pi_minus3_py'] = pi_m_lv3.Py()
#         taum_tofill['pi_minus3_pz'] = pi_m_lv3.Pz()
# 
#         taum_tofill['taum_pionn_m'] = taum_pionn_lv.M()
#         
#         ptTot = pi_m_lv1.Pt() + pi_m_lv2.Pt() + pi_m_lv3.Pt()
#         
#         taum_tofill['ignore_branch'] = taum_lv.M()
#         
#         taum_tofill['pi_minus1_normpt'] = pi_m_lv1.Pt() / ptTot
#         taum_tofill['pi_minus2_normpt'] = pi_m_lv2.Pt() / ptTot
#         taum_tofill['pi_minus3_normpt'] = pi_m_lv3.Pt() / ptTot
#         
#         taum_ntuple.Fill(array('f', taum_tofill.values()))
#         
#         """

    if tag_taup:
        print("Found tau+")
        #taup_tofill = OrderedDict(zip(taup_branches, [-99.] * len(taup_branches)))
        pi_p_lv1.SetPtEtaPhiM(g_taup_pions[0].pt(), g_taup_pions[0].eta(), g_taup_pions[0].phi(), 0.139)
        pi_p_lv2.SetPtEtaPhiM(g_taup_pions[1].pt(), g_taup_pions[1].eta(), g_taup_pions[1].phi(), 0.139)
        pi_p_lv3.SetPtEtaPhiM(g_taup_pions[2].pt(), g_taup_pions[2].eta(), g_taup_pions[2].phi(), 0.139)
        taup_lv = pi_p_lv1 + pi_p_lv2 + pi_p_lv3 + antineu_lv
        
        
       #  """
#         if(len(matched_photonp) != 0):
#             print("Tagged ", len(matched_photonp) / 2, " pion neutral particles")
#             print("Photonless tau mass: ", taup_lv.M())
#         
#         
#         temp_ph_lv = TLorentzVector()
#         for ph in matched_photonp:
#             temp_ph_lv.SetPtEtaPhiM(ph.pt(), ph.eta(), ph.phi(), 0)
#             taup_lv += temp_ph_lv
# 
#             print "taup photon E: ", temp_ph_lv.E()
#         """
#         """
#         taup_tofill['taup_neu_px'] = antineu_lv.Px()
#         taup_tofill['taup_neu_py'] = antineu_lv.Py()
#         taup_tofill['taup_neu_pz'] = antineu_lv.Pz()
#         
#         taup_tofill['pi_plus1_px'] = pi_p_lv1.Px()
#         taup_tofill['pi_plus1_py'] = pi_p_lv1.Py()
#         taup_tofill['pi_plus1_pz'] = pi_p_lv1.Pz()
#         taup_tofill['pi_plus2_px'] = pi_p_lv2.Px()
#         taup_tofill['pi_plus2_py'] = pi_p_lv2.Py()
#         taup_tofill['pi_plus2_pz'] = pi_p_lv2.Pz()
#         taup_tofill['pi_plus3_px'] = pi_p_lv3.Px()
#         taup_tofill['pi_plus3_py'] = pi_p_lv3.Py()
#         taup_tofill['pi_plus3_pz'] = pi_p_lv3.Pz()
#         
#         ptTot = pi_p_lv1.Pt() + pi_p_lv2.Pt() + pi_p_lv3.Pt()
#         
#         taup_tofill['pi_plus1_normpt'] = pi_p_lv1.Pt() / ptTot
#         taup_tofill['pi_plus2_normpt'] = pi_p_lv2.Pt() / ptTot
#         taup_tofill['pi_plus3_normpt'] = pi_p_lv3.Pt() / ptTot
#         
#         taup_tofill['ignore_branch'] = taup_lv.M()
#         taup_ntuple.Fill(array('f',taup_tofill.values()))
#         """
    

    if tag_upsilon:
        if g_taum_pions[0].eta() > 2.5 or g_taum_pions[0].eta() < -2.5  or g_taum_pions[1].eta() > 2.5 or g_taum_pions[1].eta() < -2.5  or g_taum_pions[2].eta() > 2.5 or g_taum_pions[2].eta() < -2.5  or g_taup_pions[0].eta() > 2.5 or  g_taup_pions[0].eta() < -2.5 or g_taup_pions[1].eta() > 2.5 or g_taup_pions[1].eta() < -2.5 or g_taup_pions[2].eta() > 2.5 or g_taup_pions[2].eta() < -2.5 or g_taum_pions[0].pt() < 0.7 or g_taum_pions[1].pt() < 0.7 or g_taum_pions[2].pt() < 0.7 or  g_taup_pions[0].pt() < 0.7 or  g_taup_pions[1].pt() < 0.7 or g_taup_pions[2].pt() < 0.7:
            print "one of the candidate pions failed the eta cut or failed the pT cut!"
            print "g_taum_pions[0].eta() is:", g_taum_pions[0].eta()
            print "g_taum_pions[1].eta() is:", g_taum_pions[1].eta()
            print "g_taum_pions[2].eta() is:", g_taum_pions[2].eta()
            print "g_taup_pions[0].eta() is:", g_taup_pions[0].eta()
            print "g_taup_pions[1].eta() is:", g_taup_pions[1].eta()
            print "g_taup_pions[2].eta() is:", g_taup_pions[2].eta()
            print "g_taum_pions[0].pt() is:",  g_taum_pions[0].pt()
            print "g_taum_pions[1].pt() is:",  g_taum_pions[1].pt()
            print "g_taum_pions[2].pt() is:",  g_taum_pions[2].pt()
            print "g_taup_pions[0].pt() is:",  g_taup_pions[0].pt()
            print "g_taup_pions[1].pt() is:",  g_taup_pions[1].pt()
            print "g_taup_pions[2].pt() is:",  g_taup_pions[2].pt()
            continue
        print 'Found Upsilon -> tau+ tau- -> pi+*3 pi-*3'
        tagUpsilonCount +=1
        print 'tagUpsilonCount is:', tagUpsilonCount
        tofill = OrderedDict(zip(branches, [-99.] * len(branches)))
#        print "tofill is:", tofill

        upsilon_lv = neu_lv + antineu_lv + pi_m_lv1 + pi_m_lv2 + pi_m_lv3 + pi_p_lv1 + pi_p_lv2 + pi_p_lv3
        # '''
#         tofill['taup_neu_px'] = antineu_lv.Px()
#         tofill['taup_neu_py'] = antineu_lv.Py()
#         tofill['taup_neu_pz'] = antineu_lv.Pz()
#         
#         tofill['pi_plus1_px'] = pi_p_lv1.Px()
#         tofill['pi_plus1_py'] = pi_p_lv1.Py()
#         tofill['pi_plus1_pz'] = pi_p_lv1.Pz()
#         tofill['pi_plus2_px'] = pi_p_lv2.Px()
#         tofill['pi_plus2_py'] = pi_p_lv2.Py()
#         tofill['pi_plus2_pz'] = pi_p_lv2.Pz()
#         tofill['pi_plus3_px'] = pi_p_lv3.Px()
#         tofill['pi_plus3_py'] = pi_p_lv3.Py()
#         tofill['pi_plus3_pz'] = pi_p_lv3.Pz()
#         tofill['taup_m'] = taup_lv.M()
#         
#         tofill['pi_minus1_px'] = pi_m_lv1.Px()
#         tofill['pi_minus1_py'] = pi_m_lv1.Py()
#         tofill['pi_minus1_pz'] = pi_m_lv1.Pz()
#         tofill['pi_minus2_px'] = pi_m_lv2.Px()
#         tofill['pi_minus2_py'] = pi_m_lv2.Py()
#         tofill['pi_minus2_pz'] = pi_m_lv2.Pz()
#         tofill['pi_minus3_px'] = pi_m_lv3.Px()
#         tofill['pi_minus3_py'] = pi_m_lv3.Py()
#         tofill['pi_minus3_pz'] = pi_m_lv3.Pz()
#         
#         tofill['taum_neu_px'] = neu_lv.Px()
#         tofill['taum_neu_py'] = neu_lv.Py()
#         tofill['taum_neu_pz'] = neu_lv.Pz()
#         tofill['taum_m'] = taum_lv.M()
#         
#         tofill['upsilon_m'] = upsilon_lv.M()
#         
#        ntuple.Fill(array('f', tofill.values()))

 
         
        tofill['neutrino_pt'] = gen_neu[0].pt()
        tofill['neutrino_eta'] = gen_neu[0].eta()
        tofill['neutrino_phi'] = gen_neu[0].phi()
        tofill['neutrino_m'] = neu_lv.M()
        tofill['antineutrino_pt'] = gen_antineu[0].pt()
        tofill['antineutrino_eta'] = gen_antineu[0].eta()
        tofill['antineutrino_phi'] = gen_antineu[0].phi()

        tofill['antineutrino_m'] = antineu_lv.M()
        tofill['pi_minus1_pt'] = g_taum_pions[0].pt()
        tofill['pi_minus1_eta'] = g_taum_pions[0].eta()
        tofill['pi_minus1_phi'] = g_taum_pions[0].phi()
        tofill['pi_minus1_m'] = pi_m_lv1.M()
        tofill['pi_minus2_pt'] = g_taum_pions[1].pt()
        tofill['pi_minus2_eta'] = g_taum_pions[1].eta()
        tofill['pi_minus2_phi'] = g_taum_pions[1].phi()
        
        tofill['pi_minus2_m'] = pi_m_lv2.M()
        tofill['pi_minus3_pt'] = g_taum_pions[2].pt()
        tofill['pi_minus3_eta'] = g_taum_pions[2].eta()
        tofill['pi_minus3_phi'] = g_taum_pions[2].phi()
        tofill['pi_minus3_m'] = pi_m_lv3.M()
        
        #end pi minus stuff, begin pi plus stuff
        tofill['pi_plus1_pt'] = g_taup_pions[0].pt()
        tofill['pi_plus1_eta'] = g_taup_pions[0].eta()
        tofill['pi_plus1_phi'] = g_taup_pions[0].phi()
        
        tofill['pi_plus1_m'] = pi_p_lv1.M()
        tofill['pi_plus2_pt'] = g_taup_pions[1].pt()
        tofill['pi_plus2_eta'] = g_taup_pions[1].eta()
        tofill['pi_plus2_phi'] = g_taup_pions[1].phi()
        tofill['pi_plus2_m'] = pi_p_lv2.M()
        tofill['pi_plus3_pt'] = g_taup_pions[2].pt()
        tofill['pi_plus3_eta'] = g_taup_pions[2].eta()
        tofill['pi_plus3_phi'] = g_taup_pions[2].phi()
        tofill['pi_plus3_m'] = pi_p_lv3.M()
     
        
        tofill['upsilon_m'] = upsilon_lv.M()
    
    
        upsilon_no_neu_lv = pi_m_lv1 + pi_m_lv2 + pi_m_lv3 + pi_p_lv1 + pi_p_lv2 + pi_p_lv3
        tofill['upsilon_m_no_neu'] = upsilon_no_neu_lv.M()
        
        taum_no_neu_lv = pi_m_lv1 + pi_m_lv2 + pi_m_lv3
        tofill['taum_m_no_neu'] = taum_no_neu_lv.M()
        
        taup_no_neu_lv = pi_p_lv1 + pi_p_lv2 + pi_p_lv3
        tofill['taup_m_no_neu'] = taup_no_neu_lv.M()
        
        coll_x_plus = taup_no_neu_lv.Pt() / (taup_no_neu_lv.Pt() + met.pt())
        coll_x_minus = taum_no_neu_lv.Pt() / (taum_no_neu_lv.Pt() + met.pt())

        tofill['upsilon_coll_m'] = upsilon_no_neu_lv.M() / math.sqrt(coll_x_minus * coll_x_plus)

        tofill['r_taum_pt'] = taum_lv.Pt()
        tofill['r_taup_pt'] = taup_lv.Pt()
        
        tofill['gen_taum_pt'] = gen_taum_lv.Pt()
        tofill['gen_taup_pt'] = gen_taup_lv.Pt()
        
        ntuple.Fill(array('f', tofill.values()))      
        #print 'ntuple is:', ntuple 

    
#    if tagUpsilonCount > 5: break #making sample of same size as Willem and Shray used to do a direct comparison
#    if nTot > 2: break

#print 'finalBreakCount is:', finalBreakCount
#print 'ZOINKSCount is:', ZOINKSCount


file_out.cd()
#taup_ntuple.Write()
#taum_ntuple.Write()
#pEff_reco_pi.Write()
#pEff_lostTrack.Write()
ntuple.Write()
file_out.Close()
