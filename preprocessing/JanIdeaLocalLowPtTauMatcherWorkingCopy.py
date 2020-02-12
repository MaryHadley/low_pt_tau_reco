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
from frameChangingUtilities import *

#Had P.L. look over as second pair of eyes, comments below passed his muster. MHH 11 Nov. 2019
#parser = ArgumentParser()
#parser.add_argument('--suffix', default='_', help='Suffix to be added to the end of out the output file name, such as cartesian_upsilon_taus<_suffix>, where the _suffix would be some useful suffix')

#A note about my naming, which could be better: "local_ blah blah" refers to the vector after the rotation to the VisTauMom points along z frame but before we have set the highest pT pi in that frame to point along minus x. I will call the final vectors after the second rotation toUse_local blah blah. Sorry (to myself I suppose, or anyone who might ultimately read this) for this weirdness.

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


from FWCore.ParameterSet.VarParsing import VarParsing
options = VarParsing ('python')
options.register('suffix',
                        '',
                        VarParsing.multiplicity.singleton,
                        VarParsing.varType.string,
                        'suffix to append to out file name')


options.parseArguments()
print options

handlePruned  = Handle ("std::vector<reco::GenParticle>")
labelPruned = ("prunedGenParticles")

handleReco = Handle ("std::vector<pat::PackedCandidate>")
recoLabel = ("packedPFCandidates")

lostLabel = ("lostTracks")

handleMET = Handle ("std::vector<pat::MET>")
labelMET = ("slimmedMETs")



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


branches = taum_branches + taup_branches
branches.append('upsilon_m')
branches.append('neutrino_pt')
branches.append('neutrino_phi')
branches.append('neutrino_eta')
branches.append('antineutrino_pt')
branches.append('antineutrino_phi')
branches.append('antineutrino_eta')

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

branches.append("toUse_local_pi_p_lv1_pt")
branches.append("toUse_local_pi_p_lv2_pt")
branches.append("toUse_local_pi_p_lv3_pt")
branches.append("toUse_local_antineu_lv_pt")

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

branches.append("naive_upsilon_lv_mass")
branches.append("global_naive_upsilon_lv_mass")

branches.append("check_upsilon_mass")

branches.append("tau_true_mom_mag")
branches.append("naive_tau_mom_mag")
branches.append("antitau_true_mom_mag")
branches.append("naive_antitau_mom_mag")

branches.append("diff_true_minus_naive_antitau_mom_mag")
branches.append("diff_true_minus_naive_tau_mom_mag")

#Jan idea 
branches.append("vis_ditau_px")
branches.append("vis_ditau_py")
branches.append("vis_ditau_pz")

branches.append("true_ditau_px")
branches.append("true_ditau_py")
branches.append("true_ditau_pz")

branches.append("SFx")
branches.append("SFy")
branches.append("SFz")



suffix = options.suffix
print 'suffix is:', suffix
file_out = ROOT.TFile('cartesian_upsilon_taus_%s.root'%(suffix), 'recreate')
file_out.cd()

taup_ntuple = ROOT.TNtuple('tree', 'tree', ':'.join(taum_branches))


taum_ntuple = ROOT.TNtuple('tree', 'tree', ':'.join(taup_branches))


ntuple = ROOT.TNtuple('tree', 'tree', ':'.join(branches))
#print 'ntuple is:', ntuple

verb = False

upsilon_id = 553
tau_id = 15
pion_id = 211
tau_neu_id = 16
neutral_pion_id = 111
photon_id = 22

#piPtCut = 0.35
piPtCut = 0
#piPtCut = 0.7

# Events takes either
# - single file name
# - list of file names
# - VarParsing options

# use Varparsing object
events = Events (options)
nTot = 0
tagUpsilonCount = 0


for event in events:
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
    gen_antineu = []
    gen_pionn = []
    gen_photons = []


    matched_pionp = []
    matched_pionm = []
    matched_photonp = []
    matched_photonm = []

    lost_pions = []
    taum_has_pionn = False
    taup_has_pionn = False

    # Filter reco particles
    rec_pionm = []
    rec_pionp = []
    rec_pions = []
    rec_photons = []



    
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
    for pp in reco_Particles:
        if abs(pp.pdgId()) == pion_id:
            rec_pions.append(pp)
        elif abs(pp.pdgId()) == photon_id:
            rec_photons.append(pp)

    for pp in lost_Particles:
        if abs(pp.pdgId()) == pion_id:
            lost_pions.append(pp)

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

    # comb upsilon particles, see if decayed into t-t+. gen_upsilon loop starts next line, one inside overall event loop
    for p in gen_upsilon: #Gen Upsilon loop starts here, this is inside the overall event loop
        found_taup = False
        found_taum = False
        #  normal tau (negative, should have 2- pions and 1+ pion)
        # comb list of antitau generated, gen_taup loop starts here, this is one inside gen_upsilon loop
        for pa in gen_taup: # gen_taup loop starts here, this is one inside the gen_upsilon loop
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
                    print "found something before matching! a tau!"
                    break #if we have found a good tau, leave the loop, otherwise keep iterating through and check the next tau in the gen_taup list

        #  anti tau (positive, should have 1- pion and 2+ pions) gen_taum loop starts here, this is also one inside gen_upsilon loop
        for pa in gen_taum:
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
                    break  # same deal, if we have found a good tau, great, leave the loop. otherwise, move on and check the next tau in the gen_taum list
                    
        #this if statement is one in from the gen_upsilon loop, just like the gen_taup and gen_taum loops are. this loop only entered if the conditions below are satisfied, otherwise skipped and the code goes to check the next gen_upsilon
        if found_taum and found_taup and len(g_taum_pions) == 3 and len(g_taup_pions) == 3 and found_neu and found_anti_neu:
            #print "This event has ", len(g_taup_photons) / 2, " taup pionn"
            #print "This event has ", len(g_taum_photons) / 2, " taum pionn"
            print "Found something before matching!"
            leps_mydecay.append(g_taum)
            leps_mydecay.append(g_taup)
            leps_mydecay += g_taum_pions
            leps_mydecay += g_taup_pions
            leps_mydecay += g_taum_pionn
            leps_mydecay += g_taup_pionn
            leps_mydecay.append(gen_neu)
            leps_mydecay.append(gen_antineu)
            tag_upsilon = True #at this point, we have declared tag_upsilon true, aka I have a good upsilon...but hang on, further checks coming
            taup_has_pionn = len(g_taup_pionn) != 0
            taum_has_pionn = len(g_taum_pionn) != 0

            # tau pions
            # If we think we have a good upsilon, check all the genpi in the g_taum_pions associated with the good upsilon and find the best match with a rec pion in the event
            ##this loop is one in from the loop above, the loop we enter if we think we have a good upsilon, and is two in from the general overall upsilon loop
            for genpi in g_taum_pions:
                min_ind = None
                min_deltaR = 9999
                matched_x = False
                for i in range(0, len(
                        rec_pions)):  # check if particles correspond to one another based on delta r, charge, and delta pt
                    recpi = rec_pions[i]
                    gen_lv = TLorentzVector()
                    gen_lv.SetPtEtaPhiM(genpi.pt(), genpi.eta(), genpi.phi(), 0.139)
                    rec_lv = TLorentzVector()
                    rec_lv.SetPtEtaPhiM(recpi.pt(), recpi.eta(), recpi.phi(), 0.139)
                    deltaR = gen_lv.DeltaR(rec_lv)
                    deltaPT = (rec_lv.Pt() - gen_lv.Pt()) / gen_lv.Pt()
                    
                    if recpi.pdgId() == genpi.pdgId() and abs(deltaR) < 0.1 and abs(deltaPT) < 0.3 and deltaR < min_deltaR and abs(genpi.eta()) < 2.5 and not recpi in matched_pionm:
                        min_ind = i
                        matched_x = True
                        min_deltaR = deltaR
                if matched_x:
                    matched_pionm.append(rec_pions[min_ind])
             
             
             # if the taum_has associated gen neutral pions in its decay chain!! Not just hanging around in the event. match this gen neutral pion with its best reco pion...except oh wait, we do not have reco pions, we have reco photons, so we need to dig a little deeper
            
            if taum_has_pionn:
                for genph in g_taum_photons: #ok we know the taum has neutral ph in its decay chain, because we entered this loop.
                
                    min_ind = None
                    min_deltaR = 9999
                    matched_x = False
                    for i in range(0, len(rec_photons)):
                        recph = rec_photons[i]
                        gen_lv = TLorentzVector()
                        gen_lv.SetPtEtaPhiM(genph.pt(), genph.eta(), genph.phi(), 0)
                        rec_lv = TLorentzVector()
                        rec_lv.SetPtEtaPhiM(genph.pt(), genph.eta(), genph.phi(), 0)
                        deltaR = gen_lv.DeltaR(rec_lv)
                        deltaPT = (rec_lv.Pt() - gen_lv.Pt()) / gen_lv.Pt()
                        
                        if abs(deltaR) < 0.1 and abs(deltaPT) < 0.1 and deltaR < min_deltaR and abs(genph.eta()) < 2.5 and not recph in matched_photonm:
                            min_ind = i
                            matched_x = True
                            min_deltaR = deltaR
                    if matched_x:
                        matched_photonm.append(rec_photons[min_ind]) #ok, so we decide if we have matched pion by doing the gen ph to rec ph matching. This loop is at the same level as the genpi in g_taum_pions loop, so this is one in from the good upsilon loop
                    
                    
        #antitau pions
        # # If we think we have a good upsilon, check all the genpi in the g_taum_pions associated with the good upsilon and find the best match with a rec pion in the event
            ##this loop is one in from the loop we enter if we think we have a good upsilon, and is two in from the general overall upsilon loop
            for genpi in g_taup_pions:

                min_ind = None
                min_deltaR = 99999
                matched_x = False
                for i in range(0, len(
                        rec_pions)):  # check if particles correspond to one another based on delta r, charge, and delta pt
                    recpi = rec_pions[i]
                    gen_lv = TLorentzVector()
                    gen_lv.SetPtEtaPhiM(genpi.pt(), genpi.eta(), genpi.phi(), 0.139)
                    rec_lv = TLorentzVector()
                    rec_lv.SetPtEtaPhiM(recpi.pt(), recpi.eta(), recpi.phi(), 0.139)
                    deltaR = gen_lv.DeltaR(rec_lv)
                    deltaPT = (rec_lv.Pt() - gen_lv.Pt()) / gen_lv.Pt()
                    if recpi.pdgId() == genpi.pdgId() and abs(deltaR) < 0.1 and abs(deltaPT) < 0.3 and deltaR < min_deltaR and abs(genpi.eta()) < 2.5 and not recpi in matched_pionp:
                        min_ind = i
                        min_deltaR = deltaR
                        matched_x = True
                if matched_x:
                        matched_pionp.append(rec_pions[min_ind])

            if taup_has_pionn:
                for genph in g_taup_photons:
                    min_ind = None
                    min_deltaR = 9999
                    matched_x = False
                    for i in range(0, len(rec_photons)):
                        recph = rec_photons[i]
                        gen_lv = TLorentzVector()
                        gen_lv.SetPtEtaPhiM(genph.pt(), genph.eta(), genph.phi(), 0)
                        rec_lv = TLorentzVector()
                        rec_lv.SetPtEtaPhiM(genph.pt(), genph.eta(), genph.phi(), 0)
                        deltaR = gen_lv.DeltaR(rec_lv)
                        deltaPT = (rec_lv.Pt() - gen_lv.Pt()) / gen_lv.Pt()
                        if abs(deltaR) < 0.1 and abs(deltaPT) < 0.3 and deltaR < min_deltaR and abs(genph.eta()) < 2.5 and not recph in matched_photonp:
                            min_ind = i
                            matched_x = True
                            min_deltaR = deltaR
                    if matched_x:
                        matched_photonp.append(rec_photons[min_ind])#ok, so we decide if we have matched pion by doing the gen ph to rec ph matching. This loop is at the same level as the genpi in g_taup_pions loop, so this is one in from the good upsilon loop



            if len(gen_pionn) != 0:
                tag_upsilon = False
#                print 'len(gen_pionn) is:', len(gen_pionn)
                continue#QUESTION RUBBER DUCK: ok, I don't understand why this does not kill all the events, because they made a mistake here, they checked whether there are any neutral pions hanging around, not just in the decay chain, and of course there always are. This is again one in from good upsilon loop
            
            
            if antineu_etaCheck or neu_etaCheck:
                tag_upsilon = False # this check is useless and I will likely remove it. one in from gen upsilon loop this causes tag upsilon to flip if check fails
            
            
            if len(matched_pionp) == 3 and len(matched_photonp) % 2 == 0: #RUBBER DUCK
                tag_taup = True #ok this is one in from the gen upsilon loop. Also it is not clear to me how this cuts out 3 prong plus neutral pion decays...they are just checking that there are an even number of matched photons, which could be true for decays with neutral pions in the decay chain...Update I do NOT think this is doing what it is supposed to, aka I think it is NOT cutting out decays with neutral pions in the decay chain
            if len(matched_pionm) == 3 and len(matched_photonm) % 2 == 0:
                tag_taum = True #same comment....also note that in these they are setting the individual tag_taum, tag_taup tags, not the overall tag_upsilon
            
            if len(matched_pionp) + len(matched_pionm) != 6: #one in from gen upsilon loop, this causes the tag_upsilon to flip if the check fails
                tag_upsilon = False

        break # this break takes you out of the overall upsilon loop

    nTot += 1
#    if nTot > 100: break
#    if tagUpsilonCount > 0: break 
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
    
    vis_taum_lv = TLorentzVector()
    vis_taup_lv = TLorentzVector()
    
#    tmp_local_pi_m_lv1 = TLorentzVector()
#    tmp_local_pi_m_lv2 = TLorentzVector()
#    tmp_local_pi_m_lv3 = TLorentzVector()
    
    
    

    if tag_taum and tag_taup:
        tag_upsilon = True #one in from overall event loop #ok I think this is how the gen_pionn does not kill off everything...something that failed gen pion neutral could then pass the individual tag_taum, tag_taup tags around L547ish, and then get flipped to
    
    if tag_taum: #one in from overall event loop
        print("found Tau-")
        #taum_tofill = OrderedDict(zip(taum_branches, [-99.] * len(taum_branches)))
        
        pi_m_lv1.SetPtEtaPhiM(matched_pionm[0].pt(), matched_pionm[0].eta(), matched_pionm[0].phi(), 0.139)
        pi_m_lv2.SetPtEtaPhiM(matched_pionm[1].pt(), matched_pionm[1].eta(), matched_pionm[1].phi(), 0.139)
        pi_m_lv3.SetPtEtaPhiM(matched_pionm[2].pt(), matched_pionm[2].eta(), matched_pionm[2].phi(), 0.139)
        taum_lv = pi_m_lv1 + pi_m_lv2 + pi_m_lv3 + neu_lv
        
        tau_true_mom_mag = getMag(taum_lv)
        
        print "taum_lv.Px() before rotation is:", taum_lv.Px()
        print "taum_lv.Py() before rotation is:", taum_lv.Py()
        print "taum_lv.Pz() before rotation is:", taum_lv.Pz()
        print "taum_lv.M() before rotation is:", taum_lv.M()
        
        vis_taum_lv = pi_m_lv1 + pi_m_lv2 + pi_m_lv3
        
        global_naive_taum_lv =labFrameNaiveTauScaling(vis_taum_lv)
        print "DONKEY! global_naive_taum_lv Px Py Pz E ():", global_naive_taum_lv.Px(), global_naive_taum_lv.Py(), global_naive_taum_lv.Pz(), global_naive_taum_lv.E(), global_naive_taum_lv.M()
        
        naive_tau_mom_mag = getMag(global_naive_taum_lv)
        
        diff_true_minus_naive_tau_mom_mag = tau_true_mom_mag - naive_tau_mom_mag
        
        print "GORILLA! vis_taum_lv.Px() before rotation is:", vis_taum_lv.Px()
        print "GORILLA! vis_taum_lv.Py() before rotation is:", vis_taum_lv.Py()
        print "GORILLA! vis_taum_lv.Pz() before rotation is:", vis_taum_lv.Pz()
        print "GORILLA! vis_taum_lv.M() before rotation is:", vis_taum_lv.M()
        print "GORILLA! vis_taum_lv.E() before rotation is:", vis_taum_lv.E()
        
        orig_vis_taum_theta = vis_taum_lv.Theta() #this is the theta before any rotation has been done, we need to save this
#        print "orig_vis_taum_theta is:", orig_vis_taum_theta
        orig_vis_taum_phi   = vis_taum_lv.Phi() #this is the phi before any rotation has been done, we need to save this
        
        local_vis_taum_lv = rotateToVisTauMomPointsAlongZAxis(orig_vis_taum_theta, orig_vis_taum_phi, vis_taum_lv)
        print "after rotation stuff! local_vis_taum_lv.Px(), local_vis_taum_lv.Py(), local_vis_taum_lv.Pz(), local_vis_taum_lv.M(), local_vis_taum_lv.E():", local_vis_taum_lv.Px(), local_vis_taum_lv.Py(), local_vis_taum_lv.Pz(), local_vis_taum_lv.M(), local_vis_taum_lv.E()
        
        local_pi_m_lv1 = rotateToVisTauMomPointsAlongZAxis(orig_vis_taum_theta, orig_vis_taum_phi, pi_m_lv1)
        print "local_pi_m_lv1.M() is:", local_pi_m_lv1.M()
        local_pi_m_lv2 = rotateToVisTauMomPointsAlongZAxis(orig_vis_taum_theta, orig_vis_taum_phi, pi_m_lv2)
        local_pi_m_lv3 = rotateToVisTauMomPointsAlongZAxis(orig_vis_taum_theta, orig_vis_taum_phi, pi_m_lv3)
        local_neu_lv   = rotateToVisTauMomPointsAlongZAxis(orig_vis_taum_theta, orig_vis_taum_phi, neu_lv)
        if local_neu_lv.Pt() > 1.3:
            print "WARNING DEBUG"
            print "local_neu_lv.Pt() is:", local_neu_lv.Pt()
            print "local_neu_lv.Px() is:", local_neu_lv.Px()
            print "local_neu_lv.Py() is:", local_neu_lv.Py()
            print "local_neu_lv.Pz() is:", local_neu_lv.Pz()
            print "local_neu_lv.E() is:", local_neu_lv.E()
            print "local_neu_lv.M() is:", local_neu_lv.M()
            print "neu_lv Px Py Pz E  M  Pt is:", neu_lv.Px(), neu_lv.Py(), neu_lv.Pz(), neu_lv.E(), neu_lv.M(), neu_lv.Pt()
            print "orig_vis_taum_theta, orig_vis_taum_phi", orig_vis_taum_theta, orig_vis_taum_phi
            
            print "pi_m_lv1 Px Py Pz E M pt:", pi_m_lv1.Px(), pi_m_lv1.Py(), pi_m_lv1.Pz(), pi_m_lv1.E(), pi_m_lv1.M(), pi_m_lv1.Pt()
            print "local_pi_m_lv1 Px Py Pz E M pt:", local_pi_m_lv1.Px(), local_pi_m_lv1.Py(), local_pi_m_lv1.Pz(), local_pi_m_lv1.E(), local_pi_m_lv1.M(), local_pi_m_lv1.Pt()
            
            print "pi_m_lv2 Px Py Pz E M pt:", pi_m_lv2.Px(), pi_m_lv2.Py(), pi_m_lv2.Pz(), pi_m_lv2.E(), pi_m_lv2.M(), pi_m_lv2.Pt()
            print "local pi m lv2 Px Py Pz E M Pt:", local_pi_m_lv2.Px(), local_pi_m_lv2.Py(), local_pi_m_lv2.Pz(), local_pi_m_lv2.E(), local_pi_m_lv2.M(), local_pi_m_lv2.Pt()
            
            print "pi m lv3 Px Py Pz E M Pt:", pi_m_lv3.Px(), pi_m_lv3.Py(), pi_m_lv3.Pz(), pi_m_lv3.E(), pi_m_lv3.M(), pi_m_lv3.Pt()
            print "local pi m lv3 Px Py Pz E M Pt:", local_pi_m_lv3.Px(), local_pi_m_lv3.Py(), local_pi_m_lv3.Pz(), local_pi_m_lv3.E(), local_pi_m_lv3.M(), local_pi_m_lv3.Pt()
            print "end End debug WARNING BLOCK"
#        print "Hungarian Horntail! local_pi_m_lv1 is:",  local_pi_m_lv1
        local_unsortedPiPtList_m = [local_pi_m_lv1.Pt(), local_pi_m_lv2.Pt(), local_pi_m_lv3.Pt()]
        local_unsortedPi4VecList_m = [local_pi_m_lv1, local_pi_m_lv2, local_pi_m_lv3]
        print "local_unsortedPiPtList_m is:", local_unsortedPiPtList_m
        print "local_unsortedPi4VecList_m is:", local_unsortedPi4VecList_m
#        local_sortedPiPtList = sorted(local_unsortedPiPtList, reverse=True)
#        print "local_sortedPiPtList is:", local_sortedPiPtList
        
        # idea of how to do this from: https://stackoverflow.com/questions/6422700/how-to-get-indices-of-a-sorted-array-in-python
        local_sortedPiPtOriginalIndexList_m =      [i[0] for i in sorted(enumerate(local_unsortedPiPtList_m), reverse=True, key=lambda x:x[1])]
        print "local_sortedPiPtOriginalIndexList_m:", local_sortedPiPtOriginalIndexList_m
 #       print "new local_pi_m_lv1 pt is:", local_pi_m_lv1.Pt()
 
        print local_sortedPiPtOriginalIndexList_m[0] #index of the element of the vector with the biggest pT
        print local_sortedPiPtOriginalIndexList_m[1] #index of the element of the vector with the second biggest pT
        print local_sortedPiPtOriginalIndexList_m[2] #index of the element of the vector with the smallest pT
 
        local_pi_m_lv1 = local_unsortedPi4VecList_m[local_sortedPiPtOriginalIndexList_m[0]] #make the pi_m_lv1 the vector that has the biggest pT in the new frame
        local_pi_m_lv2 = local_unsortedPi4VecList_m[local_sortedPiPtOriginalIndexList_m[1]] #make the pi_m_lv2 the vector that has the second biggest pT in the new frame
        local_pi_m_lv3 = local_unsortedPi4VecList_m[local_sortedPiPtOriginalIndexList_m[2]] #make the pi_m_lv3 the vector that has the smallest pT in the new frame 
        
        print "new local_pi_m_lv1.Pt() is:", local_pi_m_lv1.Pt()
        print "new local_pi_m_lv2.Pt() is:", local_pi_m_lv2.Pt()
        print "new local_pi_m_lv3.Pt() is:", local_pi_m_lv3.Pt()
        local_taum_lv = local_pi_m_lv1 + local_pi_m_lv2 + local_pi_m_lv3 + local_neu_lv
        print "Treacle Tart! local_taum_lv.M() is:", local_taum_lv.M()
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
        print "toUse_local_pi_m_lv1.Px(), toUse_local_pi_m_lv1.Py() is:", toUse_local_pi_m_lv1.Px(), toUse_local_pi_m_lv1.Py()
        #print "toUse_local_pi_m_lv1.Phi() is:", toUse_local_pi_m_lv1.Phi()
        toUse_local_pi_m_lv1_phi = toUse_local_pi_m_lv1.Phi()
        toUse_local_pi_m_lv2 = rotateToLeadPtPiInVisTauMomPointsAlongZFramePointsAlongNegX(initial_leadPt_pi_m_in_AllInZFrame_phi,local_pi_m_lv2)
        toUse_local_pi_m_lv3 = rotateToLeadPtPiInVisTauMomPointsAlongZFramePointsAlongNegX(initial_leadPt_pi_m_in_AllInZFrame_phi,local_pi_m_lv3)
        toUse_local_neu_lv =  rotateToLeadPtPiInVisTauMomPointsAlongZFramePointsAlongNegX(initial_leadPt_pi_m_in_AllInZFrame_phi,local_neu_lv)
        
        toUse_local_pi_m_lv1_pt =  toUse_local_pi_m_lv1.Pt()
        toUse_local_pi_m_lv2_pt =  toUse_local_pi_m_lv2.Pt()
        toUse_local_pi_m_lv3_pt =  toUse_local_pi_m_lv3.Pt()
        toUse_local_neu_lv_pt =    toUse_local_neu_lv.Pt()
        
        
            
        
        toUse_local_pi_m_lv1_theta = toUse_local_pi_m_lv1.Theta()
        toUse_local_pi_m_lv2_theta = toUse_local_pi_m_lv2.Theta()
        toUse_local_pi_m_lv3_theta = toUse_local_pi_m_lv3.Theta()
        toUse_local_neu_lv_theta = toUse_local_neu_lv.Theta()
        
#        print "toUse_local_pi_m_lv2 phi before is:", toUse_local_pi_m_lv2.Phi()
        toUse_local_pi_m_lv2_phi = get_toUse_local_phi(toUse_local_pi_m_lv2)
#        print "toUse_local_pi_m_lv2_phi after is:", toUse_local_pi_m_lv2_phi
        toUse_local_pi_m_lv2.SetPhi(toUse_local_pi_m_lv2_phi)
#        print "toUse_local_pi_m_lv2.Phi()", toUse_local_pi_m_lv2.Phi()
#        print "toUse_local_pi_m_lv2.Pt() after", toUse_local_pi_m_lv2.Pt()

        toUse_local_pi_m_lv3_phi = get_toUse_local_phi(toUse_local_pi_m_lv3)
        toUse_local_pi_m_lv3.SetPhi(toUse_local_pi_m_lv3_phi)
        
        toUse_local_neu_lv_phi = toUse_local_neu_lv.Phi() # do not apply the get_toUse_local_phi function here because we do NOT know that the nu phi should be with [-pi/2, pi/2]
        
        ##new stuff for test 21 Jan. 2020 ###
        
        toUse_local_vis_taum_lv = toUse_local_pi_m_lv1 + toUse_local_pi_m_lv2 + toUse_local_pi_m_lv3
        
        print "GORILLA! toUse_local_vis_taum_lv Px Py Pz E M is:", toUse_local_vis_taum_lv.Px(), toUse_local_vis_taum_lv.Py(), toUse_local_vis_taum_lv.Pz(), toUse_local_vis_taum_lv.E(), toUse_local_vis_taum_lv.M()
        
        toUse_local_naive_taum_lv = naiveTauScaling(toUse_local_vis_taum_lv)
        print "toUse_local_naive_taum_lv Px Py Pz E M is:", toUse_local_naive_taum_lv.Px(), toUse_local_naive_taum_lv.Py(), toUse_local_naive_taum_lv.Pz(), toUse_local_naive_taum_lv.E(), toUse_local_naive_taum_lv.M()
        
        almostLabFrame_naive_taum_lv = unrotateFromLeadPtPiInVisTauMomPointsAlongZFramePointsAlongNegX(initial_leadPt_pi_m_in_AllInZFrame_phi,toUse_local_naive_taum_lv)
        
        labFrame_naive_taum_lv = unrotateFromVisTauMomPointsAlongZAxis(orig_vis_taum_theta,orig_vis_taum_phi, almostLabFrame_naive_taum_lv)
        
        print "GORILLA! labFrame_naive_taum_lv Px Py Pz E M:", labFrame_naive_taum_lv.Px(), labFrame_naive_taum_lv.Py(), labFrame_naive_taum_lv.Pz(), labFrame_naive_taum_lv.E(), labFrame_naive_taum_lv.M()
        
#        should_be_wrong =  labFrame_naive_taum_lv = unrotateFromVisTauMomPointsAlongZAxis(orig_vis_taum_phi,orig_vis_taum_theta, almostLabFrame_naive_taum_lv)
#        print "should_be_wrong Px Py Pz E M:", should_be_wrong.Px(), should_be_wrong.Py(), should_be_wrong.Pz(), should_be_wrong.E(), should_be_wrong.M()
        ######
        
        toUse_local_taum_lv =  toUse_local_pi_m_lv1 + toUse_local_pi_m_lv2 + toUse_local_pi_m_lv3 + toUse_local_neu_lv
        toUse_local_taum_lv_mass = toUse_local_taum_lv.M()
        check1 =  unrotateFromLeadPtPiInVisTauMomPointsAlongZFramePointsAlongNegX(initial_leadPt_pi_m_in_AllInZFrame_phi,toUse_local_taum_lv)
        check1_mass = check1.M()
        check2 = unrotateFromVisTauMomPointsAlongZAxis(orig_vis_taum_theta,orig_vis_taum_phi, check1)
        check2_mass = check2.M()
        # dog = vis_taum_lv
#         print "dog.Pt()", dog.Pt() 
#         #verified that doing it the long way, e.g. vis_taum_lv.Eta() etc, gives the same results as doing vis_taum_lv.Vect() and then accessing its components
#         vis_taum_eta = vis_taum_lv.Eta() 
#         vis_taum_pt = vis_taum_lv.Pt()
#         vis_taum_phi = vis_taum_lv.Phi()
#         vis_taum_theta = vis_taum_lv.Theta()
#         print "vis_taum_lv 3 vector stuff( pt, eta, theta, phi, X, Y,Z, Px, Py, Pz):", vis_taum_pt, vis_taum_eta, vis_taum_theta, vis_taum_phi, vis_taum_lv.X(), vis_taum_lv.Y(), vis_taum_lv.Z(), vis_taum_lv.Px(), vis_taum_lv.Py(), vis_taum_lv.Pz()
#         
#         vis_taum_3vec = vis_taum_lv.Vect()
#         print "vis_taum_3vector stuff (pt, eta, theta, phi, X,Y,Z, mag of vector aka sqrt(x^2 + y^2 + z^2), Px, Py, Pz):",  vis_taum_3vec.Pt(), vis_taum_3vec.Eta(), vis_taum_3vec.Theta(),  vis_taum_3vec.Phi(), vis_taum_3vec.X(), vis_taum_3vec.Y(), vis_taum_3vec.Z(), vis_taum_3vec.Mag(), vis_taum_3vec.Px(), vis_taum_3vec.Py(), vis_taum_3vec.Pz()
#  
#         pi_m_1_3vec = pi_m_lv1.Vect()
#         orig_pi_m_1_eta = pi_m_lv1.Eta()
#         orig_pi_m_1_phi = pi_m_lv1.Phi()
#         pi_m_1_3vec_mag = pi_m_1_3vec.Mag()
#         
#         print "original pi 1 eta:",  orig_pi_m_1_eta
#         
#         
#         pi_m_2_3vec = pi_m_lv2.Vect()
#         orig_pi_m_2_eta = pi_m_lv2.Eta()
#         
#         pi_m_3_3vec = pi_m_lv3.Vect()
#         orig_pi_m_3_eta = pi_m_lv3.Eta()
#         
#         neu_3vec = neu_lv.Vect()
#         
#         pi_m_1_comp_perp_to_vis_taum = pi_m_1_3vec.Perp(vis_taum_3vec)
#         pi_m_2_comp_perp_to_vis_taum = pi_m_2_3vec.Perp(vis_taum_3vec)
#         pi_m_3_comp_perp_to_vis_taum = pi_m_3_3vec.Perp(vis_taum_3vec)
#         neu_comp_perp_to_vis_taum    = neu_3vec.Perp(vis_taum_3vec)
#         
#         # test
#         print "sanity check 1"
#         pi_m_1_comp_perp_to_itself =  pi_m_1_3vec.Perp(pi_m_1_3vec)
#         print " pi_m_1_comp_perp_to_itself is:",  pi_m_1_comp_perp_to_itself
# #         
#         print "sanity check 2"
#         pi_m_1_comp_perp_to_normal_z_axis = pi_m_1_3vec.Perp()
#         print "pi_m_1_comp_perp_to_normal_z_axis is:", pi_m_1_comp_perp_to_normal_z_axis
# #         
#         print "pi_m_1_3vec.Pt() is:", pi_m_1_3vec.Pt()
#         print "pi_m_1_3vec_mag/(cosh(orig_pi_m_1_eta)) is:", pi_m_1_3vec_mag/(math.cosh(orig_pi_m_1_eta))
#         
#         delta_orig_pi_m_1_eta_vis_taum_eta = orig_pi_m_1_eta - vis_taum_eta
#         # delta = pi_m_lv1.Eta() - vis_taum_lv.Eta()
# #         print "delta is:", delta
#         
#        # print " pi_m_1_comp_perp_to_vis_taum is:",  pi_m_1_comp_perp_to_vis_taum
#        
#         print "delta_orig_pi_m_1_eta_vis_taum_eta", delta_orig_pi_m_1_eta_vis_taum_eta
#         tmp_local_pi_m_lv1.SetPtEtaPhiM(pi_m_1_comp_perp_to_vis_taum, delta_orig_pi_m_1_eta_vis_taum_eta, orig_pi_m_1_phi, 0.139)
#         print "tmp_local_pi_m_lv1 stuff:", tmp_local_pi_m_lv1.Pt(), tmp_local_pi_m_lv1.Eta(), tmp_local_pi_m_lv1.Phi()
#         print " pi_m_1_3vec_mag:", pi_m_1_3vec_mag
#         
#         print " pi_m_1_comp_perp_to_vis_taum is:",  pi_m_1_comp_perp_to_vis_taum
#         print "pi_m_1_3vec_mag/cosh (delta_orig_pi_m_1_eta_vis_taum_eta) is:", pi_m_1_3vec_mag/(math.cosh(delta_orig_pi_m_1_eta_vis_taum_eta))
        
#         """
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

    if tag_taup: #one in from overall event loop 
        print("Found tau+")
        #taup_tofill = OrderedDict(zip(taup_branches, [-99.] * len(taup_branches)))
        pi_p_lv1.SetPtEtaPhiM(matched_pionp[0].pt(), matched_pionp[0].eta(), matched_pionp[0].phi(), 0.139)
        pi_p_lv2.SetPtEtaPhiM(matched_pionp[1].pt(), matched_pionp[1].eta(), matched_pionp[1].phi(), 0.139)
        pi_p_lv3.SetPtEtaPhiM(matched_pionp[2].pt(), matched_pionp[2].eta(), matched_pionp[2].phi(), 0.139)
        taup_lv = pi_p_lv1 + pi_p_lv2 + pi_p_lv3 + antineu_lv
        
        antitau_true_mom_mag = getMag(taup_lv)
        
        vis_taup_lv = pi_p_lv1 + pi_p_lv2 + pi_p_lv3
        print "vis_taup_lv.Px() before rot is:", vis_taup_lv.Px()
        print "vis_taup_lv.Py() before rot is:", vis_taup_lv.Py()
        print "vis_taup_lv.Pz() before rot is:", vis_taup_lv.Pz()
        print "vis_taup_lv.M before rot is:", vis_taup_lv.M()
        
        global_naive_taup_lv =labFrameNaiveTauScaling(vis_taup_lv)
        naive_antitau_mom_mag = getMag(global_naive_taup_lv)
        
        diff_true_minus_naive_antitau_mom_mag =  antitau_true_mom_mag - naive_antitau_mom_mag
        
        print "POODLE! global_naive_taup_lv stuff  is:", global_naive_taup_lv.Px(), global_naive_taup_lv.Py(), global_naive_taup_lv.Pz(), global_naive_taup_lv.E(),global_naive_taup_lv.M()
        
        orig_vis_taup_theta = vis_taup_lv.Theta()
        orig_vis_taup_phi   = vis_taup_lv.Phi()
        
        local_vis_taup_lv =  rotateToVisTauMomPointsAlongZAxis(orig_vis_taup_theta, orig_vis_taup_phi, vis_taup_lv)
        print "after rotation stuff for local_vis_taup_lv Px Py Pz M is:", local_vis_taup_lv.Px(), local_vis_taup_lv.Py(), local_vis_taup_lv.Pz(), local_vis_taup_lv.M()
        
        local_pi_p_lv1 = rotateToVisTauMomPointsAlongZAxis(orig_vis_taup_theta, orig_vis_taup_phi,pi_p_lv1)
        local_pi_p_lv2 = rotateToVisTauMomPointsAlongZAxis(orig_vis_taup_theta, orig_vis_taup_phi, pi_p_lv2)
        local_pi_p_lv3 = rotateToVisTauMomPointsAlongZAxis(orig_vis_taup_theta, orig_vis_taup_phi, pi_p_lv3)
        local_antineu_lv = rotateToVisTauMomPointsAlongZAxis(orig_vis_taup_theta, orig_vis_taup_phi, antineu_lv)
        
        local_unsortedPiPtList_p = [local_pi_p_lv1.Pt(), local_pi_p_lv2.Pt(), local_pi_p_lv3.Pt()]
        local_unsortedPi4VecList_p = [local_pi_p_lv1, local_pi_p_lv2, local_pi_p_lv3]
        print "loal_unsortedPiPtList_p is:", local_unsortedPiPtList_p
        
        # idea of how to do this from: https://stackoverflow.com/questions/6422700/how-to-get-indices-of-a-sorted-array-in-python
        local_sortedPiPtOriginalIndexList_p =      [i[0] for i in sorted(enumerate(local_unsortedPiPtList_p), reverse=True, key=lambda x:x[1])]
        print "local_sortedPiPtOriginalIndexList_p:", local_sortedPiPtOriginalIndexList_p
        
        print local_sortedPiPtOriginalIndexList_p[0] #index of the element of the vector with the biggest pT
        print local_sortedPiPtOriginalIndexList_p[1] #index of the element of the vector with the second biggest pT
        print local_sortedPiPtOriginalIndexList_p[2] #index of the element of the vector with the smallest pT
        
        local_pi_p_lv1 = local_unsortedPi4VecList_p[local_sortedPiPtOriginalIndexList_p[0]] #make the pi_m_lv1 the vector that has the biggest pT in the new frame
        local_pi_p_lv2 = local_unsortedPi4VecList_p[local_sortedPiPtOriginalIndexList_p[1]] #make the pi_m_lv2 the vector that has the second biggest pT in the new frame
        local_pi_p_lv3 = local_unsortedPi4VecList_p[local_sortedPiPtOriginalIndexList_p[2]] #make the pi_m_lv3 the vector that has the smallest pT in the new frame
        
        print "new local_pi_p_lv1.Pt() is:", local_pi_p_lv1.Pt()
        print "new local_pi_p_lv2.Pt() is:", local_pi_p_lv2.Pt()
        print "new local_pi_p_lv3.Pt() is:", local_pi_p_lv3.Pt()
        
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
        
        toUse_local_pi_p_lv1_theta = toUse_local_pi_p_lv1.Theta()
        toUse_local_pi_p_lv2_theta = toUse_local_pi_p_lv2.Theta()
        toUse_local_pi_p_lv3_theta = toUse_local_pi_p_lv3.Theta()
        toUse_local_antineu_lv_theta = toUse_local_antineu_lv.Theta()
        
        toUse_local_pi_p_lv2_phi = get_toUse_local_phi(toUse_local_pi_p_lv2)
        toUse_local_pi_p_lv2.SetPhi(toUse_local_pi_p_lv2_phi)
        
        toUse_local_pi_p_lv3_phi = get_toUse_local_phi(toUse_local_pi_p_lv3)
        toUse_local_pi_p_lv3.SetPhi(toUse_local_pi_p_lv3_phi)
        
        toUse_local_antineu_lv_phi = toUse_local_antineu_lv.Phi() # do not apply the get_toUse_local_phi function here because we do NOT know that the antinu phi should be with [-pi/2, pi/2]
        
        ### 21 Jan. 2020 test ####
        
        toUse_local_vis_taup_lv = toUse_local_pi_p_lv1 + toUse_local_pi_p_lv2 + toUse_local_pi_p_lv3
        toUse_local_naive_taup_lv = naiveTauScaling(toUse_local_vis_taup_lv)
        almostLabFrame_naive_taup_lv = unrotateFromLeadPtPiInVisTauMomPointsAlongZFramePointsAlongNegX(initial_leadPt_pi_p_in_AllInZFrame_phi, toUse_local_naive_taup_lv)
        labFrame_naive_taup_lv = unrotateFromVisTauMomPointsAlongZAxis(orig_vis_taup_theta, orig_vis_taup_phi, almostLabFrame_naive_taup_lv)
        print "POODLE! labFrame_naive_taup_lv Px Py Pz E M:", labFrame_naive_taup_lv.Px(), labFrame_naive_taup_lv.Py(), labFrame_naive_taup_lv.Pz(), labFrame_naive_taup_lv.E(), labFrame_naive_taup_lv.M()
        ###
        
        toUse_local_taup_lv = toUse_local_pi_p_lv1 + toUse_local_pi_p_lv2 + toUse_local_pi_p_lv3 + toUse_local_antineu_lv
       
        toUse_local_taup_lv_mass = toUse_local_taup_lv.M()
        
        check3 =  unrotateFromLeadPtPiInVisTauMomPointsAlongZFramePointsAlongNegX(initial_leadPt_pi_p_in_AllInZFrame_phi,toUse_local_taup_lv)
        check3_mass = check3.M()
        check4 = unrotateFromVisTauMomPointsAlongZAxis(orig_vis_taup_theta,orig_vis_taup_phi, check3)
        check4_mass = check4.M()
        
        
#         """
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
    

    if tag_upsilon: # one in from overall event loop #these matched_pionblah are the unsorted and unrotated at all values...do we just want to have the cuts be done in this way to save time...I think it vaguely makes sense, check with Greg to get his opinion. Also this is nominal mass sample so maybe we want the cut to be 0.35
        tagUpsilonCount += 1
        if matched_pionm[0].pt() < piPtCut or matched_pionm[1].pt() < piPtCut or matched_pionm[2].pt() < piPtCut or matched_pionp[0].pt() < piPtCut or matched_pionp[1].pt() < piPtCut or matched_pionp[2].pt() < piPtCut:
            print "one of the candidate pions failed the pT cut!"
            print "matched_pionm[0].pt() is:", matched_pionm[0].pt()
            print "matched_pionm[1].pt() is:", matched_pionm[1].pt()
            print "matched_pionm[2].pt() is:", matched_pionm[2].pt()
            print "matched_pionp[0].pt() is:", matched_pionp[0].pt()
            print "matched_pionp[1].pt() is:", matched_pionp[1].pt()
            print "matched_pionp[2].pt() is:", matched_pionp[2].pt()
            continue 
        vis_ditau_lv = vis_taum_lv + vis_taup_lv
        
        vis_ditau_px = vis_ditau_lv.Px()
        vis_ditau_py = vis_ditau_lv.Py()
        vis_ditau_pz = vis_ditau_lv.Pz()
        
        if vis_ditau_px == 0 or vis_ditau_py == 0 or vis_ditau_pz ==0:
            print "one of the vis_ditau_pi components was 0!"
            print "vis_ditau_px:", vis_ditau_px
            print "vis_ditau_py:", vis_ditau_py
            print "vis_ditau_pz:", vis_ditau_pz
            continue
        
        print 'Found Upsilon -> tau+ tau- -> pi+*3 pi-*3'
        #fill stuff, note we fill with reco info
        tofill = OrderedDict(zip(branches, [-99.] * len(branches)))
        print "tofill is:", tofill
        
        
       
        
        print "HUNGARIAN HORNTAIL! labFrame_naive_taum_lv stuff", labFrame_naive_taum_lv.Px(), labFrame_naive_taum_lv.Py(), labFrame_naive_taum_lv.Pz(), labFrame_naive_taum_lv.E(), labFrame_naive_taum_lv.M()
        print "HUNGARIAN HORNTAIL labFrame_naive_taup_lv stuff", labFrame_naive_taup_lv.Px(), labFrame_naive_taup_lv.Py(), labFrame_naive_taup_lv.Pz(),  labFrame_naive_taup_lv.E(), labFrame_naive_taup_lv.M()
        naive_upsilon_lv = labFrame_naive_taum_lv + labFrame_naive_taup_lv
        naive_upsilon_lv_mass = naive_upsilon_lv.M()
        print "POODLE! naive_upsilon lv Px Py Pz E M is:",  naive_upsilon_lv.Px(),  naive_upsilon_lv.Py(),  naive_upsilon_lv.Pz(),  naive_upsilon_lv.E(),  naive_upsilon_lv.M()
        
        
        print "HUNGARIAN HORNTAIL! global_naive_taum_lv Px Py Pz E M",  global_naive_taum_lv.Px(),  global_naive_taum_lv.Py(),  global_naive_taum_lv.Pz(),  global_naive_taum_lv.E(),  global_naive_taum_lv.M()
        print "HUNGARIAN HORNTAIL! global_naive_taup_lv Px Py Pz E M", global_naive_taup_lv.Px(), global_naive_taup_lv.Py(), global_naive_taup_lv.Pz(), global_naive_taup_lv.E(), global_naive_taup_lv.M()
        global_naive_upsilon_lv = global_naive_taum_lv + global_naive_taup_lv
        print "HUNGARIAN HORNTAIL! global_naive_upsilon_lv stuff:", global_naive_upsilon_lv.Px(), global_naive_upsilon_lv.Py(), global_naive_upsilon_lv.Pz(), global_naive_upsilon_lv.E(), global_naive_upsilon_lv.M()
        global_naive_upsilon_lv_mass = global_naive_upsilon_lv.M()
        upsilon_lv = neu_lv + antineu_lv + pi_m_lv1 + pi_m_lv2 + pi_m_lv3 + pi_p_lv1 + pi_p_lv2 + pi_p_lv3
        
        true_ditau_px = upsilon_lv.Px()
        true_ditau_py = upsilon_lv.Py()
        true_ditau_pz = upsilon_lv.Pz()
        
        SFx = true_ditau_px/vis_ditau_px
        
        if SFx > 10: 
            print "ELEPHANT!"
            print "true_ditau_px:", true_ditau_px
            print "vis_ditau_px:", vis_ditau_px
            print "SFx:", SFx
            
        SFy = true_ditau_py/vis_ditau_py
        
        if SFy > 10:
            print "ELEPHANT!"
            print "true_ditau_py", true_ditau_py
            print "vis_ditau_py", vis_ditau_py
            print "SFy:", SFy
            
        SFz = true_ditau_pz/vis_ditau_pz
        
        if SFz > 10:
            print "ELEPHANT!"
            print "true_ditau_pz:", true_ditau_pz
            print "vis_ditau_pz", vis_ditau_pz
            print "SFz:", SFz
        
        check_upsilon_lv = check2 + check4
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
        tofill['pi_minus1_pt'] = matched_pionm[0].pt()
        tofill['pi_minus1_eta'] = matched_pionm[0].eta()
        tofill['pi_minus1_phi'] = matched_pionm[0].phi()
        tofill['pi_minus1_m'] = pi_m_lv1.M()
        tofill['pi_minus2_pt'] = matched_pionm[1].pt()
        tofill['pi_minus2_eta'] = matched_pionm[1].eta()
        tofill['pi_minus2_phi'] = matched_pionm[1].phi()
        
        tofill['pi_minus2_m'] = pi_m_lv2.M()
        tofill['pi_minus3_pt'] = matched_pionm[2].pt()
        tofill['pi_minus3_eta'] = matched_pionm[2].eta()
        tofill['pi_minus3_phi'] = matched_pionm[2].phi()
        tofill['pi_minus3_m'] = pi_m_lv3.M()
        tofill['pi_plus1_pt'] = matched_pionp[0].pt()
        tofill['pi_plus1_eta'] = matched_pionp[0].eta()
        tofill['pi_plus1_phi'] = matched_pionp[0].phi()
        
        tofill['pi_plus1_m'] = pi_p_lv1.M()
        tofill['pi_plus2_pt'] = matched_pionp[1].pt()
        tofill['pi_plus2_eta'] = matched_pionp[1].eta()
        tofill['pi_plus2_phi'] = matched_pionp[1].phi()
        tofill['pi_plus2_m'] = pi_p_lv2.M()
        tofill['pi_plus3_pt'] = matched_pionp[2].pt()
        tofill['pi_plus3_eta'] = matched_pionp[2].eta()
        tofill['pi_plus3_phi'] = matched_pionp[2].phi()
        tofill['pi_plus3_m'] = pi_p_lv3.M()
     
        
        tofill['upsilon_m'] = upsilon_lv.M()
    
    
        upsilon_no_neu_lv = pi_m_lv1 + pi_m_lv2 + pi_m_lv3 + pi_p_lv1 + pi_p_lv2 + pi_p_lv3
        print "upsilon_no_neu_lv Px Py Pz E M:",  upsilon_no_neu_lv.Px(),  upsilon_no_neu_lv.Py(),  upsilon_no_neu_lv.Pz(),  upsilon_no_neu_lv.E(),  upsilon_no_neu_lv.M()
        tofill['upsilon_m_no_neu'] = upsilon_no_neu_lv.M()
        
        taum_no_neu_lv = pi_m_lv1 + pi_m_lv2 + pi_m_lv3
        tofill['taum_m_no_neu'] = taum_no_neu_lv.M()
        tofill['taum_mass'] = taum_lv.M()
        
        taup_no_neu_lv = pi_p_lv1 + pi_p_lv2 + pi_p_lv3
        tofill['taup_m_no_neu'] = taup_no_neu_lv.M()
        tofill['taup_mass'] = taup_lv.M()
        
        coll_x_plus = taup_no_neu_lv.Pt() / (taup_no_neu_lv.Pt() + met.pt())
        coll_x_minus = taum_no_neu_lv.Pt() / (taum_no_neu_lv.Pt() + met.pt())

        tofill['upsilon_coll_m'] = upsilon_no_neu_lv.M() / math.sqrt(coll_x_minus * coll_x_plus)

        tofill['r_taum_pt'] = taum_lv.Pt()
        tofill['r_taup_pt'] = taup_lv.Pt()
        
        tofill['gen_taum_pt'] = gen_taum_lv.Pt()
        tofill['gen_taup_pt'] = gen_taup_lv.Pt()
        
#        tofill['vis_taum_eta'] = vis_taum_eta
        
        tofill['orig_vis_taum_phi'] = orig_vis_taum_phi
        tofill['orig_vis_taum_theta'] = orig_vis_taum_theta 
        tofill['orig_vis_taup_phi'] = orig_vis_taup_phi
        tofill['orig_vis_taup_theta'] = orig_vis_taup_theta
        
        
        tofill['local_pi_m_lv1_phi'] = local_pi_m_lv1_phi
        tofill['local_pi_m_lv1_eta'] = local_pi_m_lv1_eta
        tofill['local_pi_m_lv1_pt'] = local_pi_m_lv1_pt
        tofill['local_pi_m_lv1_mass'] = local_pi_m_lv1_mass
        tofill['local_pi_m_lv2_pt'] = local_pi_m_lv2_pt
        tofill['local_pi_m_lv3_pt'] = local_pi_m_lv3_pt
        tofill['local_taum_lv_mass'] = local_taum_lv_mass
        tofill['local_taup_lv_mass'] = local_taup_lv_mass
        
        tofill["local_pi_p_lv1_pt"]  = local_pi_p_lv1_pt
        tofill["local_pi_p_lv2_pt"]  = local_pi_p_lv2_pt   
        tofill["local_pi_p_lv3_pt"]  = local_pi_p_lv3_pt
        
        tofill["initial_leadPt_pi_m_in_AllInZFrame_phi"] =  initial_leadPt_pi_m_in_AllInZFrame_phi
        tofill["initial_leadPt_pi_p_in_AllInZFrame_phi"] =  initial_leadPt_pi_p_in_AllInZFrame_phi
        tofill["toUse_local_taum_lv_mass"] = toUse_local_taum_lv_mass
        tofill['toUse_local_taup_lv_mass'] = toUse_local_taup_lv_mass
        tofill["toUse_local_pi_m_lv1_phi"] = toUse_local_pi_m_lv1_phi
        tofill['toUse_local_pi_p_lv1_phi'] = toUse_local_pi_p_lv1_phi
        
        #toUse pT stuff
        
        tofill["toUse_local_pi_m_lv1_pt"] = toUse_local_pi_m_lv1_pt
        tofill["toUse_local_pi_m_lv2_pt"] = toUse_local_pi_m_lv2_pt
        tofill["toUse_local_pi_m_lv3_pt"] = toUse_local_pi_m_lv3_pt
        tofill["toUse_local_neu_lv_pt"]   = toUse_local_neu_lv_pt
        
        tofill["toUse_local_pi_p_lv1_pt"] = toUse_local_pi_p_lv1_pt 
        tofill["toUse_local_pi_p_lv2_pt"] = toUse_local_pi_p_lv2_pt 
        tofill["toUse_local_pi_p_lv3_pt"] = toUse_local_pi_p_lv3_pt 
        tofill["toUse_local_antineu_lv_pt"] =  toUse_local_antineu_lv_pt
        
        #toUse theta stuff
        
        tofill["toUse_local_pi_m_lv1_theta"] = toUse_local_pi_m_lv1_theta
        tofill["toUse_local_pi_m_lv2_theta"] = toUse_local_pi_m_lv2_theta
        tofill["toUse_local_pi_m_lv3_theta"] = toUse_local_pi_m_lv3_theta
        tofill["toUse_local_neu_lv_theta"] =   toUse_local_neu_lv_theta
        
        tofill["toUse_local_pi_p_lv1_theta"] = toUse_local_pi_p_lv1_theta
        tofill["toUse_local_pi_p_lv2_theta"] = toUse_local_pi_p_lv2_theta
        tofill["toUse_local_pi_p_lv3_theta"] = toUse_local_pi_p_lv3_theta
        tofill["toUse_local_antineu_lv_theta"] = toUse_local_antineu_lv_theta
        
        tofill["toUse_local_pi_m_lv2_phi"] = toUse_local_pi_m_lv2_phi
        tofill["toUse_local_pi_m_lv3_phi"] = toUse_local_pi_m_lv3_phi
        
        tofill["toUse_local_pi_p_lv2_phi"] = toUse_local_pi_p_lv2_phi
        tofill["toUse_local_pi_p_lv3_phi"] = toUse_local_pi_p_lv3_phi
        
        tofill["toUse_local_neu_lv_phi"] = toUse_local_neu_lv_phi
        tofill["toUse_local_antineu_lv_phi"] = toUse_local_antineu_lv_phi
        
        tofill["check1_mass"] = check1_mass #check1_mass and check2_mass give back the taum mass, as they should
        tofill["check2_mass"] = check2_mass #so unrotation works out!
        
        tofill["naive_upsilon_lv_mass"] = naive_upsilon_lv_mass
        
        tofill["global_naive_upsilon_lv_mass"] = global_naive_upsilon_lv_mass
        
        tofill['check_upsilon_mass'] = check_upsilon_lv.M()
        
        tofill["tau_true_mom_mag"] = tau_true_mom_mag
        tofill["naive_tau_mom_mag"] = naive_tau_mom_mag
        tofill["antitau_true_mom_mag"] = antitau_true_mom_mag
        tofill["naive_antitau_mom_mag"] = naive_antitau_mom_mag
        
        tofill["diff_true_minus_naive_antitau_mom_mag"] = diff_true_minus_naive_antitau_mom_mag
        tofill["diff_true_minus_naive_tau_mom_mag"] = diff_true_minus_naive_tau_mom_mag
        
        #Jan Idea
        
        tofill["vis_ditau_px"] = vis_ditau_px
        tofill["vis_ditau_py"] = vis_ditau_py
        tofill["vis_ditau_pz"] = vis_ditau_pz
        
        tofill["true_ditau_px"] = true_ditau_px
        tofill["true_ditau_py"] = true_ditau_py
        tofill["true_ditau_pz"] = true_ditau_pz
        
        tofill["SFx"] = SFx
        tofill["SFy"] = SFy
        tofill["SFz"] = SFz
        
        ntuple.Fill(array('f', tofill.values()))      
#        print 'ntuple is:', ntuple 

    






file_out.cd()
#taup_ntuple.Write()
#taum_ntuple.Write()
#pEff_reco_pi.Write()
#pEff_lostTrack.Write()
ntuple.Write()
file_out.Close()

