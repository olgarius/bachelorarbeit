import numpy as np
from pathlib import Path

import util

DATAPATH = Path('/nfs/dust/cms/user/kramerto/hbt_resonant_run2/HHSkims/SKIMS_uhh_2017_v2_31Aug22/SKIM_ggF_BulkGraviton_m300/output0.npz')
data1= np.load(DATAPATH)
data = data1['events']
DR =0.3

def deltaR2(eta1, phi1, eta2=None, phi2=None):
    """Take either 4 arguments (eta,phi, eta,phi) or two objects that have 'eta', 'phi' methods)"""
    de = eta1 - eta2
    dp = deltaPhi(phi1, phi2)
    return de*de + dp*dp

def deltaR( **args ):
    return np.sqrt( deltaR2(**args) )

def deltaPhi( p1, p2):
    '''Computes delta phi, handling periodic limit conditions.'''
    
    newcol = p1 - p2
    debug_count = 0
    while (newcol > np.pi).any():
        newcol= np.where(newcol>np.pi, newcol - 2*np.pi, newcol)
        debug_count += 1
        if debug_count == 10:
            print(f"looped in >np.pi condition {debug_count} times!")
    debug_count = 0

    while (newcol < -np.pi).any():
        newcol=np.where(newcol<-np.pi, newcol + 2*np.pi, newcol)
        debug_count += 1
        if debug_count == 10:
            print(f"looped in < -np.pi condition {debug_count} times!")
    return newcol


def apply_delta_r_selection(var1, var2):
    deltaR_mask1 = (var1 <= DR) & (var2 <= DR)
    deltaR_mask2 = (var1 > DR) & (var2 > DR)

    return ~deltaR_mask1 & ~deltaR_mask2



bjet1_eta =  util.structToArray(data,'bjet1_eta')
m_genB1_eta =  util.structToArray(data,'genBQuark1_eta')
bjet2_eta =  util.structToArray(data,'bjet2_eta')
m_genB2_eta =  util.structToArray(data,'genBQuark2_eta')
bjet1_phi =  util.structToArray(data,'bjet1_phi')
m_genB1_phi =  util.structToArray(data,'genBQuark1_phi')
bjet2_phi =  util.structToArray(data,'bjet2_phi')
m_genB2_phi =  util.structToArray(data,'genBQuark2_phi')


deltaR_det1_gen1 = deltaR(eta1=bjet1_eta, phi1=bjet1_phi, eta2 = m_genB1_eta, phi2 = m_genB1_phi)
deltaR_det1_gen2 = deltaR(eta1=bjet1_eta, phi1=bjet1_phi, eta2 = m_genB2_eta, phi2 = m_genB2_phi)
deltaR_det2_gen1 = deltaR(eta1=bjet2_eta, phi1=bjet2_phi, eta2 = m_genB1_eta, phi2 = m_genB1_phi)
deltaR_det2_gen2 = deltaR(eta1=bjet2_eta, phi1=bjet2_phi, eta2 = m_genB2_eta, phi2 = m_genB2_phi)

# Filter Daten mit Maske raus
eventMask = apply_delta_r_selection(deltaR_det1_gen1, deltaR_det1_gen2) & apply_delta_r_selection(deltaR_det2_gen1, deltaR_det2_gen2)
print(f'{np.sum(eventMask)}/{eventMask.size} = {np.mean(eventMask):.2f}')

# zuordnung zu bjet1 oder bjet2
# true_bjet1_pt = np.where(deltaR_det1_gen1 < DR, m_genB1_pt, m_genB2_pt)
# true_bjet2_pt= np.where(deltaR_det2_gen1 < DR, m_genB1_pt, m_genB2_pt)