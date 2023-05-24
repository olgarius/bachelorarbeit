import numpy as np


import structutil as su 
import mathutil as mu

DRT = 0.4

def getFacs(events, bjet1_pt_key,bjet1_phi_key, bjet1_eta_key,bjet2_pt_key, bjet2_phi_key, bjet2_eta_key):

    b1vp = np.array([su.structToArray(events, bjet1_pt_key),su.structToArray(events, bjet1_phi_key),su.structToArray(events, bjet1_eta_key)])
    b2vp = np.array([su.structToArray(events, bjet2_pt_key),su.structToArray(events, bjet2_phi_key),su.structToArray(events, bjet2_eta_key)])

    alpha = mu.angleBetweenVectorsInDetCoordinates(b1vp,b2vp)
    facs  = mu.MHIGGS**2/(2*(1-np.cos(alpha)))

    return facs

def getSigmas(data, ptKey, etaKey, NSCtable, rhokey=None,):
    pts = su.structToArray(data, ptKey)
    etas = su.structToArray(data,etaKey)
    if rhokey is None:
        rhos = [mu.DEFAULTRHO] * len(pts)
    else:
        rhos = su.structToArray(data,rhokey)

    sigmas = []
    wc = 0

    for pt, eta, rho in zip(pts,etas,rhos):

        sigmabool = False
        for l in NSCtable:
            # print(eta, l[0] , eta , l[1] , rho > l[2] , rho , l[3] , pt , l[5] , pt , l[6])
            if eta > l[0] and eta < l[1] and rho > l[2] and rho < l[3] and pt > l[5] and pt < l[6]:
                n = l[7]
                s = l[8]
                c = l[9]
                d = l[10]
                sigmabool = True
                break
        if sigmabool:
            sigma = mu.sigmaBjetAcc(pt,n,s,c,d)
        else:
            sigma = 0.1 * pt
            wc += 1
        
        sigmas.append(sigma)

    print('WC:',wc)
    return(sigmas)


def singleBjetMatching(bjet1_eta,bjet2_eta,bjet1_phi,bjet2_phi, genB1_eta,genB2_eta,genB1_phi,genB2_phi):
    bjet1 = np.array([bjet1_eta,bjet1_phi])
    bjet1shifted = np.array([bjet1_eta,bjet1_phi+2*np.pi])
    bjet2 = np.array([bjet2_eta,bjet2_phi])
    bjet2shifted = np.array([bjet2_eta,bjet2_phi+2*np.pi])
    b1 = np.array([genB1_eta, genB1_phi])
    b1shifted = np.array([genB1_eta,genB1_phi+2*np.pi])
    b2 = np.array([genB2_eta, genB2_phi])
    b2shifted = np.array([genB2_eta,genB2_phi+2*np.pi])
    delta11 = mu.dist(bjet1,b1)
    delta1s1 = mu.dist(bjet1shifted,b1)
    delta11s = mu.dist(bjet1,b1shifted)
    delta22 = mu.dist(bjet2,b2)
    delta2s2 = mu.dist(bjet2shifted,b2)
    delta22s = mu.dist(bjet2,b2shifted)
    
    delta12 = mu.dist(bjet1,b2)
    delta1s2 = mu.dist(bjet1shifted,b2)
    delta12s = mu.dist(bjet1,b2shifted)
    delta21 = mu.dist(bjet2,b1)
    delta2s1 = mu.dist(bjet2shifted,b1)
    delta21s = mu.dist(bjet2,b1shifted)

    d11 = min(delta11,delta1s1,delta11s)
    d22 = min(delta22,delta2s2,delta22s)
    d12 = min(delta12,delta1s2,delta12s)
    d21 = min(delta21,delta2s1,delta21s)
    
    out = -1

    if d11 < DRT and d22 < DRT and d12 > DRT and d21 > DRT:
        out = 0
    if d11 > DRT and d22 > DRT and d12 < DRT and d21 < DRT:
        out = 1
    
    return out

def matchBjets(data, bjet1_e_key, bjet2_e_key, bjet1_pt_key, bjet2_pt_key, bjet1_phi_key, bjet2_phi_key, genB1_phi_key,genB2_phi_key,bjet1_eta_key, bjet2_eta_key, genB1_eta_key,genB2_eta_key ):
    notusabeleEvents = []

    new_bjet1e = []
    new_bjet2e = []
    
    new_bjet1pt = []
    new_bjet2pt = []
     
    new_bjet1phi = []
    new_bjet2phi = []
     
    new_bjet1eta = []
    new_bjet2eta = []
     

    bjet1e = data[bjet1_e_key]
    bjet2e = data[bjet2_e_key]


    bjet1pt = data[bjet1_pt_key]
    bjet2pt = data[bjet2_pt_key]

    bjet1phi = data[bjet1_phi_key]
    bjet2phi = data[bjet2_phi_key]
    genB1phi = data[genB1_phi_key]
    genB2phi = data[genB2_phi_key]

    bjet1eta = data[bjet1_eta_key]
    bjet2eta = data[bjet2_eta_key]
    genB1eta = data[genB1_eta_key]
    genB2eta = data[genB2_eta_key]

    for i in range(len(bjet1e)):
        switch = singleBjetMatching(bjet1eta[i], bjet2eta[i], bjet1phi[i], bjet2phi[i], genB1eta[i], genB2eta[i], genB1phi[i], genB2phi[i])

        if switch is 0:
            new_bjet1e.append(bjet1e[i])
            new_bjet2e.append(bjet2e[i])
            
            new_bjet1pt.append(bjet1pt[i])
            new_bjet2pt.append(bjet2pt[i])
            
            new_bjet1phi.append(bjet1phi[i])
            new_bjet2phi.append(bjet2phi[i])
           
            new_bjet1eta.append(bjet1eta[i])
            new_bjet2eta.append(bjet2eta[i])
        
        if switch is 1:
            new_bjet1e.append(bjet2e[i])
            new_bjet2e.append(bjet1e[i])
            
            new_bjet1pt.append(bjet2pt[i])
            new_bjet2pt.append(bjet1pt[i])
            
            new_bjet1phi.append(bjet2phi[i])
            new_bjet2phi.append(bjet1phi[i])
           
            new_bjet1eta.append(bjet2eta[i])
            new_bjet2eta.append(bjet1eta[i])
        
        if switch is -1:
            notusabeleEvents.append(i)

            new_bjet1e.append(bjet1e[i])
            new_bjet2e.append(bjet2e[i])
                
            new_bjet1pt.append(bjet1pt[i])
            new_bjet2pt.append(bjet2pt[i])
                
            new_bjet1phi.append(bjet1phi[i])
            new_bjet2phi.append(bjet2phi[i])
            
            new_bjet1eta.append(bjet1eta[i])
            new_bjet2eta.append(bjet2eta[i])

    return np.array(new_bjet1e), np.array(new_bjet2e), np.array(new_bjet1pt), np.array(new_bjet2pt), np.array(new_bjet1phi), np.array(new_bjet2phi), np.array(new_bjet1eta), np.array(new_bjet2eta), notusabeleEvents

def deltaR(v1, v2):
    dPhi = abs(v1[1]-v2[1])
    return np.sqrt((v1[0] - v2[0])**2 +  (np.minimum(dPhi, abs(2*np.pi - dPhi)))**2)

def deltaRmask(eta1,eta2,phi1,phi2, gen_eta1,gen_eta2,gen_phi1,gen_phi2):
    bjet1 = np.array([eta1,phi1])
    bjet2 = np.array([eta2,phi2])
    b1 = np.array([gen_eta1, gen_phi1])
    b2 = np.array([gen_eta2, gen_phi2])

    d11 = deltaR(bjet1,b1)
    d12 = deltaR(bjet1,b2)
    d21 = deltaR(bjet2,b1)
    d22 = deltaR(bjet2,b2)

    l = len(eta1)
    out =  - np.ones(l)
    ones = np.ones(l)
    zeroes = np.zeros(l)

    out = np.where(mu.multipleAnd(d11 < DRT, d22 < DRT, d12 > DRT, d21 > DRT), zeroes, out)
    out = np.where(mu.multipleAnd(d11 > DRT, d22 > DRT, d12 < DRT, d21 < DRT), ones, out)

    return out



def deltaRmatching(structArray,eta1_key,eta2_key,phi1_key,phi2_key, gen_eta1_key,gen_eta2_key,gen_phi1_key,gen_phi2_key, keyPairList):
    eta1 = structArray[eta1_key]
    eta2 = structArray[eta2_key]
    geneta1 = structArray[gen_eta1_key]
    geneta2 = structArray[gen_eta2_key]

    phi1 = structArray[phi1_key]
    phi2 = structArray[phi2_key]
    genphi1 = structArray[gen_phi1_key]
    genphi2 = structArray[gen_phi2_key]

    mask = deltaRmask(eta1,eta2,phi1,phi2,geneta1,geneta2,genphi1,genphi2)
    

    unuseableeEvents = np.where(mask == -1 )[0].tolist()
    

    for kp in keyPairList:
        structArray =  su.switchEntries(structArray, kp[0],kp[1], mask)

    return structArray, unuseableeEvents








def readTable(path):
    with open(path, 'r') as f:
        f.readline()
        out = []
        for x1 in f:
            x2 = x1.split(' ')
            x3 = [float(x) for x in x2 if x is not ''] 
            out.append(x3)
        f.close()
    return out
 
def combineArrayToMatrixArray(a,b,c,d):
    return np.dstack((np.c_[a,b], np.c_[c,d]))