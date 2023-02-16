import numpy as np
import numpy.lib.recfunctions as rf  
import functions 

DEFAULTRHO = 30
DRT = 0.3
MHIGGS = 125.25


def structToArray(structarray,key):
    return np.array([k[0] for k in structarray[[key]]])

def pseudoRapToPolar(eta):
    return 2*np.arctan(np.exp(-eta))


def angleBetweenVectorsInPolarCoordinats(x1,x2):
    v1 = (x1[0]*np.sin(x1[1])*np.cos(x1[2]), x1[0]*np.sin(x1[1])*np.sin(x1[2]), x1[0]*np.cos(x1[1]))
    v2 = (x2[0]*np.sin(x2[1])*np.cos(x2[2]), x2[0]*np.sin(x2[1])*np.sin(x2[2]), x2[0]*np.cos(x2[1]))
    
    v1v2 = dotProduct(v1,v2)
    v1len = np.sqrt(dotProduct(v1,v1))
    v2len = np.sqrt(dotProduct(v2,v2))

    return np.arccos(v1v2/(v1len*v2len))

def dotProduct(v1,v2):
    out = 0
    for x,y in zip(v1,v2):
        out += x*y
    return out

def dist(v1,v2):
    delta = v2 - v1
    return np.sqrt(dotProduct(delta,delta))


def axisValueTransformToMu(axisVal,mu,sigma,n):
    minval = mu - n*sigma
    maxval = mu + n*sigma
    delta = maxval - minval
    return minval + delta * axisVal

def axisValueTransformToMu1Mu2(axisVal, mu1, mu2 ,sigma1, sigma2, axis2to1transform, n):
    x = [mu1-n*sigma1,axis2to1transform(mu2+n*sigma2),mu1+n*sigma1,axis2to1transform(mu2-n*sigma2)]
    minval = min(x)
    maxval = max(x)
    delta = maxval - minval
    return minval + delta * axisVal


def updateDataSet(path,key,data):
    olddata = np.load(path)
    newdata = updateStructArray(olddata, key, data)
    np.save(path,newdata.data)

def updateStructArray(olddata, key, data):
    keys = olddata.dtype.names 
    if key in keys:
        olddata[key] = data
        newdata = olddata
    else:
        newdata = rf.append_fields(olddata,key,data, usemask=False)
    return newdata


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

def getFacs(events, bjet1_phi_key, bjet1_eta_key, bjet2_phi_key, bjet2_eta_key):

    b1vp = np.array([np.ones(len(events)),structToArray(events, bjet1_phi_key),pseudoRapToPolar(structToArray(events, bjet1_eta_key))])
    b2vp = np.array([np.ones(len(events)),structToArray(events, bjet2_phi_key),pseudoRapToPolar(structToArray(events, bjet2_eta_key))])

    alpha = angleBetweenVectorsInPolarCoordinats(b1vp,b2vp)
    facs  = MHIGGS**2/(2*(1-np.cos(alpha)))

    return facs

def getSigmas(data, ptKey, etaKey, NSCtable, rhokey=None,):
    pts = structToArray(data, ptKey)
    etas = structToArray(data,etaKey)
    if rhokey is None:
        rhos = [DEFAULTRHO] * len(pts)
    else:
        rhos = structToArray(data,rhokey)

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
            sigma = functions.sigmaBjetAcc(pt,n,c,s,d)
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
    delta11 = dist(bjet1,b1)
    delta1s1 = dist(bjet1shifted,b1)
    delta11s = dist(bjet1,b1shifted)
    delta22 = dist(bjet2,b2)
    delta2s2 = dist(bjet2shifted,b2)
    delta22s = dist(bjet2,b2shifted)
    delta12 = dist(bjet1,b2)
    delta1s2 = dist(bjet1shifted,b2)
    delta12s = dist(bjet1,b2shifted)
    delta21 = dist(bjet2,b1)
    delta2s1 = dist(bjet2shifted,b1)
    delta21s = dist(bjet2,b1shifted)

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
    genB2phi = data[genB1_phi_key]
    genB1phi = data[genB2_phi_key]

    bjet1eta = data[bjet1_eta_key]
    bjet2eta = data[bjet2_eta_key]
    genB2eta = data[genB1_eta_key]
    genB1eta = data[genB2_eta_key]

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
        
            
def getIndexForCondition(data, key, condition, comparevalue):
    values = structToArray(data, key)
    indices = []
    for i, v in enumerate(values):
        if condition(v, comparevalue):
            indices.append(i)
    return indices

def removeEvents(data, indexList):
    indexSet = list(set(indexList))
    return np.delete(data,indexSet)


# arr = np.array([(0,'a'),(1,'b'),(2,'c'),(3,'d'),(4,'e'),(5,'f')],dtype=[('position','i4'),('letter','U10')])
# print(removeEvents(arr,[1,1,4]))






# print(bjetMatching(4,-1,1,2,4.1,-1.1,1.1,2.1))
# print(bjetMatching(-1,4,2,1,4.1,-1.1,1.1,2.1))
# print(bjetMatching(-5,4,2,1,4.1,-1.1,1.1,2.1))
# print(bjetMatching(4,-1,-3.141,2,4.1,-1.1,3.141,2.1))
# print(bjetMatching(4,-1,4.05,-1.05,4.1,-1.1,1.1,2.1))

# a = np.array([4,-1,-5,4,4])
# b = np.array([-1,4,4,-1,-1])
# c = np.array([1,2,2,-3,4.05])
# d = np.array([2,1,1,-3.141,4.05])
# e = np.array([4.1]*5)
# f = np.array([-1.1] * 5)
# g = np.array([1.1,1.1,1.1,3.141,1.1])
# h = np.array([2.1]*5)

# print(bjetMatching(a,b,c,d,e,f,g,h))





    


