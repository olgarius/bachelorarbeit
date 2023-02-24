import numpy as np


DEFAULTRHO = 30
DRT = 0.3
MHIGGS = 125.25


def pseudoRapToPolar(eta):
    return 2*np.arctan(np.exp(-eta))

def polarCoordinatesToCartesian(x):
    return (x[0]*np.sin(x[2])*np.cos(x[1]),
            x[0]*np.sin(x[2])*np.sin(x[1]),
            x[0]*np.cos(x[2]))

def detCoordinatesToCartesian(pt, phi, eta):
    return (pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta) )

def angleBetweenVectorsInCartesianCoordinates(x1,x2):
    v1v2 = dotProduct(x1,x2)
    v1len = np.sqrt(dotProduct(x1,x1))
    v2len = np.sqrt(dotProduct(x2,x2))

    return np.arccos(v1v2/(v1len*v2len))

def angleBetweenVectorsInPolarCoordinats(x1,x2):
    v1 = polarCoordinatesToCartesian(x1)
    v2 = polarCoordinatesToCartesian(x2)
    
    return angleBetweenVectorsInCartesianCoordinates(v1,v2)

def angleBetweenVectorsInDetCoordinates(x1,x2):
    v1 = detCoordinatesToCartesian(x1[0],x1[1],x1[2])
    v2 = detCoordinatesToCartesian(x2[0],x2[1],x2[2])
    
    return angleBetweenVectorsInCartesianCoordinates(v1,v2)


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

def bEiofbEj(H_mass,B1_e,alpha):
    return (H_mass**2/(2*B1_e*(1-np.cos(alpha))))

def dauEiofdauEj(H_mass,Dau1_e,alpha):
    return (H_mass**2/(2*Dau1_e*(1-np.cos(alpha))))



def makeChi2(x1,sigma):
    def f(xf):
        return ((x1-xf)/sigma)**2
    return f


def sigmaBjet(BjetE):
    return 0.1*BjetE

def sigmaBjetAcc(pT,N,S,C,d):
    squared = np.sign(N)*N**2/pT**2+S**2/pT**d+C**2
    return np.sqrt(squared)







      

