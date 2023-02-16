import numpy as np

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


