import numpy as np


DEFAULTRHO = 30
DRT = 0.3
MHIGGS = 125.25


def pseudoRapToPolar(eta):
    '''
    Converts pseudo rapidity eta to polar angle theta

    returns theta in range (-pi, pi)

    Parameters
    ----------

    eta : float

    '''
    return 2*np.arctan(np.exp(-eta))


def polarCoordinatesToCartesian(x):
    '''
    Converts polar coordinates to cartesian coordinates

    returns a tuple of size 3 with entries x, y, z

    Parameters
    ----------

    x : Tuple of size 3
        with the entries r, phi, theta  
    '''

    return (x[0]*np.sin(x[2])*np.cos(x[1]),
            x[0]*np.sin(x[2])*np.sin(x[1]),
            x[0]*np.cos(x[2]))

def detCoordinatesToCartesian(x):
    '''
    Converts detector coordinates to cartesian coordinates

    returns a tuple of size 3 with entries x, y, z

    Parameters
    ----------

    x : Tuple of size 3
        with the entries r, phi, eta (r is only in transversal direction!)  
    '''
    return (x[0] * np.cos(x[1]), x[0] * np.sin(x[1]),x[0] * np.sinh(x[2]) )

def angleBetweenVectorsInCartesianCoordinates(x1,x2):
    '''
    Calculates the angle between two vectors in cartesian coordinates

    returns angle in rad  

    Parameters
    ----------

    x1 : Tuple of size 3
        with entries x, y, z

    x2 : Tuple of size 3
        with entries x, y, z

    '''
    v1v2 = dotProduct(x1,x2)
    v1len = np.sqrt(dotProduct(x1,x1))
    v2len = np.sqrt(dotProduct(x2,x2))

    return np.arccos(v1v2/(v1len*v2len))

def angleBetweenVectorsInPolarCoordinats(x1,x2):
    '''
    Calculates the angle between two vectors in polar coordinates

    returns angle in rad  

    Parameters
    ----------

    x1 : Tuple of size 3
        with entries r, phi, theta

    x2 : Tuple of size 3
        with entries r, phi, theta

    '''
    v1 = polarCoordinatesToCartesian(x1)
    v2 = polarCoordinatesToCartesian(x2)
    
    return angleBetweenVectorsInCartesianCoordinates(v1,v2)

def angleBetweenVectorsInDetCoordinates(x1,x2):
    '''
    Calculates the angle between two vectors in detector coordinates

    returns angle in rad  

    Parameters
    ----------

    x1 : Tuple of size 3
        with entries r, phi, eta

    x2 : Tuple of size 3
        with entries r, phi, eta

    '''
    v1 = detCoordinatesToCartesian(x1)
    v2 = detCoordinatesToCartesian(x2)
    
    return angleBetweenVectorsInCartesianCoordinates(v1,v2)


def dotProduct(v1,v2):
    '''
    Calculates the dot product of two vectors

    returns float
        dot product of vector 1 and 2

    Parameters
    ----------

    v1: Iterable
        n-dim vector 1
    v1: Iterable 
        m-dim vector 2

    '''
    out = 0
    for x,y in zip(v1,v2):
        out += x*y
    return out

    
def dist(p1,p2):
    '''
    Calculates the distanance between two points

    returns float
        distance between point 1 and point 2

    Parameters
    ----------

    v1: Iterable
        position of point 1
    v1: Iterable 
        position of point 2

    '''
    delta = p2 - p1
    return np.sqrt(dotProduct(delta,delta))


def axisValueTransformToMu(axisVal,mu,sigma,n):
    '''
    maps any value to a sigma environment of a given value mu. 0 gets mapped to mu - n * sigma, 1 to mu + * sigma  

    returns float
        transformed value 

    Parameters
    ----------

    axisValue : float
        value to be transformed

    mu : float
    
    sigma : float

    n : float
        sigma range 0 and 1 get mapped to 
    '''
    minval = mu - n*sigma
    maxval = mu + n*sigma
    delta = maxval - minval
    return minval + delta * axisVal

def axisValueTransformToMu1Mu2(axisVal, mu1, mu2 ,sigma1, sigma2, axis2to1transform, n):
    '''
    maps any value to the largest sigma environment of two given value mu1 and mu2. 0 gets mapped to the smallest mu_i+- n*sigma_i, and 1 to the largest mu_i +- n * sigma. It is possible to transform mu2 +- n*sigma2 to the axis of mu1.  

    returns float
        transformed value 

    Parameters
    ----------

    axisValue : float
        value to be transformed

    mu1 : float
    mu2 : float
    
    sigma1 : float
    sigma2 : float

    axis2to1transform : function(x)
        function that mapps from the mu2 space to the mu1 space

    n : float
        sigma range 0 and 1 get mapped to 
    '''
    x = [mu1-n*sigma1,axis2to1transform(mu2+n*sigma2),mu1+n*sigma1,axis2to1transform(mu2-n*sigma2)]
    minval = min(x)
    maxval = max(x)
    delta = maxval - minval
    return minval + delta * axisVal


def bEiofbEj(H_mass, B1_e, alpha):
    '''
    Calculates the energy of a b-jet in a higgs-decay dependent on the higgsmass, the energy of the other b-jet and the angle of both b-jets. High energies are assumed, such that the b-masses can be neglected.

    returns float
        calculated energie of the second b-jet

    Parameters
    ----------

    H_mass : float
        mass of the higgs boson

    B1_e : float
        energy of the first b-jet
    
    alpha : float
        angle between the b-jets

    '''
    return (H_mass**2/(2*B1_e*(1-np.cos(alpha))))




def makeChi2(x1,sigma):
    def f(xf):
        return ((x1-xf)/sigma)**2
    return f
    

def sigmaBjetAcc(pT,N,S,C,d):
    squared = np.sign(N)*N**2/pT**2+S**2/pT**d+C**2
    return np.sqrt(squared)

def multipleAnd(*x):
    out = np.ones(len(x[0]))
    for b in x:
        out = np.logical_and(out,b)
    return out

def inverse2x2(A):
    det =  A[0][0]*A[1][1]-A[0][1]*A[1][0]
    if det == 0:
        return np.array([[np.nan, np.nan],[np.nan, np.nan]])

    return np.array([
        [A[1][1],-A[1][0]],
        [-A[0][1],A[0][0]]]
        )/det


 
def vT_x_A_x_v(v,A):
    return v.dot(A.dot(v))


      

