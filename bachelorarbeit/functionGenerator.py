import mathutil as u
import structutil as su

HMASS = 125.25



def getBBE2Chi2(data):
    BE2_mes = su.structToArray(data,'bjet2_e') 
    fac = su.structToArray(data,'bie_to_bje')
    sigma = su.structToArray(data, 'bjet2_sigma') * BE2_mes

    farray = []
    farray2 = []

    for e,f,s in zip(BE2_mes,fac,sigma):
        CHi2,BE2_fit = genBBE2Chi2(e,f,s)
                
        farray.append(CHi2)
        farray2.append(BE2_fit)

    return farray, [1], farray2
          
          
def genBBE2Chi2(BE2_mes,fac,sigma):
    def BE2_fit(BE1_fit):
        return fac/BE1_fit
    def BE2Chi2(BE1_fit):
        BE2_fitval = BE2_fit(BE1_fit)
        return ((BE2_fitval - BE2_mes)/sigma)**2
    return BE2Chi2,BE2_fit

def getBBE1Chi2(data):
    BE1_mes = su.structToArray(data,'bjet1_e')
    sigma = su.structToArray(data, 'bjet1_sigma') * BE1_mes

    farray = []

    for e,s in zip(BE1_mes,sigma):
        Chi2 =genBBE1Chi2(e,s)

        farray.append(Chi2)
    
    return farray, [1]

    
def genBBE1Chi2(BE1_mes,sigma):
    def BE1Chi2(BE1_fit):
        return ((BE1_fit - BE1_mes)/sigma)**2
    return BE1Chi2



