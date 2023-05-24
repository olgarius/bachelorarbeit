import plotting 
import structutil as su
import datautil as du

import matplotlib.pyplot as plt

import numpy as np



from PATHS import WORKINGDATAPATH
from PATHS import PATHTOPLOTSMET


data= np.load(WORKINGDATAPATH)

edgevalue = su.structToArray(data,'taufit_valueOnEdge')

edgeindex = np.where(edgevalue<1)[0]
data2 = np.take(data, edgeindex )

print(len(data2)/len(data)*100,'%')

PATHTOPLOTSMET = PATHTOPLOTSMET+ ' NoEdge'

data2= data

fitmet_x = su.structToArray(data2,'fitmet_x')
met_x = su.structToArray(data2,'met_x')


fitmet_y = su.structToArray(data2,'fitmet_y')
met_y = su.structToArray(data2,'met_y')

absmet = np.sqrt(met_x**2+met_y**2)
fitabsmet = np.sqrt(fitmet_x**2+fitmet_y**2)

a = su.structToArray(data2,'fitdau_a')

edgevalue = su.structToArray(data2,'taufit_valueOnEdge')

edgeindex = np.where(edgevalue<1)[0]

chi2 = su.structToArray(data2,'dau_chi2val')

invCovMat = du.combineArrayToMatrixArray(data2['met_invcov00'], data2['met_invcov01'], data2['met_invcov01'], data2['met_invcov11'])
covMat00  = su.structToArray(data2,'met_cov00')
covMat01  = su.structToArray(data2,'met_cov01')
covMat11  = su.structToArray(data2,'met_cov11')


chi2val = su.structToArray(data2,'dau_chi2val')

fitmet_x_NoEdge = np.take(fitmet_x,edgeindex)
fitmet_y_NoEdge = np.take(fitmet_y,edgeindex)


met_x_NoEdge = np.take(met_x,edgeindex)
met_y_NoEdge = np.take(met_y,edgeindex)

fitabsmet_NoEdge = np.take(fitabsmet,edgeindex)
absmet_NoEdge = np.take(absmet,edgeindex)

a_NoEdge = np.take(a,edgeindex)


met = np.c_[met_x,met_y]

asigma = su.structToArray(data2,'fitdau_asigma')
a = su.structToArray(data2,'fitdau_a')
tau1_ex = su.structToArray(data2,'dau1_ex')
tau1_ey = su.structToArray(data2,'dau1_ey')
tau1_eT = np.sqrt(tau1_ex**2,tau1_ey**2)
tau2_ex = su.structToArray(data2,'dau2_ex')
tau2_ey = su.structToArray(data2,'dau2_ey')
tau2_eT = np.sqrt(tau2_ex**2,tau2_ey**2)
TauE1_mes = su.structToArray(data2,'dau1_e') 
TauE2_mes = su.structToArray(data2,'dau2_e') 

fac = su.structToArray(data2,'tauie_to_tauje')

# fitmetsigma = np.abs(asigma * (tau1_eT-fac/(TauE1_mes*TauE2_mes*a**2)*tau2_eT))

k =  fac/(TauE1_mes*TauE2_mes)

fitmetsigma_part2x = (tau2_ex *(k/a -1) + (a-1)*tau1_ex+met_x)
fitmetsigma_part2y = (tau2_ey *(k/a -1) + (a-1)*tau1_ey+met_y)
fitmetsigma_part1x = (tau1_ex - k*tau2_ex/a**2) * fitmetsigma_part2x
fitmetsigma_part1y = (tau1_ey - k*tau2_ey/a**2) * fitmetsigma_part2y



fitmetsigma = (fitmetsigma_part1x+fitmetsigma_part1y)/np.sqrt(fitmetsigma_part2x**2+fitmetsigma_part2y**2)
measmetsigma = np.sqrt((covMat00**2*met_x**2 + covMat11**2*met_y**2)/(absmet**2) + 2*met_x*met_y*covMat01/absmet**3)

print(np.mean(fitmetsigma))


postfit_pull2 = np.sqrt(((0-fitabsmet)/fitmetsigma)**2)

print((np.where(postfit_pull2>2100)[0]))

# print('mean: ',np.mean(data2['met_invcov00']),np.mean(data2['met_invcov01']),np.mean(data2['met_invcov11']))
# print('std: ',np.std(data2['met_invcov00']),np.std(data2['met_invcov01']),np.std(data2['met_invcov11']))



prefit_pull = np.sqrt(np.einsum('ij,ij->i',met,np.einsum('ijk,ij->ik',invCovMat,met)))
postfit_pull = np.sqrt(chi2val)

prefit_pull2 = absmet/measmetsigma 

def standardNormalDistx2(x):
    return np.exp(-x**2/2)/np.sqrt(np.pi*2)*2

plotting.plotHistCompare(PATHTOPLOTSMET,r'$\sqrt{\left(\frac{0-met}{\sigma}\right)^2}$',r'MET Pull', [0,10],200,(prefit_pull2,r'measured '),(postfit_pull2,r'fitted '), comparefunctions=[(standardNormalDistx2,r'standard normal distribution $\cdot2$')], ylim=[0,1.25],pdf=True)
# (postfit_pull,r'met$^f$ prefit $\sigma$ '),


plotting.plotHistCompare(PATHTOPLOTSMET,r"missing $p_{Ti}$/GeV", r"Change of missing $p_T$", [-250,250],100,(met_x,r"missing $p_{Tx}^m$"),(fitmet_x,r"missing $p_{Tx}^f$"),(met_y,r"missing $p_{Ty}^m$"),(fitmet_y,r"missing $p_{Ty}^f$"),alttitle='metcomparison',ylim=[0,0.025],pdf=True)
plotting.plotHistCompare(PATHTOPLOTSMET,r"missing $p_{T}$/GeV", r"Change of missing $p_T$", [0,300],100,(absmet,r"measured "),(fitabsmet,r"fitted"),alttitle='absmetcomparison',ylim=[0,0.025],pdf=True)
# plotting.plot2d2(fitabsmet,a,PATHTOPLOTSMET,'missing p_T','a',[0,300],[1,max(a)])
# plotting.plot2d2(absmet,chi2,PATHTOPLOTSMET,'missing p_T','chi2value',[0,300],[0,200])
# plotting.plot2d2(fitabsmet,chi2,PATHTOPLOTSMET,'missing p_T_fit','chi2value',[0,300],[0,200])
plotting.plotHistCompare(PATHTOPLOTSMET, r"$\frac{\sigma_{MET}}{MET}$", r"$\sigma_{MET}$ distribution", [0,12],100,(abs(measmetsigma)/abs(absmet),r'measured'),(abs(fitmetsigma)/abs(fitabsmet),r'fitted '), ylim=[0,1],alttitle='sigmamet',pdf=True)
plotting.plotHistCompare(PATHTOPLOTSMET, r"$\frac{\sigma_s}{s}$", r"$\sigma_a$ distribution", [0,3],100,(asigma/a,r'$\frac{\sigma_s}{s}$'),ylim=[0,6] ,alttitle='sigmaa',pdf=True)




# plotting.plotHistCompare(PATHTOPLOTSMET,r"missing $p_{Ti}$/GeV", r"Change of missing $p_T$", [-250,250],100,(met_x_NoEdge,r"missing $p_{Tx}^m$"),(fitmet_x_NoEdge,r"missing $p_{Tx}^f$"),(met_y_NoEdge,r"missing $p_{Ty}^m$"),(fitmet_y_NoEdge,r"missing $p_{Ty}^f$"),alttitle='metcomparison_NoEdge',ylim=[0,0.01])
# plotting.plotHistCompare(PATHTOPLOTSMET,r"missing $p_{T}$/GeV", r"Change of missing $p_T$", [0,300],100,(absmet_NoEdge,r"missing $p_{T}^m$"),(fitabsmet_NoEdge,r"missing $p_{T}^f$"),alttitle='absmetcomparison_NoEdge',ylim=[0,0.015])
plotting.plot2d2(fitmetsigma,fitabsmet,PATHTOPLOTSMET,'sigma MET','MET',[0,300],[0,300])
