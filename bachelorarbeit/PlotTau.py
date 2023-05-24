import plotting 
import structutil as su
import datautil as du

import matplotlib.pyplot as plt

import numpy as np
from scipy.special import gamma


from PATHS import WORKINGDATAPATH
from PATHS import PATHTOPLOTS1D
from PATHS import PATHTOPLOTS2D
from PATHS import PATHTOPLOTSTAU

BINSIZE = 10
XLIM = 2000
YLIM = 10000

#load Data and select events

data= np.load(WORKINGDATAPATH)

edgevalue = su.structToArray(data,'taufit_valueOnEdge')

edgeindex = np.where(edgevalue<1)[0]
data2 = np.take(data, edgeindex )

print(len(data2)/len(data)*100,'%')

PATHTOPLOTSTAU = PATHTOPLOTSTAU+ ' NoEdge'

# data2= data


e1fit = su.structToArray(data2,'fitdau1_e')
e2fit = su.structToArray(data2,'fitdau2_e')




e1 = su.structToArray(data2,'dau1_e')
e2 = su.structToArray(data2,'dau2_e')
e1gen = su.structToArray(data2,'genLepton1_e')
e2gen = su.structToArray(data2,'genLepton2_e')
fac = su.structToArray(data2,'tauie_to_tauje')
genfac = su.structToArray(data2,'gentauie_to_gentauje')

phi1 = su.structToArray(data2,'dau1_phi')
eta1 = su.structToArray(data2,'dau1_eta')
eta2 = su.structToArray(data2,'dau2_eta')
phi2 = su.structToArray(data2,'dau2_phi')

# edgevalue = su.structToArray(data2,'taufit_valueOnEdge')

# edgeindex = np.where(edgevalue<1)[0]

phi1gen = su.structToArray(data2,'genLepton1_phi')
eta1gen = su.structToArray(data2,'genLepton1_eta')


chi2sigma1 = su.structToArray(data2,'fitdau1_esigma')

a = su.structToArray(data2,'fitdau_a')
chi2sigma2 = su.structToArray(data2,'fitdau_asigma')*fac/(a**2*e1*e2)*e2


print(np.where(chi2sigma1 == min(chi2sigma1))[0])


chi2val = su.structToArray(data2,'dau_chi2val')

met = su.structToArray(data2,'met_et')


pullFit1 = (e1fit-e1gen)/e1gen
pullOriginal1 = (e1 - e1gen)/e1gen

pullFit2 = (e2fit-e2gen)/e2gen
pullOriginal2 = (e2 - e2gen)/e2gen

realpullFit1 = (e1fit-e1gen)/chi2sigma1
realpullOriginal1 = (e1 - e1gen)/(0.1*e1)

realpullFit2 = (e2fit-e2gen)/chi2sigma2
realpullOriginal2 = (e2 - e2gen)/(0.1*e2)


print(len(np.where(chi2sigma1 == 0)[0])/len(chi2sigma1)*100,'%')
print(np.where(chi2sigma1 == 0)[0])


plotting.plotHistCompare(PATHTOPLOTSTAU,r"$\frac{\sigma_{\tau_1}}{E_{\tau_1}}$",r"$\sigma_{\tau_1}$-Distribution Normalized", [0,0.8], 100,(np.array([0.1]),r'assumed for measured'),(chi2sigma1/e1fit, r"fitted" ),ylim=[0,10] ,alttitle='Tau_sigmaChi2DistributionNormed',pdf=True)
plotting.plotHistCompare(PATHTOPLOTSTAU,r"$\frac{\sigma_{\tau_2}}{E_{\tau_2}}$",r"$\sigma_{\tau_2}$-Distribution Normalized", [0,0.8], 100,(np.array([0.1]),r'assumed for measured'),(chi2sigma2/e2fit, r"fitted" ),ylim=[0,10] ,alttitle='Tau_sigmaChi2DistributionNormed2',pdf=True)


# plotting.plotHist(chi2sigma,PATHTOPLOTSTAU,r"$\sigma_{\chi^2}$ in GeV",r"$\sigma_{\chi^2}$-Distribution", [-1.25,100], 100,ylim=[0.001,0.1] ,alttitle='Tau_sigmaChi2Distribution',yscale='log')

def standardNormalDist(x):
    return np.exp(-x**2/2)/np.sqrt(np.pi*2)

plotting.plotHistCompare(PATHTOPLOTSTAU, r"$\frac{E_{\tau_1}}{E_{\tau_1}^{g}}-1$ ", r"Deviation from Gen in $E_{\tau_1}$", [-1,1.5], 100, (pullOriginal1, r"measured"),(pullFit1,r"fitted"),ylim=[0,2.25], alttitle='TauFitVsMeasDeviation',pdf=True)
plotting.plotHistCompare(PATHTOPLOTSTAU, r"$\frac{E_{\tau_2}}{E_{\tau_2}^{g}}-1$ ", r"Deviation from Gen in $E_{\tau_2}$", [-1,1.5], 100, (pullOriginal2, r"measured"),(pullFit2,r"fitted"),ylim=[0,2.25], alttitle='TauFitVsMeasDeviation2',pdf=True)

plotting.plotHistCompare(PATHTOPLOTSTAU, r"$\frac{E_{\tau_1}-E_{\tau_1}^g}{\sigma_{\tau 1}}$ ", r"Pulltest for $E_{\tau_1}$", [-3,3], 100, (realpullOriginal1, r"measured"),(realpullFit1,r"fitted"),comparefunctions=[(standardNormalDist,r'standard normal distribution')],ylim=[0,0.8], alttitle='TauPullFitVsMeas',pdf=True)
plotting.plotHistCompare(PATHTOPLOTSTAU, r"$\frac{E_{\tau_2}-E_{\tau_2}^g}{\sigma_{\tau 2}}$ ", r"Pulltest for $E_{\tau_2}$", [-3,3], 100, (realpullOriginal2, r"measured"),(realpullFit2,r"fitted"),comparefunctions=[(standardNormalDist,r'standard normal distribution')],ylim=[0,0.8], alttitle='TauPullFitVsMeas2',pdf=True)



# plotting.plotHistCompare(PATHTOPLOTSTAU, r"$\frac{E_{\tau 1}^{i}}{E_{\tau 1}^{g}}-1$ ", r"Deviation from Gen in $E_{\tau 1}$", [-1,1.5], 100, (np.take(pullOriginal1,edgeindex) , r"measured Value "),(np.take(pullFit1,edgeindex) ,r"fitted Value"),ylim=[0,2], alttitle='TauFitVsMeasDeveation_noEdge')
# plotting.plotHistCompare(PATHTOPLOTSTAU, r"$\frac{E_{\tau 1}^{i}-E_{\tau1}^g}{\sigma_{\tau 1}}$ ", r"Pulltest for $E_{\tau 1}$", [-2.5,2.5], 100, (np.take(realpullOriginal1,edgeindex) , r"measured Value "),(np.take(realpullFit1,edgeindex) ,r"fitted Value"),ylim=[0,1.5], alttitle='TauPullFitVsMeas_noEdge')

def chi2dist(x):
    dof = 1
    return 1/(2**(dof/2)*gamma(dof/2))*x**(dof/2-1)*np.exp(-x/2)

# plotting.plotHist(chi2val,PATHTOPLOTSTAU,r"$\chi^2$-Value",r"$\chi^2$-Value Distribution",xlim=[0,8], ylim =[0.005,0.5], yscale='log',bins=80, alttitle='Tau_Chi2ValueDistribution')
plotting.plotHistCompare(PATHTOPLOTSTAU,r"$\chi^2$",r"$\chi^2$-Value Distribution",[0,10],80, (chi2val,r"$\chi^2$-Value"), comparefunctions=[(chi2dist,r"theoretical distribution")], ylim =[0.0,1],  alttitle='Tau_Chi2ValueDistributionNonLog',pdf=True)
# plotting.plotHist(np.take(chi2val,edgeindex),PATHTOPLOTSTAU,r"$\chi^2$-Value",r"$\chi^2$-Value Distribution",xlim=[0,8], ylim =[0.0,0.5], bins=80, alttitle='Tau_Chi2ValueDistributionNonLog_NoEdge')


# v1m = (tau1eta,tau1phi)
# v1g = (tau1etagen,tau1phigen)
# v2m = (tau2eta,tau2phi)
# drmg = du.deltaR(v1m,v1g)
# dr12 = du.deltaR(v1m,v2m)

# deta12 = abs(tau1eta-tau2eta)
# dphi12 = np.minimum(abs(tau1phi-tau2phi),abs(2*np.pi - tau1phi-tau2phi))




# maxeta = np.where(np.greater(np.absolute(tau1eta),np.absolute(tau2eta)), tau1eta,tau2eta)


# indexPullFit1greater0d5 = np.where(pullFit1>0.5)[0]
# indexPullFitPeak1 = np.where(np.logical_and(pullFit1>=1,pullFit1<=1.025))[0]
# indexPullFitPeak2 = np.where(np.logical_and(pullFit1>=-0.5,pullFit1<=-0.475))[0]




# dau1phivsetaCut = (np.take(tau1phi,indexPullFit1greater0d5),np.take(tau1eta,indexPullFit1greater0d5),r"$(\phi, \eta) $ for $ \frac{E_{\tau 1}^{f}}{E_{\tau 1}^{g}}-1 > 0.5 $") 
# dau1phivseta = (tau1phi,tau1eta,r"$(\phi, \eta) $")
# plotting.scatter(PATHTOPLOTSTAU, 'Comparison for high Pullvalues', r'$\phi$', r'$\eta$',dau1phivsetaCut, lim=[-np.pi,np.pi], yLim=[-5,5])

# plotting.plotHistCompare(PATHTOPLOTSTAU,r"$\eta$", "Eta for high Pullvalues", [-5,5],100, (np.take(tau1eta,indexPullFit1greater0d5),r"$\eta $ for $ \frac{E_{\tau 1}^{f}}{E_{\tau 1}^{g}}-1 > 0.5 $   "),(tau1eta,r"$\eta$      "), yLabel='Amount', density=False, ylim=[0,1000])
# plotting.plotHistCompare(PATHTOPLOTSTAU,r"$\Delta r(\tau^{m,g}))$", "Delta r tau gm for high Pullvalues", [0,0.1],100, (np.take(drmg,indexPullFit1greater0d5),r"$\Delta r$ for $ \frac{E_{\tau 1}^{f}}{E_{\tau 1}^{g}}-1 > 0.5 $   "),(drmg,r"$\Delta r$      "), yLabel='Amount', density=False, ylim=[0,1000])
# plotting.plotHistCompare(PATHTOPLOTSTAU,r"max $\eta$", "max Eta for high Pullvalues", [-5,5],100, (np.take(maxeta,indexPullFit1greater0d5),r"max $\eta $ for $ \frac{E_{\tau 1}^{f}}{E_{\tau 1}^{g}}-1 > 0.5 $   "),(maxeta,r"max $\eta$      "), yLabel='Amount', density=False, ylim=[0,1000])
# plotting.plotHistCompare(PATHTOPLOTSTAU,r"$\Delta r(\tau_{1,2}))$", "Delta r tau 12 for high Pullvalues", [0,2.5],100, (np.take(dr12,indexPullFit1greater0d5),r"$\Delta r$ for $ \frac{E_{\tau 1}^{f}}{E_{\tau 1}^{g}}-1 > 0.5 $   "),(dr12,r"$\Delta r$      "), yLabel='Amount', density=False, ylim=[0,1000])
# plotting.plotHistCompare(PATHTOPLOTSTAU,r"$\Delta \eta(\tau_{1,2}))$", "Delta eta for high Pullvalues", [0,4],100, (np.take(deta12,indexPullFit1greater0d5),r"$\Delta \eta$ for $ \frac{E_{\tau 1}^{f}}{E_{\tau 1}^{g}}-1 > 0.5 $   "),(deta12,r"$\Delta \eta$      "), yLabel='Amount', density=False, ylim=[0,1000])
# plotting.plotHistCompare(PATHTOPLOTSTAU,r"$\Delta \phi(\tau_{1,2}))$", "Delta phi for high Pullvalues", [0,3.5],100, (np.take(dphi12,indexPullFit1greater0d5),r"$\Delta \phi$ for $ \frac{E_{\tau 1}^{f}}{E_{\tau 1}^{g}}-1 > 0.5 $   "),(dphi12,r"$\Delta \phi$      "), yLabel='Amount', density=False, ylim=[0,1000])
# plotting.plotHistCompare(PATHTOPLOTSTAU,r"missing $E_T$", "met for high Pullvalues", [0,300],100, (np.take(met,indexPullFit1greater0d5),r"missing $E_T$ for $ \frac{E_{\tau 1}^{f}}{E_{\tau 1}^{g}}-1 > 0.5 $   "),(met,r"missing $E_T$      "), yLabel='Amount', density=False, ylim=[0,1000])



# frac = taue1/taue1gen

# plotting.plotHistCompare(PATHTOPLOTSTAU,r"$\frac{E_{\tau1}^{m}}{E_{\tau1}^{g}}$", "Visible fraction for high Pullvalues", [0,2],100, (np.take(frac,indexPullFit1greater0d5),r"Visible Fraction for $ \frac{E_{\tau 1}^{f}}{E_{\tau 1}^{g}}-1 > 0.5 $   "),(frac,r"Visible Fraction     "), yLabel='Amount', density=False, ylim=[0,1000])


# plotting.binCompare(PATHTOPLOTSTAU,np.take(tau1eta,indexPullFit1greater0d5),tau1eta,'test','BinContentComparisonEta',[-5,5],100, label= r"$\log \frac{\mathrm{bin\;content\; of\; } \eta \mathrm{\;for\;}  \frac{E_{\tau 1}^{f}}{E_{\tau 1}^{g}}-1 > 0.5   }{\mathrm{bin\; content\; of\; } \eta}$" )

# plotting.binCompare(PATHTOPLOTSTAU,np.take(frac,indexPullFit1greater0d5),frac,'test','BinContentComparisonVisibleFrac',[0,2],100)















# plotting.plotHist(pullFit2, PATHTOPLOTSTAU,r"$\frac{E_{B2}^{\textit{\small{fit}}}-E_{B2}^{\textit{\small{gen}}}}{E_{B2}^{\textit{\small{gen}}}}$", r"Pull Test 2 Tau E",[-2.5,12.5],200, ylim=[0.8,10000], density=False , yLabel=r"Amount", yscale='log')
# plotting.plotHist(pullOriginal2, PATHTOPLOTSTAU,r"$\frac{E_{B2}^{\textit{\small{meas}}}-E_{B2}^{\textit{\small{gen}}}}{E_{B2}^{\textit{\small{gen}}}}$",r"Pull Test Original Values 2 Tau E",[-2.5,12.5],200, ylim=[0.8,10000], density=False , yLabel=r"Amount", yscale='log')



# indexchi2Smaller0d5 = np.where(chi2val<0.5)[0]
# indexchi2Smaller2d5 = np.where(chi2val<2.5)[0]
# indexchi2Smaller10 = np.where(chi2val<10)[0]

# pullFitComp0d5 = (np.take(pullFit1,indexchi2Smaller0d5),r"$\chi^2 < 0.5$ ")
# pullFitComp2d5 = (np.take(pullFit1,indexchi2Smaller2d5),r"$\chi^2 < 2.5$ ")
# pullFitComp10 = (np.take(pullFit1,indexchi2Smaller10),r"$\chi^2 < 10$ ")


# pullOriginalComp0d5 = (np.take(pullOriginal1,indexchi2Smaller0d5),r"$\chi^2 < 0.5$ ")
# pullOriginalComp2d5 = (np.take(pullOriginal1,indexchi2Smaller2d5),r"$\chi^2 < 2.5$ ")
# pullOriginalComp10 = (np.take(pullOriginal1,indexchi2Smaller10),r"$\chi^2 < 10$ ")

# chi2sigmaComp0d5 = (np.take(chi2sigma,indexchi2Smaller0d5),r"$\chi^2 < 0.5$ ")
# chi2sigmaComp2d5 = (np.take(chi2sigma,indexchi2Smaller2d5),r"$\chi^2 < 2.5$ ")
# chi2sigmaComp10 = (np.take(chi2sigma,indexchi2Smaller10),r"$\chi^2 < 10$ ")

# plotting.plotHist(chi2sigma,PATHTOPLOTSTAU,r"$\sigma_{\chi^2}$",r"$\sigma_{\chi^2}$-Distribution", [0,500], 100, chi2sigmaComp0d5, chi2sigmaComp2d5, chi2sigmaComp10,ylim=[0,400] ,alttitle='Tau_sigmaChi2Distribution', density=False , yLabel=r"Amount")

# plotting.plotHist(pullFit1, PATHTOPLOTSTAU,r"$\frac{E_{B1}^{\textit{\small{fit}}}-E_{B1}^{\textit{\small{gen}}}}{E_{B1}^{\textit{\small{gen}}}}$", r"Pull Test 1 Tau E",[-2.5,12.5],200,pullFitComp0d5, pullFitComp2d5, pullFitComp10, ylim=[0.8,10000], density=False , yLabel=r"Amount", yscale='log')
# plotting.plotHist(pullOriginal1, PATHTOPLOTSTAU,r"$\frac{E_{B1}^{\textit{\small{meas}}}-E_{B1}^{\textit{\small{gen}}}}{E_{B1}^{\textit{\small{gen}}}}$",r"Pull Test Original Values 1 Tau E",[-2.5,12.5],200,pullOriginalComp0d5, pullOriginalComp2d5, pullOriginalComp10, ylim=[0.8,10000], density=False , yLabel=r"Amount", yscale='log')

# plotting.plot1v2hist(pullFit1, be1fit,[0,0],'label', [-2.5,12.5],[0,300],100, r"Pull Value", r"Fit Value", r"Pull vs Fit Tau", PATHTOPLOTSTAU )
# plotting.plot1v2hist(pullFit1, taue1gen,[0,0],'label', [-2.5,12.5],[0,300],100, r"Pull Value", r"Gen Value", r"Pull vs Gen Tau" , PATHTOPLOTSTAU )
# plotting.plot1v2hist(taue1gen, be1fit,[0,0],'label', [0,300],[0,300],100, r"Gen Value", r"Fit Value", r"Gen vs Fit Tau", PATHTOPLOTSTAU, scale='log' )

