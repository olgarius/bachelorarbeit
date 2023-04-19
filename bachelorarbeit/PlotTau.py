import plotting 
import structutil as su
import datautil as du

import matplotlib.pyplot as plt

import numpy as np



from PATHS import WORKINGDATAPATH
from PATHS import PATHTOPLOTS1D
from PATHS import PATHTOPLOTS2D
from PATHS import PATHTOPLOTSTAU

BINSIZE = 10
XLIM = 2000
YLIM = 10000

#load Data and select events

data2= np.load(WORKINGDATAPATH)




tau1efit = su.structToArray(data2,'fitdau1_e')
tau2efit = su.structToArray(data2,'fitdau2_e')




taue1 = su.structToArray(data2,'dau1_e')
taue2 = su.structToArray(data2,'dau2_e')
taue1gen = su.structToArray(data2,'genLepton1_e')
taue2gen = su.structToArray(data2,'genLepton2_e')
fac = su.structToArray(data2,'tauie_to_tauje')
genfac = su.structToArray(data2,'gentauie_to_gentauje')

tau1phi = su.structToArray(data2,'dau1_phi')
tau1eta = su.structToArray(data2,'dau1_eta')
tau2eta = su.structToArray(data2,'dau2_eta')
tau2phi = su.structToArray(data2,'dau2_phi')



tau1phigen = su.structToArray(data2,'genLepton1_phi')
tau1etagen = su.structToArray(data2,'genLepton1_eta')


chi2sigma = su.structToArray(data2,'fitdau1_esigma')

chi2val = su.structToArray(data2,'dau_chi2val')

met = su.structToArray(data2,'met_et')


pullFit1 = (tau1efit-taue1gen)/taue1gen
pullOriginal1 = (taue1 - taue1gen)/taue1gen

pullFit2 = (tau2efit-taue2gen)/taue2gen
pullOriginal2 = (taue2 - taue2gen)/taue2gen

print(np.mean(chi2sigma))
print(np.mean(chi2sigma/tau1efit))
print(len(np.where(chi2sigma==-1)[0])/len(chi2sigma))

print('large pullvalues',np.where(pullFit1>1))


plotting.plotHist(chi2sigma/tau1efit,PATHTOPLOTSTAU,r"$\frac{\sigma_{\chi^2}}{E_{\tau 1}^f}$",r"$\sigma_{\chi^2}$-Distribution", [-1.25,4], 100,ylim=[0.002,20] ,alttitle='Tau_sigmaChi2DistributionNormed',yscale='log')


plotting.plotHist(chi2sigma,PATHTOPLOTSTAU,r"$\sigma_{\chi^2}$ in GeV",r"$\sigma_{\chi^2}$-Distribution", [-1.25,100], 100,ylim=[0.001,0.1] ,alttitle='Tau_sigmaChi2Distribution',yscale='log')

plotting.plotHistCompare(PATHTOPLOTSTAU, r"Pull Value", r"Fitted vs. Measuerd $E_{\tau 1}$ Values", [-1,1.5], 100, (pullOriginal1, r"$\frac{E_{\tau 1}^{m}}{E_{\tau 1}^{g}}-1$ "),(pullFit1,r"$\frac{E_{\tau 1}^{f}}{E_{\tau 1}^{g}}-1$ "),ylim=[0,2], alttitle='TauPullFitVsMeas')



plotting.plotHist(chi2val,PATHTOPLOTSTAU,r"$\chi^2$-Value",r"$\chi^2$-Value Distribution",xlim=[0,70], ylim =[0.005,0.1], yscale='log',bins=80, alttitle='Tau_Chi2ValueDistribution')
plotting.plotHist(chi2val,PATHTOPLOTSTAU,r"$\chi^2$-Value",r"$\chi^2$-Value Distribution",xlim=[0,0.5], ylim =[0.0,10], bins=200, alttitle='Tau_Chi2ValueDistributionNonLog')

v1m = (tau1eta,tau1phi)
v1g = (tau1etagen,tau1phigen)
v2m = (tau2eta,tau2phi)
drmg = du.deltaR(v1m,v1g)
dr12 = du.deltaR(v1m,v2m)

deta12 = abs(tau1eta-tau2eta)
dphi12 = np.minimum(abs(tau1phi-tau2phi),abs(2*np.pi - tau1phi-tau2phi))




maxeta = np.where(np.greater(np.absolute(tau1eta),np.absolute(tau2eta)), tau1eta,tau2eta)


indexPullFit1greater0d5 = np.where(pullFit1>0.5)[0]
indexPullFitPeak1 = np.where(np.logical_and(pullFit1>=1,pullFit1<=1.025))[0]
indexPullFitPeak2 = np.where(np.logical_and(pullFit1>=-0.5,pullFit1<=-0.475))[0]

print((len(indexPullFitPeak1)+len(indexPullFitPeak2)))
print((len(indexPullFitPeak1)+len(indexPullFitPeak2))/len(pullFit1)*100)
# print(list(zip(np.take(tau1efit,indexPullFitPeak1),np.take(taue1gen,indexPullFitPeak1))))


dau1phivsetaCut = (np.take(tau1phi,indexPullFit1greater0d5),np.take(tau1eta,indexPullFit1greater0d5),r"$(\phi, \eta) $ for $ \frac{E_{\tau 1}^{f}}{E_{\tau 1}^{g}}-1 > 0.5 $") 
dau1phivseta = (tau1phi,tau1eta,r"$(\phi, \eta) $")
plotting.scatter(PATHTOPLOTSTAU, 'Comparison for high Pullvalues', r'$\phi$', r'$\eta$',dau1phivsetaCut, lim=[-np.pi,np.pi], yLim=[-5,5])

plotting.plotHistCompare(PATHTOPLOTSTAU,r"$\eta$", "Eta for high Pullvalues", [-5,5],100, (np.take(tau1eta,indexPullFit1greater0d5),r"$\eta $ for $ \frac{E_{\tau 1}^{f}}{E_{\tau 1}^{g}}-1 > 0.5 $   "),(tau1eta,r"$\eta$      "), yLabel='Amount', density=False, ylim=[0,1000])
plotting.plotHistCompare(PATHTOPLOTSTAU,r"$\Delta r(\tau^{m,g}))$", "Delta r tau gm for high Pullvalues", [0,0.1],100, (np.take(drmg,indexPullFit1greater0d5),r"$\Delta r$ for $ \frac{E_{\tau 1}^{f}}{E_{\tau 1}^{g}}-1 > 0.5 $   "),(drmg,r"$\Delta r$      "), yLabel='Amount', density=False, ylim=[0,1000])
plotting.plotHistCompare(PATHTOPLOTSTAU,r"max $\eta$", "max Eta for high Pullvalues", [-5,5],100, (np.take(maxeta,indexPullFit1greater0d5),r"max $\eta $ for $ \frac{E_{\tau 1}^{f}}{E_{\tau 1}^{g}}-1 > 0.5 $   "),(maxeta,r"max $\eta$      "), yLabel='Amount', density=False, ylim=[0,1000])
plotting.plotHistCompare(PATHTOPLOTSTAU,r"$\Delta r(\tau_{1,2}))$", "Delta r tau 12 for high Pullvalues", [0,2.5],100, (np.take(dr12,indexPullFit1greater0d5),r"$\Delta r$ for $ \frac{E_{\tau 1}^{f}}{E_{\tau 1}^{g}}-1 > 0.5 $   "),(dr12,r"$\Delta r$      "), yLabel='Amount', density=False, ylim=[0,1000])
plotting.plotHistCompare(PATHTOPLOTSTAU,r"$\Delta \eta(\tau_{1,2}))$", "Delta eta for high Pullvalues", [0,4],100, (np.take(deta12,indexPullFit1greater0d5),r"$\Delta \eta$ for $ \frac{E_{\tau 1}^{f}}{E_{\tau 1}^{g}}-1 > 0.5 $   "),(deta12,r"$\Delta \eta$      "), yLabel='Amount', density=False, ylim=[0,1000])
plotting.plotHistCompare(PATHTOPLOTSTAU,r"$\Delta \phi(\tau_{1,2}))$", "Delta phi for high Pullvalues", [0,3.5],100, (np.take(dphi12,indexPullFit1greater0d5),r"$\Delta \phi$ for $ \frac{E_{\tau 1}^{f}}{E_{\tau 1}^{g}}-1 > 0.5 $   "),(dphi12,r"$\Delta \phi$      "), yLabel='Amount', density=False, ylim=[0,1000])
plotting.plotHistCompare(PATHTOPLOTSTAU,r"missing $E_T$", "met for high Pullvalues", [0,300],100, (np.take(met,indexPullFit1greater0d5),r"missing $E_T$ for $ \frac{E_{\tau 1}^{f}}{E_{\tau 1}^{g}}-1 > 0.5 $   "),(met,r"missing $E_T$      "), yLabel='Amount', density=False, ylim=[0,1000])



frac = taue1/taue1gen

plotting.plotHistCompare(PATHTOPLOTSTAU,r"$\frac{E_{\tau1}^{m}}{E_{\tau1}^{g}}$", "Visible fraction for high Pullvalues", [0,2],100, (np.take(frac,indexPullFit1greater0d5),r"Visible Fraction for $ \frac{E_{\tau 1}^{f}}{E_{\tau 1}^{g}}-1 > 0.5 $   "),(frac,r"Visible Fraction     "), yLabel='Amount', density=False, ylim=[0,1000])


plotting.binCompare(PATHTOPLOTSTAU,np.take(tau1eta,indexPullFit1greater0d5),tau1eta,'test','BinContentComparisonEta',[-5,5],100, label= r"$\log \frac{\mathrm{bin\;content\; of\; } \eta \mathrm{\;for\;}  \frac{E_{\tau 1}^{f}}{E_{\tau 1}^{g}}-1 > 0.5   }{\mathrm{bin\; content\; of\; } \eta}$" )

plotting.binCompare(PATHTOPLOTSTAU,np.take(frac,indexPullFit1greater0d5),frac,'test','BinContentComparisonVisibleFrac',[0,2],100)















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

