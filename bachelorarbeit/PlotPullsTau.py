import plotting 
import structutil as su

import matplotlib.pyplot as plt

import numpy as np


from PATHS import WORKINGDATAPATH
from PATHS import PATHTOPLOTS1D
from PATHS import PATHTOPLOTS2D
from PATHS import PATHTOPLOTSDATVAL

BINSIZE = 10
XLIM = 2000
YLIM = 10000

#load Data and select events

data2= np.load(WORKINGDATAPATH)




be1fit = su.structToArray(data2,'fitdau1_e')
be2fit = su.structToArray(data2,'fitdau2_e')



taue1 = su.structToArray(data2,'dau1_e')
taue2 = su.structToArray(data2,'dau2_e')
taue1gen = su.structToArray(data2,'genLepton1_e')
taue2gen = su.structToArray(data2,'genLepton2_e')
fac = su.structToArray(data2,'tauie_to_tauje')
genfac = su.structToArray(data2,'gentauie_to_gentauje')




chi2sigma = su.structToArray(data2,'fitdau1_esigma')

chi2val = su.structToArray(data2,'dau_chi2val')
indexchi2Smaller0d5 = np.where(chi2val<0.5)[0]
indexchi2Smaller2d5 = np.where(chi2val<2.5)[0]
indexchi2Smaller10 = np.where(chi2val<10)[0]



pullFit1 = (be1fit-taue1gen)/taue1gen
pullOriginal1 = (taue1 - taue1gen)/taue1gen

pullFit2 = (be2fit-taue2gen)/taue2gen
pullOriginal2 = (taue2 - taue2gen)/taue2gen

pullFitComp0d5 = (np.take(pullFit1,indexchi2Smaller0d5),r"$\chi^2 < 0.5$ ")
pullFitComp2d5 = (np.take(pullFit1,indexchi2Smaller2d5),r"$\chi^2 < 2.5$ ")
pullFitComp10 = (np.take(pullFit1,indexchi2Smaller10),r"$\chi^2 < 10$ ")


pullOriginalComp0d5 = (np.take(pullOriginal1,indexchi2Smaller0d5),r"$\chi^2 < 0.5$ ")
pullOriginalComp2d5 = (np.take(pullOriginal1,indexchi2Smaller2d5),r"$\chi^2 < 2.5$ ")
pullOriginalComp10 = (np.take(pullOriginal1,indexchi2Smaller10),r"$\chi^2 < 10$ ")

chi2sigmaComp0d5 = (np.take(chi2sigma,indexchi2Smaller0d5),r"$\chi^2 < 0.5$ ")
chi2sigmaComp2d5 = (np.take(chi2sigma,indexchi2Smaller2d5),r"$\chi^2 < 2.5$ ")
chi2sigmaComp10 = (np.take(chi2sigma,indexchi2Smaller10),r"$\chi^2 < 10$ ")



plotting.plotHist(chi2sigma,PATHTOPLOTSDATVAL,r"$\sigma_{\chi^2}$",r"$\sigma_{\chi^2}$-Distribution", [0,500], 100, chi2sigmaComp0d5, chi2sigmaComp2d5, chi2sigmaComp10,ylim=[0,400] ,alttitle='Tau_sigmaChi2Distribution', density=False , yLabel=r"Amount")


plotting.plotHist(pullFit1, PATHTOPLOTSDATVAL,r"$\frac{E_{B1}^{\textit{\small{fit}}}-E_{B1}^{\textit{\small{gen}}}}{E_{B1}^{\textit{\small{gen}}}}$", r"Pull Test 1 Tau E",[-2.5,12.5],200,pullFitComp0d5, pullFitComp2d5, pullFitComp10, ylim=[0.8,10000], density=False , yLabel=r"Amount", yscale='log')
plotting.plotHist(pullOriginal1, PATHTOPLOTSDATVAL,r"$\frac{E_{B1}^{\textit{\small{meas}}}-E_{B1}^{\textit{\small{gen}}}}{E_{B1}^{\textit{\small{gen}}}}$",r"Pull Test Original Values 1 Tau E",[-2.5,12.5],200,pullOriginalComp0d5, pullOriginalComp2d5, pullOriginalComp10, ylim=[0.8,10000], density=False , yLabel=r"Amount", yscale='log')

plotting.plotHist(pullFit2, PATHTOPLOTSDATVAL,r"$\frac{E_{B2}^{\textit{\small{fit}}}-E_{B2}^{\textit{\small{gen}}}}{E_{B2}^{\textit{\small{gen}}}}$", r"Pull Test 2 Tau E",[-2.5,12.5],200, ylim=[0.8,10000], density=False , yLabel=r"Amount", yscale='log')
plotting.plotHist(pullOriginal2, PATHTOPLOTSDATVAL,r"$\frac{E_{B2}^{\textit{\small{meas}}}-E_{B2}^{\textit{\small{gen}}}}{E_{B2}^{\textit{\small{gen}}}}$",r"Pull Test Original Values 2 Tau E",[-2.5,12.5],200, ylim=[0.8,10000], density=False , yLabel=r"Amount", yscale='log')


plotting.plotHist(chi2val,PATHTOPLOTSDATVAL,r"$\chi^2$-Value",r"$\chi^2$-Value Distribution",xlim=[0,4], ylim =[0.01,10], yscale='log',bins=80, alttitle='Tau_Chi2ValueDistribution')
plotting.plotHist(chi2val,PATHTOPLOTSDATVAL,r"$\chi^2$-Value",r"$\chi^2$-Value Distribution",xlim=[0,8], ylim =[0.0,0.5], bins=200, alttitle='Tau_Chi2ValueDistributionNonLog')

plotting.plot1v2hist(pullFit1, be1fit,[0,0],'label', [-2.5,12.5],[0,300],100, r"Pull Value", r"Fit Value", r"Pull vs Fit Tau", PATHTOPLOTSDATVAL )
plotting.plot1v2hist(pullFit1, taue1gen,[0,0],'label', [-2.5,12.5],[0,300],100, r"Pull Value", r"Gen Value", r"Pull vs Gen Tau" , PATHTOPLOTSDATVAL )
plotting.plot1v2hist(taue1gen, be1fit,[0,0],'label', [0,300],[0,300],100, r"Gen Value", r"Fit Value", r"Gen vs Fit Tau", PATHTOPLOTSDATVAL, scale='log' )







