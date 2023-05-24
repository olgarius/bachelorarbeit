import plotting 
import structutil as su

import matplotlib.pyplot as plt

import numpy as np
from scipy.special import gamma

from PATHS import WORKINGDATAPATH
from PATHS import PATHTOPLOTS1D
from PATHS import PATHTOPLOTS2D
from PATHS import PATHTOPLOTSBJET

BINSIZE = 10
XLIM = 2000
YLIM = 10000

#load Data and select events

data= np.load(WORKINGDATAPATH)

edgevalue = su.structToArray(data,'bfit_valueOnEdge')


edgeindex = np.where(edgevalue<1)[0]
data3 = np.take(data, edgeindex )

data2 = data
chi2val = su.structToArray(data3,'chi2val')

# data2 = np.take(data3,np.where(chi2val < 2)[0])

e1fit = su.structToArray(data2,'fitbjet1_e')
e2fit = su.structToArray(data2,'fitbjet2_e')




e1 = su.structToArray(data2,'bjet1_e')
e2 = su.structToArray(data2,'bjet2_e')
e1gen = su.structToArray(data2,'genBQuark1_e')
e2gen = su.structToArray(data2,'genBQuark2_e')
fac = su.structToArray(data2,'bie_to_bje')
genfac = su.structToArray(data2,'genbie_to_genbje')
sigmameas = su.structToArray(data2,'bjet1_sigma') *e1
sigmameas2 = su.structToArray(data2,'bjet2_sigma') *e2


phi1 = su.structToArray(data2,'bjet1_phi')
eta1 = su.structToArray(data2,'bjet1_eta')
eta2 = su.structToArray(data2,'bjet2_eta')
phi2 = su.structToArray(data2,'bjet2_phi')

# edgevalue = su.structToArray(data2,'bfit_valueOnEdge')

# edgeindex = np.where(edgevalue<1)[0]

phi1gen = su.structToArray(data2,'genBQuark1_phi')
eta1gen = su.structToArray(data2,'genBQuark1_eta')


chi2sigma = su.structToArray(data2,'fitbjet1_esigma')
chi2sigma2 = su.structToArray(data2,'fitbjet1_esigma') * fac/e1fit**2









pullFit1 = (e1fit-e1gen)/e1gen
pullOriginal1 = (e1 - e1gen)/e1gen

realpullFit1 = (e1fit-e1gen)/chi2sigma
realpullOriginal1 = (e1 - e1gen)/(sigmameas)

realpullFit2 = (e2fit-e2gen)/chi2sigma2
realpullOriginal2 = (e2 - e2gen)/(sigmameas2)

pullFit2 = (e2fit-e2gen)/e2gen
pullOriginal2 = (e2 - e2gen)/e2gen


def standardNormalDist(x):
    return np.exp(-x**2/2)/np.sqrt(np.pi*2)

plotting.plotHistCompare(PATHTOPLOTSBJET,r"$\frac{\sigma_{b_1}}{E_{b_1}}$",r"$\sigma_{b_1}$-Distribution Normalized", [0,0.5], 100,(sigmameas/e1,r"measured"),(chi2sigma/e1fit,r"fitted") ,ylim=[0,30], pdf=True, alttitle='b_sigmaChi2DistributionNormed')
plotting.plotHistCompare(PATHTOPLOTSBJET,r"$\frac{\sigma_{b_2}}{E_{b_2}}$",r"$\sigma_{b_2}$-Distribution Normalized", [0,0.5], 100,(sigmameas2/e2,r"measured"),(chi2sigma2/e2fit,r"fitted") ,ylim=[0,30], pdf=True, alttitle='b_sigmaChi2DistributionNormed2')


# plotting.plotHist(chi2sigma,PATHTOPLOTSBJET,r"$\sigma_{\chi^2}$ in GeV",r"$\sigma_{\chi^2}$-Distribution", [-1.25,100], 100,ylim=[0.001,0.1] ,alttitle='Tau_sigmaChi2Distribution',yscale='log')

plotting.plotHistCompare(PATHTOPLOTSBJET, r"$\frac{E_{b_1}}{E_{b_1}^{g}}-1$ ", r"Deviation from Gen in $E_{b_1}$", [-1,1.5], 100, (pullOriginal1, r"measured"),(pullFit1,r"fitted"),ylim=[0,3.25], pdf=True, alttitle='bFitVsMeasDeviation')
plotting.plotHistCompare(PATHTOPLOTSBJET, r"$\frac{E_{b_2}}{E_{b_2}^{g}}-1$ ", r"Deviation from Gen in $E_{b_2}$", [-1,1.5], 100, (pullOriginal2, r"measured"),(pullFit2,r"fitted"),ylim=[0,3.25], pdf=True, alttitle='bFitVsMeasDeviation2')

plotting.plotHistCompare(PATHTOPLOTSBJET, r"$\frac{E_{b_1}-E_{b_1}^g}{\sigma_{b_1}}$ ", r"Pulltest for $E_{b_1}$", [-5,5], 100, (realpullOriginal1, r"measured"),(realpullFit1,r"fitted"),comparefunctions=[(standardNormalDist,r'standard normal distribution')], ylim=[0,0.5], pdf=True, alttitle='bPullFitVsMeas')
plotting.plotHistCompare(PATHTOPLOTSBJET, r"$\frac{E_{b_2}-E_{b_2}^g}{\sigma_{b_2}}$ ", r"Pulltest for $E_{b_2}$", [-5,5], 100, (realpullOriginal2, r"measured"),(realpullFit2,r"fitted"),comparefunctions=[(standardNormalDist,r'standard normal distribution')], ylim=[0,0.5], pdf=True, alttitle='bPullFitVsMeas2')



# plotting.plotHistCompare(PATHTOPLOTSBJET, r"$\frac{E_{\tau 1}^{i}}{E_{\tau 1}^{g}}-1$ ", r"Deviation from Gen in $E_{\tau 1}$", [-1,1.5], 100, (np.take(pullOriginal1,edgeindex) , r"measured Value "),(np.take(pullFit1,edgeindex) ,r"fitted Value"),ylim=[0,2], alttitle='TauFitVsMeasDeveation_noEdge')
# plotting.plotHistCompare(PATHTOPLOTSBJET, r"$\frac{E_{\tau 1}^{i}-E_{\tau1}^g}{\sigma_{\tau 1}}$ ", r"Pulltest for $E_{\tau 1}$", [-2.5,2.5], 100, (np.take(realpullOriginal1,edgeindex) , r"measured Value "),(np.take(realpullFit1,edgeindex) ,r"fitted Value"),ylim=[0,1.5], alttitle='TauPullFitVsMeas_noEdge')


def chi2dist(x):
    dof = 1
    return 1/(2**(dof/2)*gamma(dof/2))*x**(dof/2-1)*np.exp(-x/2)


# plotting.plotHist(chi2val,PATHTOPLOTSBJET,r"$\chi^2$-Value",r"$\chi^2$-Value Distribution",xlim=[0,8], ylim =[0.005,0.5], yscale='log',bins=80, alttitle='Tau_Chi2ValueDistribution')
plotting.plotHistCompare(PATHTOPLOTSBJET,r"$\chi^2$",r"$\chi^2$-Value Distribution",[0,10],80,(chi2val,r"$\chi^2$-Value"), comparefunctions=[(chi2dist,r"theoretical distribution")] ,ylim =[0.0,1], pdf=True,  alttitle='b_Chi2ValueDistributionNonLog')
# plotting.plotHist(np.take(chi2val,edgeindex),PATHTOPLOTSBJET,r"$\chi^2$-Value",r"$\chi^2$-Value Distribution",xlim=[0,8], ylim =[0.0,0.5], bins=80, alttitle='Tau_Chi2ValueDistributionNonLog_NoEdge')






























# data2= np.load(WORKINGDATAPATH)




# be1fit = su.structToArray(data2,'fitbjet1_e')
# be2fit = su.structToArray(data2,'fitbjet2_e')
# be1sigma = su.structToArray(data2, 'bjet1_sigma')


# be1 = su.structToArray(data2,'bjet1_e')
# be2 = su.structToArray(data2,'bjet2_e')
# be1gen = su.structToArray(data2,'genBQuark1_e')
# be2gen = su.structToArray(data2,'genBQuark2_e')
# fac = su.structToArray(data2,'bie_to_bje')
# genfac = su.structToArray(data2,'genbie_to_genbje')




# chi2sigma = su.structToArray(data2,'fitbjet1_esigma')

# chi2val = su.structToArray(data2,'chi2val')




# pullFit1 = (be1fit-be1gen)/be1gen
# pullOriginal1 = (be1 - be1gen)/be1gen

# pullFit2 = (be2fit-be2gen)/be2gen
# pullOriginal2 = (be2 - be2gen)/be2gen

# indexchi2Smaller3d84 = np.where(chi2val<3.84)[0]
# pF13d84 = np.take(pullFit1,indexchi2Smaller3d84)
# pO13d84 = np.take(pullOriginal1,indexchi2Smaller3d84)


# plotting.plotHist(chi2sigma/be1fit,PATHTOPLOTSBJET,r"$\frac{\sigma_{\chi^2}}{E_{B1}^f}$",r"$\sigma_{\chi^2}$-Distribution", [0,3], 100, ylim=[0,2] ,alttitle='sigmaChi2DistributionNormed')

# plotting.plotHist(chi2sigma,PATHTOPLOTSBJET,r"$\sigma_{\chi^2}$ in GeV",r"$\sigma_{\chi^2}$-Distribution", [0,1000], 100, ylim=[0,0.005] ,alttitle='sigmaChi2Distribution')
# plotting.plotHistCompare(PATHTOPLOTSBJET, r"Pull Value", r"Fitted vs. Measuerd $E_{B1}$ Values", [-1,1], 80, (pO13d84 , r"$\frac{E_{B1}^{m}}{E_{B1}^{g}}-1$, $\chi^2 < 3.84$ "),(pF13d84,r"$\frac{E_{B1}^{f}}{E_{B1}^{g}}-1$, $\chi^2 < 3.84$ "),ylim=[0,3], alttitle='BPullFitVsMeas')



# # plotting.plotHist(chi2val,PATHTOPLOTSBJET,r"$\chi^2$-Value",r"$\chi^2$-Value Distribution",xlim=[0,4], ylim =[0.01,10], yscale='log',bins=80, alttitle='Chi2ValueDistribution')
# plotting.plotHist(chi2val,PATHTOPLOTSBJET,r"$\chi^2$-Value",r"$\chi^2$-Value Distribution",xlim=[0,5], ylim =[0.0,3], bins=200, alttitle='Chi2ValueDistributionNonLog')






# plotting.plotHist(pullFit2, PATHTOPLOTSBJET,r"$\frac{E_{B2}^{\textit{\small{fit}}}-E_{B2}^{\textit{\small{gen}}}}{E_{B2}^{\textit{\small{gen}}}}$", r"Pull Test 2",[-2.5,12.5],200, ylim=[0.8,10000], density=False , yLabel=r"Amount", yscale='log')
# plotting.plotHist(pullOriginal2, PATHTOPLOTSBJET,r"$\frac{E_{B2}^{\textit{\small{meas}}}-E_{B2}^{\textit{\small{gen}}}}{E_{B2}^{\textit{\small{gen}}}}$",r"Pull Test Original Values 2",[-2.5,12.5],200, ylim=[0.8,10000], density=False , yLabel=r"Amount", yscale='log')

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

# plotting.plotHist(chi2sigma,PATHTOPLOTSBJET,r"$\sigma_{\chi^2}$",r"$\sigma_{\chi^2}$-Distribution", [0,500], 100, chi2sigmaComp0d5, chi2sigmaComp2d5, chi2sigmaComp10,ylim=[0,400] ,alttitle='sigmaChi2Distribution', density=False , yLabel=r"Amount")


# plotting.plotHist(pullFit1, PATHTOPLOTSBJET,r"$\frac{E_{B1}^{\textit{\small{fit}}}-E_{B1}^{\textit{\small{gen}}}}{E_{B1}^{\textit{\small{gen}}}}$", r"Pull Test 1",[-2.5,12.5],200,pullFitComp0d5, pullFitComp2d5, pullFitComp10, ylim=[0.8,10000], density=False , yLabel=r"Amount", yscale='log')
# plotting.plotHist(pullOriginal1, PATHTOPLOTSBJET,r"$\frac{E_{B1}^{\textit{\small{meas}}}-E_{B1}^{\textit{\small{gen}}}}{E_{B1}^{\textit{\small{gen}}}}$",r"Pull Test Original Values 1",[-2.5,12.5],200,pullOriginalComp0d5, pullOriginalComp2d5, pullOriginalComp10, ylim=[0.8,10000], density=False , yLabel=r"Amount", yscale='log')


# plotting.plot1v2hist(pullFit1, be1fit,[0,0],'label', [-2.5,12.5],[0,300],100, r"Pull Value", r"Fit Value", r"Pull vs Fit", PATHTOPLOTSBJET )
# plotting.plot1v2hist(pullFit1, be1gen,[0,0],'label', [-2.5,12.5],[0,300],100, r"Pull Value", r"Gen Value", r"Pull vs Gen", PATHTOPLOTSBJET )
# plotting.plot1v2hist(be1gen, be1fit,[0,0],'label', [0,300],[0,300],100, r"Gen Value", r"Fit Value", r"Gen vs Fit", PATHTOPLOTSBJET, scale='log' )








