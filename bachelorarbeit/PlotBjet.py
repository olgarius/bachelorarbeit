import plotting 
import structutil as su

import matplotlib.pyplot as plt

import numpy as np


from PATHS import WORKINGDATAPATH
from PATHS import PATHTOPLOTS1D
from PATHS import PATHTOPLOTS2D
from PATHS import PATHTOPLOTSBJET

BINSIZE = 10
XLIM = 2000
YLIM = 10000

#load Data and select events

data2= np.load(WORKINGDATAPATH)




be1fit = su.structToArray(data2,'fitbjet1_e')
be2fit = su.structToArray(data2,'fitbjet2_e')
be1sigma = su.structToArray(data2, 'bjet1_sigma')


be1 = su.structToArray(data2,'bjet1_e')
be2 = su.structToArray(data2,'bjet2_e')
be1gen = su.structToArray(data2,'genBQuark1_e')
be2gen = su.structToArray(data2,'genBQuark2_e')
fac = su.structToArray(data2,'bie_to_bje')
genfac = su.structToArray(data2,'genbie_to_genbje')




chi2sigma = su.structToArray(data2,'fitbjet1_esigma')

chi2val = su.structToArray(data2,'chi2val')




pullFit1 = (be1fit-be1gen)/be1gen
pullOriginal1 = (be1 - be1gen)/be1gen

pullFit2 = (be2fit-be2gen)/be2gen
pullOriginal2 = (be2 - be2gen)/be2gen

indexchi2Smaller3d84 = np.where(chi2val<3.84)[0]
pF13d84 = np.take(pullFit1,indexchi2Smaller3d84)
pO13d84 = np.take(pullOriginal1,indexchi2Smaller3d84)


plotting.plotHist(chi2sigma/be1fit,PATHTOPLOTSBJET,r"$\frac{\sigma_{\chi^2}}{E_{B1}^f}$",r"$\sigma_{\chi^2}$-Distribution", [0,3], 100, ylim=[0,2] ,alttitle='sigmaChi2DistributionNormed')

plotting.plotHist(chi2sigma,PATHTOPLOTSBJET,r"$\sigma_{\chi^2}$ in GeV",r"$\sigma_{\chi^2}$-Distribution", [0,1000], 100, ylim=[0,0.005] ,alttitle='sigmaChi2Distribution')
plotting.plotHistCompare(PATHTOPLOTSBJET, r"Pull Value", r"Fitted vs. Measuerd $E_{B1}$ Values", [-1,1], 80, (pO13d84 , r"$\frac{E_{B1}^{m}}{E_{B1}^{g}}-1$, $\chi^2 < 3.84$ "),(pF13d84,r"$\frac{E_{B1}^{f}}{E_{B1}^{g}}-1$, $\chi^2 < 3.84$ "),ylim=[0,3], alttitle='BPullFitVsMeas')



# plotting.plotHist(chi2val,PATHTOPLOTSBJET,r"$\chi^2$-Value",r"$\chi^2$-Value Distribution",xlim=[0,4], ylim =[0.01,10], yscale='log',bins=80, alttitle='Chi2ValueDistribution')
plotting.plotHist(chi2val,PATHTOPLOTSBJET,r"$\chi^2$-Value",r"$\chi^2$-Value Distribution",xlim=[0,5], ylim =[0.0,3], bins=200, alttitle='Chi2ValueDistributionNonLog')






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







