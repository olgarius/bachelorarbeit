import plotting 
import structutil as su

import numpy as np


from PATHS import WORKINGDATAPATH
from PATHS import PATHTOPLOTS1D
from PATHS import PATHTOPLOTS2D
from PATHS import PATHTOPLOTSDATVAL

#load Data and select events

data2= np.load(WORKINGDATAPATH)

be1fit = su.structToArray(data2,'fitbjet1_e')
be2fit = su.structToArray(data2,'fitbjet2_e')


be1 = su.structToArray(data2,'bjet1_e')
be2 = su.structToArray(data2,'bjet2_e')
be1gen = su.structToArray(data2,'genBQuark1_e')
be2gen = su.structToArray(data2,'genBQuark2_e')
fac = su.structToArray(data2,'bie_to_bje')
genfac = su.structToArray(data2,'genbie_to_genbje')


chi2val = su.structToArray(data2,'chi2val')
indexchi2Smaller0d5 = np.where(chi2val<0.5)[0]
indexchi2Smaller2d5 = np.where(chi2val<2.5)[0]
indexchi2Smaller10 = np.where(chi2val<10)[0]


sfac = np.sqrt(fac)
sgenfac = np.sqrt(genfac)


be1fitdfac = be1fit/sfac
be2fitdfac = be2fit/sfac

be1dfac = be1/sfac
be2dfac = be2/sfac
be1gendfac = be1gen/sgenfac
be2gendfac = be2gen/sgenfac



be1dfacComp0d5 = np.take(be1dfac,indexchi2Smaller0d5)
be1dfacComp2d5 = np.take(be1dfac,indexchi2Smaller2d5)
be1dfacComp10 = np.take(be1dfac,indexchi2Smaller10)

be2dfacComp0d5 = np.take(be2dfac,indexchi2Smaller0d5)
be2dfacComp2d5 = np.take(be2dfac,indexchi2Smaller2d5)
be2dfacComp10 = np.take(be2dfac,indexchi2Smaller10)

be1gendfacComp0d5 = np.take(be1gendfac,indexchi2Smaller0d5)
be1gendfacComp2d5 = np.take(be1gendfac,indexchi2Smaller2d5)
be1gendfacComp10 = np.take(be1gendfac,indexchi2Smaller10)

be2gendfacComp0d5 = np.take(be2gendfac,indexchi2Smaller0d5)
be2gendfacComp2d5 = np.take(be2gendfac,indexchi2Smaller2d5)
be2gendfacComp10 = np.take(be2gendfac,indexchi2Smaller10)


t = np.linspace(1,10,30)

fx = 1/t
fy = 1/fx

fxx = np.append(fy,fx)
fyy = np.append(fx,fy)
f = [fxx,fyy]

plotting.plot1v2hist(be1fitdfac,be2fitdfac, f,r"$y=\frac{1}{x}$",[0,10],[0,10], 100, r"$\frac{E_{B2}^{fit}}{f}$",r"$\frac{E_{B2}^{fit}}{f}$",r" $E_{B2}^{fit}$ vs $ E_{B2}^{fit}$ ", PATHTOPLOTSDATVAL, r"$f = \sqrt{\frac{m_H²}{2 \cdot (1-\cos{(\alpha)})}}$",'SanityCheck')


plotting.plot1v2hist(be1dfac,be2dfac, f,r"$y=\frac{1}{x}$",[0,10],[0,10], 100, r"$\frac{E_{B2}^{meas}}{f}$",r"$\frac{E_{B2}^{meas}}{f}$",r" $E_{B2}^{meas}$ vs $ E_{B2}^{meas}$ ", PATHTOPLOTSDATVAL, r"$f = \sqrt{\frac{m_H²}{2 \cdot (1-\cos{(alpha)})}}$",'BE1vsBE2')
plotting.plot1v2hist(be1dfacComp0d5,be2dfacComp0d5, f,r"$y=\frac{1}{x}$",[0,10],[0,10], 100,r"$\frac{E_{B2}^{meas}}{f}$",r"$\frac{E_{B2}^{meas}}{f}$",r" $E_{B2}^{meas}$ vs $ E_{B2}^{meas}$ for $\chi^2 < 0.5$", PATHTOPLOTSDATVAL, r"$f = \sqrt{\frac{m_H²}{2 \cdot (1-\cos{(\alpha}))}}$",'BE1vsBE2Chi2smaller0d5')
plotting.plot1v2hist(be1dfacComp2d5,be2dfacComp2d5, f,r"$y=\frac{1}{x}$",[0,10],[0,10], 100, r"$\frac{E_{B2}^{meas}}{f}$",r"$\frac{E_{B2}^{meas}}{f}$",r" $E_{B2}^{meas}$ vs $ E_{B2}^{meas}$ for $\chi^2 < 2.5$", PATHTOPLOTSDATVAL, r"$f = \sqrt{\frac{m_H²}{2 \cdot (1-\cos{(\alpha}))}}$",'BE1vsBE2Chi2smaller2d5')
plotting.plot1v2hist(be1dfacComp10,be2dfacComp10, f,r"$y=\frac{1}{x}$",[0,10],[0,10], 100,  r"$\frac{E_{B2}^{meas}}{f}$",r"$\frac{E_{B2}^{meas}}{f}$",r" $E_{B2}^{meas}$ vs $ E_{B2}^{meas}$ for $\chi^2 < 10$", PATHTOPLOTSDATVAL, r"$f = \sqrt{\frac{m_H²}{2 \cdot (1-\cos{(\alpha}))}}$",'BE1vsBE2Chi2smaller10')

plotting.plot1v2hist(be1gendfac,be2gendfac, f,r"$y=\frac{1}{x}$",[0,10],[0,10], 100, r"$\frac{E_{B2}^{gen}}{f}$",r"$\frac{E_{B2}^{gen}}{f}$",r" $E_{B2}^{gen}$ vs $ E_{B2}^{gen}$ ", PATHTOPLOTSDATVAL, r" $f = \sqrt{\frac{m_H²}{2 \cdot (1-\cos{(alpha)})}}$",'genBE1vsBE2')
plotting.plot1v2hist(be1gendfacComp0d5,be2gendfacComp0d5, f,r"$y=\frac{1}{x}$",[0,10],[0,10], 100, r"$\frac{E_{B2}^{gen}}{f}$",r"$\frac{E_{B2}^{gen}}{f}$",r" $E_{B2}^{gen}$ vs $ E_{B2}^{gen}$ for $\chi^2 < 0.5$", PATHTOPLOTSDATVAL, r"$f = \sqrt{\frac{m_H²}{2 \cdot (1-\cos{(\alpha}))}}$",'genBE1vsBE2Chi2smaller0d5')
plotting.plot1v2hist(be1gendfacComp2d5,be2gendfacComp2d5, f,r"$y=\frac{1}{x}$",[0,10],[0,10], 100, r"$\frac{E_{B2}^{gen}}{f}$",r"$\frac{E_{B2}^{gen}}{f}$",r" $E_{B2}^{gen}$ vs $ E_{B2}^{gen}$ for $\chi^2 < 2.5$", PATHTOPLOTSDATVAL, r"$f = \sqrt{\frac{m_H²}{2 \cdot (1-\cos{(\alpha}))}}$",'genBE1vsBE2Chi2smaller2d5')
plotting.plot1v2hist(be1gendfacComp10,be2gendfacComp10, f,r"$y=\frac{1}{x}$",[0,10],[0,10], 100, r"$\frac{E_{B2}^{gen}}{f}$",r"$\frac{E_{B2}^{gen}}{f}$",r" $E_{B2}^{gen}$ vs $ E_{B2}^{gen}$ for $\chi^2 < 10$", PATHTOPLOTSDATVAL, r"$f = \sqrt{\frac{m_H²}{2 \cdot (1-\cos{(\alpha}))}}$",'genBE1vsBE2Chi2smaller10')

