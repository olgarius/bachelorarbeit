import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as mcolor
import mplhep as hep

from collections.abc import Iterable

plt.style.use('/afs/desy.de/user/l/lukastim/code/bachelorarbeit/plot.mplstyle')


def plot2d(xkey,ykey,events,path,xlim=[0,1000],ylim=[0,1000],bins=100):
    
    fig = plt.figure()
    
    x = np.array([k[0] for k in events[[xkey]]])
    y = np.array([k[0] for k in events[[ykey]]])

    h,xb,yb = np.histogram2d(x,y,bins=bins)


    
    hep.hist2dplot(h,xb,yb,norm=mcolor.LogNorm())
 

    plt.xlabel(xkey)
    plt.ylabel(ykey)
    plt.xlim(xlim)
    plt.ylim(ylim)  
    
    plt.savefig(path + xkey + '_' + ykey + '.png')


def plot2d2(x, y, path, xLabel, yLabel ,xlim=[0,1000] ,ylim=[0,1000] ,bins=100):
    
    fig = plt.figure()


    # h,xb,yb = np.histogram2d(x,y,bins=bins)
    h,xb,yb = np.histogram2d(x,y,bins=bins,range=[xlim,ylim])



    
    hep.hist2dplot(h,xb,yb,norm=mcolor.LogNorm())
 
    
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)


    plt.xlim(xlim)
    plt.ylim(ylim)  
    
    plt.savefig(path + xLabel + '_' + yLabel + '.png') 

    return fig

def plot2d3(Z,Coords, path, xLabel,yLabel,levels=6):
    fig = plt.figure()
    # im = plt.imshow(X)
    # fig.colorbar(im)
    


    X,Y = np.meshgrid(Coords[0],Coords[1])


    plt.pcolor(X,Y,Z)
    cs =plt.contour(X,Y,Z,levels,colors = 'k')
    plt.clabel(cs,fontsize=9,inline=True)


    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    



    # plt.xticks(np.linspace(*xlim,numXticks),np.around(np.linspace(Coords[0][0],Coords[0][-1],numXticks),decimals=dec))
    # plt.yticks(np.linspace(*ylim,numYticks),np.around(np.linspace(Coords[1][0],Coords[1][-1],numYticks),decimals=dec))

    # plt.xlim(xlim)
    # plt.ylim(ylim)  
    
    plt.savefig(path + xLabel + '_' + yLabel + '.png') 


def plot1d1(xkey, events, path,xlim=[0,1000], binsize=5, density=True,):
    fig = plt.figure()

    x = np.array([k[0] for k in events[[xkey]]])

    plt.hist(x,bins=np.arange(min(x),max(x)+binsize,binsize), density=density)
    plt.xlabel(xkey)
    plt.xlim(xlim)
    plt.yscale('log')

    plt.savefig(path + xkey + '.png')


def plot1d2(x, path, xLabel,xlim=[0,1000], ylim=[0,1],binsize=5,density = True, scale = 'log'):
    fig = plt.figure()

    
    plt.hist(x,bins=np.arange(min(x),max(x)+binsize,binsize),density=density)
    plt.xlim(xlim)
    plt.ylim(ylim)

    plt.xlabel(xLabel)
    plt.yscale(scale)
    


    plt.savefig(path + xLabel + '.png')

def plot1d3(x1, x2, path, xLabel,xlim=[0,1000], binsize=5,density = True):
    fig = plt.figure()
    plt.hist(x1,bins=np.arange(min(x1),max(x1)+binsize,binsize),density=density)
    plt.hist(x2,bins=np.arange(min(x2),max(x2)+binsize,binsize),density=density,histtype='step')
    plt.xlim(xlim)
    plt.xlabel(xLabel)
    plt.yscale('log')
    
    


    plt.savefig(path + xLabel + '.png')

def plotHist(x, path, xLabel, title, xlim, bins,*xcompare, yLabel= r"Normalized Denisty", ylim=[0,1], density=True, xscale='linear', yscale='linear', alttitle = None):
    fig = plt.figure()
    # a,b = np.histogram(x,bins,density=density, range=xlim)
    mean = np.mean(x)
    std = np.std(x)

    plt.rcParams["text.usetex"] = True
    # plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'



    plt.hist(x,bins,range=tuple(xlim),density=density, label=r"default "+  r"$\mu = $" + str(round(mean,2)) + r" $\sigma = $" + str(round(std,2)))

    if len(xcompare) > 0:
        
    
        r"default "+  r"$\mu = $" + str(round(mean,2)) + r" $\sigma = $" + str(round(std,2))
        for xc in xcompare:
            mean2 = np.mean(xc[0])
            std2 = np.std(xc[0])
            label  = xc[1] +r"$\mu = $" + str(round(mean2,2)) + r" $\sigma = $" + str(round(std2,2))
            plt.hist(xc[0],bins,range=tuple(xlim),density=density,histtype=u'step', label=label)
        plt.legend(fontsize=15, loc = 1)

    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.yscale(yscale)
    plt.xscale(xscale)


    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(title, fontsize = 30)

    textxpos = (xlim[1]+xlim[0])/2
    textypos =0.9*ylim[1]

    if len(xcompare) is 0:
        if yscale is 'log':
            textypos = 0.9* np.log(ylim[1])
        if xscale is 'log':
            textxpos = (np.log(xlim[1])+np.log(xlim[0]))/2

        plt.text(textxpos,textypos,r"$\mu = $" + str(round(mean,2)) + r" $\sigma = $" + str(round(std,2)),horizontalalignment='center')

    plt.tight_layout()


    if alttitle is not None:
        title =alttitle

    plt.savefig(path + title)
    plt.rcParams["text.usetex"] = False

def plot1v2hist(x,y, expectedxy, expectedlabel, xlim, ylim, bins, xlabel, ylabel, title, savepath, suptitle= None, alttitle = None, scale = 'log'):
    fig,ax  = plt.subplots(figsize = (10,10))
    
    plt.rcParams["text.usetex"] = True
    # plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

    h,xb,yb = np.histogram2d(x,y,bins=bins,range=[xlim,ylim])
    if scale is 'log':
        hep.hist2dplot(h,xb,yb,norm=mcolor.LogNorm())
    else:
        hep.hist2dplot(h,xb,yb)
        

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xlim)
    plt.ylim(ylim)  

    plt.scatter(*expectedxy, c='r', marker='x', s=4)
    plt.legend([expectedlabel], fontsize= 'xx-small')
    plt.title(title, fontsize = 30)
    ax.set_box_aspect(1)


    if suptitle is not None:
        plt.text((xlim[0]+xlim[1])/2, 0.9*ylim[1], suptitle  ,fontsize = 25,horizontalalignment='center')

    plt.tight_layout()

    if alttitle is not None:
        title = alttitle

    plt.savefig(savepath + title + '.png')

    plt.rcParams["text.usetex"] = False

def plotHistCompare(path, xLabel, title, xlim, bins,*x, yLabel= r"Normalized Denisty", ylim=[0,1], density=True, xscale='linear', yscale='linear', alttitle = None):
    fig = plt.figure()
  

    plt.rcParams["text.usetex"] = True





    for xc in x:
        mean2 = np.mean(xc[0])
        std2 = np.std(xc[0])
        label  = xc[1] +r"$\mu = $" + str(round(mean2,2)) + r" $\sigma = $" + str(round(std2,2))
        plt.hist(xc[0],bins,range=tuple(xlim),density=density,histtype=u'step', label=label, linewidth=3)
    plt.legend(fontsize=15, loc = 1)

    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.yscale(yscale)
    plt.xscale(xscale)


    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(title, fontsize = 30)

    plt.tight_layout()


    if alttitle is not None:
        title =alttitle

    plt.savefig(path + title)
    plt.rcParams["text.usetex"] = False


def scatter(path, title, xlabel, ylabel, *x, lim=[0,300], alttitle= None, yLim=None,s=2):
    fig,ax  = plt.subplots(figsize = (10,10))
    plt.rcParams["text.usetex"] = True

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    

    for i,xs in enumerate(x):
        if isinstance(xs[0],Iterable):
            plt.scatter(xs[0],xs[1], label = xs[2], s=s, color = colors[i%len(colors)])
        else:
            plt.axvline(xs[0],label = xs[2],color = colors[i%len(colors)])

    plt.legend(fontsize=15, loc = 1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if yLim is None:
        yLim = lim
    plt.xlim(lim)
    plt.ylim(yLim)
    plt.title(title, fontsize=30)
    ax.set_box_aspect(1)

    plt.tight_layout()

    
    if alttitle is not None:
        title =alttitle

    plt.savefig(path + title)



    plt.rcParams["text.usetex"] = False

def binCompare(path, x, y, xLabel, title, xlim, bins, yLabel= r"Log Bin Content Ratio", ylim=[-5,5],  xscale='linear', label = r'$\frac{x}{y}$', alttitle = None):
    fig = plt.figure()
  

    plt.rcParams["text.usetex"] = True



    hist1, edges1  = np.histogram(x,bins,xlim)
    hist2, edges2 = np.histogram(y,bins,xlim)

    hist3l = []
    
    for h1,h2 in zip(hist1,hist2):
        if h1 != 0 and h2 !=0:
            hist3l.append(np.log(h1/h2))
        elif  h1 != 0:    
            hist3l.append(ylim[1])
        elif h2 != 0:
            hist3l.append(ylim[0])
        else:
            hist3l.append(0)
    hist3 = np.array(hist3l)


   

   

    plt.bar(edges1[:-1],hist3,width=(xlim[1]-xlim[0])/bins,label=label)

    
    plt.legend(fontsize=15, loc = 1)

    plt.xlim(xlim)
    plt.ylim(ylim)
  
    plt.xscale(xscale)


    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(title, fontsize = 30)

    plt.tight_layout()


    if alttitle is not None:
        title =alttitle

    plt.savefig(path + title)
    plt.rcParams["text.usetex"] = False