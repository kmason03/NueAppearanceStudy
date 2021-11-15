from __future__ import print_function
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.colors as colors
import numpy as np
import seaborn as sns
import math
from math import sqrt,cos,acos,pi,exp,sin,atan2,log
from numpy import arange,mean,asarray,isnan,isinf
import pickle as pkl
from scipy import stats

import csv
import sys
import pandas as pd
import os

# global parameters
mu_cutoff = 0.46149198646840239
Nickdir = '/home/kmason/fullosc/NicksCode/1L1PSelection/1e1pBDT/'

# poisson stat errors
def poisson_errors(k, CL = 0.6827):

    # 1 Sig = 0.6827
    # 2 Sig = 0.9545
    # 3 Sig = 0.9973

    a = 1.0 - CL
    low, high = (stats.chi2.ppf(a/2, 2*k) / 2, stats.chi2.ppf(1-a/2, 2*k + 2) / 2)
    low = np.where(k==0,0,low)
    return k - low, high - k

# get the sys cov matrices
def getSysErrors(nbins,varName,mode,sigcut):
    det_sys_m = np.loadtxt(Nickdir+'CovMatrices/DetVar/3March2021_vA_fullLowE_withPi0Sample_newShowerCalib/7September2021/covMatrix/avgscore/BDTcut0.95/detsys_Enu_1e1p_run13_cutMode0_cov_smooth.csv',delimiter=',')
    rwt_sys_m = np.loadtxt(Nickdir+'CovMatrices/RwghtSys/frac_covar_rewgt_highE.txt')

    rwt_sys_m = np.where(np.isnan(rwt_sys_m),0,rwt_sys_m)
    det_sys_m = np.where(np.isnan(det_sys_m),0,det_sys_m)
    return rwt_sys_m, det_sys_m

# add in fractional detvar for numu backgrounds
def getDetSysTot(det_sys_nue,nue,numu):
    det_sys_numu = np.diag(0.04*np.ones(det_sys_nue.shape[0]))
    det_sys_tot = det_sys_nue + det_sys_numu

    # tot = nue+numu
    # det_sys_tot = (det_sys_nue * np.outer(nue,nue) + det_sys_numu * np.outer(numu,numu))/np.outer(tot,tot)
    return det_sys_tot

# make the BDT cut on the data frames
def MakeBDTcut(idf,sigcut,mode,nBDTs,r2overlay=False,ttc=0.1):

    # Conglemerate BDT scores and weights based on strategy

    bdtweight = np.zeros(idf.shape[0])
    sigprobmax = np.zeros(idf.shape[0])
    sigprobavg = np.zeros(idf.shape[0])
    sigprobmedian = np.zeros(idf.shape[0])
    sigproblist = np.zeros((idf.shape[0],nBDTs))
    notintrain = np.zeros((idf.shape[0],nBDTs),dtype=bool)
    numnottrain = np.zeros(idf.shape[0])
    for b in range(nBDTs):
        sp = idf['sigprob%i'%b]
        tvw = idf['tvweight%i'%b]
        sigprobmax = np.where(np.logical_and(tvw>0,sp>sigprobmax),sp,sigprobmax) # cut on the maximum non-train score in ensemble
        if mode == 'fracweight':
            #bdtweight += np.where(sp>sigcut,tvw/float(nBDTs),0)
            bdtweight += np.where((tvw>ttc) & (sp>sigcut),1.0,0.0)
        sigprobavg += np.where(tvw>ttc,sp,0)
        numnottrain += np.where(tvw>ttc,1,0)
        sigproblist[:,b] = sp
        notintrain[:,b] = tvw > ttc
    sigprobavg /= np.where(numnottrain>0,numnottrain,1)
    for i,(tlist,siglist) in enumerate(zip(notintrain,sigproblist)):
        splist = siglist[tlist]
        if splist.size!=0: sigprobmedian[i] = np.median(splist)
        else: sigprobmedian[i] = 0

    idf['sigprobavg'] = sigprobavg
    idf['sigprobmedian'] = sigprobmedian
    idf['sigprobmax'] = sigprobmax


    if mode == 'avgscore':
        idf['sigprob'] = idf['sigprobavg']
        bdtweight = np.where(sigprobavg>sigcut,1,0)
    elif mode == 'medianscore':
        idf['sigprob'] = idf['sigprobmedian']
        bdtweight = np.where(sigprobmedian>sigcut,1,0)
    elif mode == 'fracweight':
        idf['sigprob'] = idf['sigprobmax']
        bdtweight /= np.where(numnottrain>0,numnottrain,1)

    idf['bdtweight'] = bdtweight

    # Drop duplicates
    idf.sort_values(by=['run','subrun','event','sigprob'],ascending=False,inplace=True)
    if r2overlay:
        idf.drop_duplicates(subset=['run','subrun','event','EnuTrue'],inplace=True)
    else:
        idf.drop_duplicates(subset=['run','subrun','event'],inplace=True)

# save selection to text files to make covar/input to sbnfit on fermi
def SaveTextFile(idf,run,filetag,sigcut,mode,tsavedir):
    nBDTs = int(idf['nBDTs'][0])
    MakeBDTcut(idf,sigcut,mode,nBDTs)
    idf.query('sigprob > @sigcut',inplace=True)
    savestr = 'eventlist_%s_run%i.txt'%(filetag,run)
    idf.to_csv(tsavedir+savestr,index=None,sep=' ')

# get sterile osc probs (3+1)
# includes nue app, nue dis, and numu_dis
def oscweights(df,ue4,um4,m41,filetag):
    weights = []
    m =m41
    for i in range(len(df)):
        label = df["label"].values[i]
        P=1.0
        E=df['EnuTrue'].values[i]*1e-3
        z = df['Zreco'].values[i]*.01 - 5
        L= .47+(z*.001)
        massterm = (sin(1.27*((m*L)/E)))**2
        # disappearence
        if (filetag=="overlay" or filetag=="ncpi0" or filetag=="ccpi0") and ("numu" in label):
            P = 1-4*(um4**2)*(1-um4**2)*massterm
        # nue disappearce
        elif ((filetag=='intrinsics' or filetag=="overlay" or filetag=="ncpi0" or filetag=="ccpi0") and ("nue" in label)):
            P = 1-4*(ue4**2)*(1-ue4**2)*massterm
        # nue appearence
        if (filetag=='fullosc'):
            P = 4*(ue4**2)*(um4**2)*massterm
        weights.append(P)
    return weights

# function to get bin weights for plotting
# inputs are the data frame to weight and the bin list
def GetBinWeights(dfc,bins):
    binweights=np.ones(len(dfc))
    # get the bin widths
    binwidths = []
    for i in range(len(bins)-1):
        binwidths.append(bins[i+1]-bins[i])

    for i in range(len(dfc)):
        enu = dfc['Enu_1e1p'].values[i]
        weight =0.0
        for j in range(len(bins)):
            if enu < bins[j] and weight==0.0 and enu>bins[0]:
                weight = 100.0/float(binwidths[j-1])
        binweights[i]=weight

    return binweights


# now a bunch of helper stats functions
def chi2_cnp(M,mu):
    c2c=0
    if mu>0:
        if M != 0:
            c2c = (M - mu)**2/(3 / (2 / mu + 1 / M))
        else:
            c2c = mu/2#(M - mu)**2/(3 / (2 / mu + 1 / mu))
    return c2c

def chi2_pois(M,mu):
    c2c=0
    if mu > 0:
        if M != 0:
            c2c = 2*(mu - M + M*np.log(M/mu))
        else:
            c2c = 2*mu
    return c2c

def chi2_pears(M,mu):
    c2c=0
    if mu > 0:
        if M != 0:
            c2c = (M-mu)**2 / mu
        else:
            c2c = 2*mu
    return c2c

def chi2_thresh(M,mu,mode='cnp'):
    c2c=0
    if mu > mu_cutoff and mode=='cnp':
        c2c = chi2_cnp(M,mu)
    elif mu > mu_cutoff and mode=='pearson':
        c2c = chi2_pears(M,mu)
    else:
        c2c = chi2_pois(M,mu)
    return c2c

def cov_pois(M,mu):
    cov=0
    if mu > 0:
        if M != 0:
            cov = (M-mu)**2 / (2*(mu - M + M*np.log(M/mu)))
        else:
            cov = mu/2
    return cov

def cov_cnp(M,mu):
    cov=0
    if mu > 0:
        if M != 0:
            cov = (3 / (2 / mu + 1 / M))
        else:
            cov = mu/2
    return cov

def cov_pears(M,mu):
    cov=0
    if mu > 0:
        cov = mu
    return cov

def cov_thresh(M,mu,mode='cnp'):
    cov=0
    if mu>mu_cutoff and mode=='cnp':
        cov=cov_cnp(M,mu)
    elif mu>mu_cutoff and mode=='pearson':
        cov=cov_pears(M,mu)
    else:
        cov=cov_pois(M,mu)
    return cov

def getnormalizationunc(det_sys_m,rwt_sys_m,p):
    msys = det_sys_m+rwt_sys_m
    f_unc = (sum(sum(msys)))/(sum(p)**2)
    print("func",f_unc,(sum(p)),(sum(sum(msys))))
    return f_unc

def fmt(x):
    s = f"{x*100:.0f}"
    return rf"{s} \%" if plt.rcParams["text.usetex"] else f"{s} %"

# main function
def main():
    # First define all parameters
    #Binary Settings
    sys     = False
    usedata = False
    shapeonly = False
    makeCutData = False
    ntestbins = 5

    #which of Nick's selections do we use?
    finalPred = False
    fset = 12 # 0 = open data, 1-7 = fake datasets, 8 = High E far sideband, 9 = Low BDT far sideband, 10 = Full near sideband
    cutMode = 0 # 0 = Final Selection, 1 = Kin Cut Sample, 2 = High E, 3 = Low BDT, 4 =
    cutmodedir = {0:'FinalSelection',1:'KinCut',2:'HighE',3:'LowBDT'}[cutMode]
    vtxCut    = 5#np.inf

    # data POT values
    C1_POT = 1.558e+20 + 1.129e+17 + 1.869e+19
    D2_POT = 1.63e+20 + 2.964e+19 + 1.239e+19
    E1_POT = 5.923e+19
    F1_POT = 4.3e+19
    G1_POT = 1.701e+20 + 2.97e+19 + 1.524e+17
    DAT_POT1 = C1_POT
    DAT_POT2 = E1_POT+D2_POT
    DAT_POT3 = F1_POT+G1_POT
    DATA_POT_TOT = DAT_POT1 + DAT_POT2 + DAT_POT3
    print("total POT", DAT_POT1+DAT_POT2+DAT_POT3)

    # set up data frames
    tag='23Aug2021_vA_fullLowE_withPi0Sample_newShowerCalib'
    filetaglist = {1:['ext','overlay','intrinsics','ncpi0','ccpi0','fullosc'],
                   2:['ext','overlay','intrinsics','fullosc'],
                   3:['ext','overlay','intrinsics','ncpi0','ccpi0','fullosc']}
    df = {1:{},2:{},3:{}}
    psavedir = 'Parquets/'+tag+'/'
    for r in [1,2,3]:
        for filetag in filetaglist[r]:
            pstring = 'SelectionMode%i_%s_fset%i_run%i.parquet.gzip'%(cutMode,filetag,fset,r)
            try:
                df[r][filetag] = pd.read_parquet(psavedir+pstring)
            except:
                pass
    # use run 3 ext for run 2
    df[2]['ext'] = df[3]['ext'].copy()
    print("Loaded data frames")

    #BDT Settings
    bdtmode = 'avgscore'
    nBDTs = 20
    bdtpower = 0
    bdtcutRange = (0.95,1.0)

    runs_to_plot = [[1,2,3]]

    POTdict = {1:{ft:DAT_POT1 for ft in filetaglist[1]},
               2:{ft:DAT_POT2 for ft in filetaglist[2]},
               3:{ft:DAT_POT3 for ft in filetaglist[3]}}

    # make lists of sin22theta, m41, chi, nullspectrum
    ue4_list = np.logspace(.1, 4, ntestbins, endpoint=True)
    ue4_list= (ue4_list/2.0)*.0001
    m41_list = np.logspace(.1, 4, ntestbins, endpoint=True)
    m41_list= m41_list*.01

    # define my test binnings
    anabins = []
    anabins_numbins = []
    anabins_name = []

    anabins_name.append('LEE+400+800')
    anabins.append([200,300,400,500,600,700,800,900,1000,1100,1200,1600,2400])
    # anabins_name.append('LEE')
    # anabins.append([200,300,400,500,600,700,800,900,1000,1100,1200])
    # anabins_name.append('all100')
    # anabins.append([200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400])
    # anabins_name.append('all200')
    # anabins.append([200,400,600,800,1000,1200,1400,1600,1800,2000,2200,2400])
    # anabins_name.append('all400')
    # anabins.append([200,600,1000,1400,1800,2200,2400])
    # anabins_name.append('LEE+1200')
    # anabins.append([200,300,400,500,600,700,800,900,1000,1100,1200,2400])
    # anabins_name.append('LEE+3x400')
    # anabins.append([200,300,400,500,600,700,800,900,1000,1100,1200,1600,2000,2400])
    # anabins_name.append('LEE+6x200')
    # anabins.append([200,300,400,500,600,700,800,900,1000,1100,1200,1400,1600,1800,2000,2200,2400])
    # anabins_name.append('LEE+2x600')
    # anabins.append([200,300,400,500,600,700,800,900,1000,1100,1200,1800,2400])

    for ana in anabins:
        anabins_numbins.append(len(ana))

    # loop over each binnings
    num_good_bins90 = []
    num_good_bins99 = []
    for idx in np.arange(len(anabins)):
        bins = anabins[idx]
        name = anabins_name[idx]
        chi2_list = np.zeros((ntestbins, ntestbins))
        bestm =-999
        bests = -999
        bestchi = 100000000

        # now loop to get oscillated spectra for each m,sin
        for m in np.arange(ntestbins):
            m41=m41_list[m]
            for s in np.arange(ntestbins):
                um4 =.135
                ue4 = ue4_list[s]
                # print statement to track progress
                print("m41: ",m41, " ue4: ", ue4, 'bin:', idx)
                # setup bins and dfs
                varName = 'Enu_1e1p'
                numbins = len(bins)-1
                low   = bins[0]
                high  = bins[-1]
                dflist = []
                dftot = pd.DataFrame()
                dftot_e = pd.DataFrame()
                dftot_mu = pd.DataFrame()
                FullPOT = 0
                POT1 = 0
                POT23 = 0

                # loop over all three runs
                for r in runs_to_plot[0]:

                    FullPOT += POTdict[r]['overlay']
                    if r == 1: POT1 += POTdict[r]['overlay']
                    elif r in [2,3]: POT23 += POTdict[r]['overlay']

                    #loop to make cuts and set event weights
                    for filetag in df[r].keys():
                        dfc = df[r][filetag].copy()
                        # make sure the file cuts are right and drop duplicates
                        dfc['sigprob'] = dfc['BDTscore_1e1p']
                        dfc['bdtweight'] = np.where(np.array(dfc['sigprob'])>.95,1,0)
                        dfc.sort_values(by=['run','subrun','event','sigprob'],ascending=False,inplace=True)
                        if r==2 and filetag=='overlay':
                            dfc.drop_duplicates(subset=['run','subrun','event','EnuTrue'],inplace=True)
                        else:
                            dfc.drop_duplicates(subset=['run','subrun','event'],inplace=True)

                        # add labels based on event type
                        numu_l = np.core.defchararray.find(np.array(dfc['label'],dtype=np.string_),'m')!=-1
                        labels = np.where(np.logical_and(numu_l,dfc['scedr']>vtxCut),'offvtx',dfc['label'])
                        dfc['label'] = labels

                        # get weights from oscillation probabilities
                        oscprob=oscweights(dfc,ue4,um4,m41,filetag)
                        dfc['oscprob']=oscprob
                        weights = dfc['GenieWeight'] * dfc['oscprob'] * dfc['POTweight']**(-1)
                        weights*= dfc['bdtweight'] * dfc['sigprob']**bdtpower
                        weights*=dfc['sigprob']>0.95
                        weights*=dfc['sigprob']<1.0
                        weights*=dfc['Enu_1e1p']<high
                        weights*=dfc['Enu_1e1p']>low
                        # add in weights for different binning
                        binweights = GetBinWeights(dfc,bins)
                        weights*=binweights
                        dfc['weights'] = weights*POTdict[r][filetag]

                        #now get null weights
                        nullweights = dfc['GenieWeight'] *  dfc['POTweight']**(-1)
                        nullweights*= dfc['bdtweight'] * dfc['sigprob']**bdtpower
                        nullweights*=dfc['sigprob']>0.95
                        nullweights*=dfc['sigprob']<1.0
                        nullweights*=dfc['Enu_1e1p']<high
                        nullweights*=dfc['Enu_1e1p']>low
                        # add in weights for different binning
                        binweights = GetBinWeights(dfc,bins)
                        nullweights*=binweights
                        if filetag=='fullosc':
                            dfc['nullweights'] = nullweights*0.0
                        else:
                            dfc['nullweights'] = nullweights*POTdict[r][filetag]

                        dftot = pd.concat((dftot,dfc))
                        if filetag=='intrinsics':
                            dftot_e= pd.concat((dftot_e,dfc))
                        else:
                            dftot_mu = pd.concat((dftot_mu,dfc))
                        dflist.append(dfc)
                    # end of loop over files types
                # end of loop over runs


                # get the histograms
                oschist= plt.hist(dftot['Enu_1e1p'],bins=bins,range=(low,high),weights=dftot['weights'])
                nullhist= plt.hist(dftot['Enu_1e1p'],bins=bins,range=(low,high),weights=dftot['nullweights'])
                nullhist_e= plt.hist(dftot_e['Enu_1e1p'],bins=bins,range=(low,high),weights=dftot_e['nullweights'])
                nullhist_mu= plt.hist(dftot_mu['Enu_1e1p'],bins=bins,range=(low,high),weights=dftot_mu['nullweights'])

                xbins  = [(bins[i]+bins[i+1])/2.0 for i in range(numbins)]
                xbin_edges = bins
                pred = np.array(nullhist[0])
                pred_e = np.array(nullhist_e[0])
                pred_mu = np.array(nullhist_mu[0])
                obs = np.array(oschist[0])
                valerrs = poisson_errors(obs)
                binwid = [50,50,50,50,50,50,50,50,50,50,200,400]

                #mcstat error (very tiny)
                tothist= plt.hist(dftot.query('nullweights > 0')['Enu_1e1p'],bins=bins,range=(low,high))
                stkerr = [1.0/sqrt(float(i)) for i in tothist[0]]

                # start of cov matrix and chi2 calculations
                cov = np.zeros((numbins,numbins))

                for j in range(numbins):
                    # add cnp errors to cov matrix
                    cov[j][j] = cov_cnp(obs[j],pred[j]) + stkerr[j]**2


                mask1D = np.where(pred==0,False,True)
                mask2D = np.outer(mask1D,mask1D)#np.where(cov==0,False,True)

                Del = (obs - pred)[mask1D]
                cov = cov[mask2D].reshape((len(Del),len(Del)))

                # calculate chi2 and analytic p-value
                chi2 = np.matmul(np.matmul(Del,np.linalg.inv(cov)),Del)
                chi2_list[s,m]=(chi2)
                print(chi2)
                if(chi2<bestchi):
                    bestchi=chi2
                    bestm=m41_list[m]
                    bests=sin22theta_list[s]

                # end of loop over runs
            #end of loop over s
        # end of loop over m
        # get contour levels 0 = 90% CL, 1 = 99% CL, 2 = other
        # deltachi < 4.6 for 90%, <9.2 for 99% : https://people.richland.edu/james/lecture/m170/tbl-chi.html
        # confidence_arr = np.zeros((ntestbins,ntestbins))

        good90=0
        good99=0
        confidence_arr = np.zeros((ntestbins,ntestbins))
        for s in range(len(chi2_list)):
            for m in range(len(chi2_list[s])):
                    delta = chi2_list[s][m]
                    if delta > 4.6:
                        good90+=1
                    if delta > 9.2:
                        good99+=1
                    if delta < 4.6:
                        confidence_arr[s][m] = .9
                    elif delta < 9.2:
                        confidence_arr[s][m] = .99
                    else:
                        confidence_arr[s][m] = 1
        num_good_bins90.append(good90)
        num_good_bins99.append(good99)
        # make CL plots
        y,x = np.meshgrid(m41_list, sin22theta_list)

        #convert intensity (list of lists) to a numpy array for plotting
        intensity = np.array(chi2_list)
        # print(intensity)

        # now just plug the data into pcolormesh

        plt.pcolormesh(x, y, chi2_list,norm=colors.LogNorm(vmin=intensity.min(), vmax=intensity.max()),shading='auto')
        # print(x,y,intensity)
        # plt.pcolormesh(x, y, intensity, vmax=20)
        cbar = plt.colorbar()
        cbar.set_label(r'$\Delta \chi^2$',rotation=270,fontsize=20)
        #contour plots
        contours = plt.contour(x, y, confidence_arr,[.9,.99], colors='white',fontsize=20);

        label1 = plt.clabel(contours, contours.levels, inline=True, fmt=fmt, fontsize=10,rightside_up=False)

        plt.xlabel(r"$U_{e4}$)",fontsize=30)
        plt.ylabel(r"$\Delta m^2_{41}$",fontsize=30)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlim([0,.5])
        plt.ylim([0,100])
        plt.savefig('Markovtest_'+name+'.png')
        plt.close()

    # end of loop over binnings
    print("num in 90: ",num_good_bins90)
    print("num in 99: ",num_good_bins99)

    print()





# call main function
if __name__ == "__main__":
    main()
