from __future__ import print_function
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import math
from math import sqrt,cos,acos,pi,exp,sin,atan2,log
from numpy import arange,mean,asarray,isnan,isinf
import pickle as pkl
from scipy import stats

import csv
import sys
import pandas as pd
import os
import time
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
def getSysErrors():
    sys_dir         = "/home/jmills/workdir/Disappearance/NueAppearanceStudy/Systematics/"
    flux_xsex_sys_m = np.nan_to_num(np.loadtxt(sys_dir+"frac_covar_sel_total_withPi0Weights__nu_energy_reco_19binMod.txt",delimiter=','))
    det_sys_m       = np.nan_to_num(np.loadtxt(sys_dir+"detsys_Enu_1m1p_run13_cov_19binMod.csv",delimiter=','))
    return flux_xsex_sys_m, det_sys_m

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
def oscweights(df,ue4,um4,m41,filetag,numuMode=True,sin22=None):
    weights = []
    m =m41
    for i in range(len(df)):
        label = df["label"].values[i]
        P=1.0
        E=df['EnuTrue'].values[i]*1e-3
        L=None
        if df['nu_L_true'].values[i] != 1.0:
            L = df['nu_L_true'].values[i]*0.01*0.001
        else:
            z = df['Zreco'].values[i]*.01 - 5
            L = .47+(z*.001)
        massterm = (sin(1.27*((m*L)/E)))**2
        # numu disappearence
        if (filetag=="overlay" or filetag=="ncpi0" or filetag=="ccpi0") and ("numu" in label):
            if not numuMode:
                P = 1-4*(um4**2)*(1-um4**2)*massterm
            else:
                P = 1-sin22*massterm
        # nue disappearce
        elif ((filetag=='intrinsics' or filetag=="overlay" or filetag=="ncpi0" or filetag=="ccpi0") and ("nue" in label)):
            P = 1-4*(ue4**2)*(1-ue4**2)*massterm
            P = 1
        # nue appearence
        if (filetag=='fullosc'):
            P = 4*(ue4**2)*(um4**2)*massterm
            P = 0

        weights.append(P)
    return weights

# get sterile osc probs (3+1)
# includes nue app, nue dis, and numu_dis
def oscweights_v2(df,ue4,um4,m41,filetag,numuMode=True,sin22=None):
    weights = []
    m =m41
    do_NueDisappearance = False
    do_NueAppearance    = False

    label_np = df["label"].to_numpy()
    P_np = np.ones(len(df))
    E_np = df['EnuTrue'].to_numpy()*1e-3
    L_np = np.where(df['nu_L_true'] != 1.0,df['nu_L_true'],.47+((df['Zreco'].to_numpy()*.01 - 5)*.001))

    massterm_np = (np.sin(1.27*((m*L_np)/E_np)))**2
    # numu disappearence
    if (filetag in ["overlay","ncpi0","ccpi0","intrinsics"]):
        # do stuff
        if not numuMode:
            P_np = np.where("numu" in label_np,1-4*(um4**2)*(1-um4**2)*massterm_np,1.0)
            if do_NueDisappearance:
                P_np = np.where("nue"  in label_np,1-4*(ue4**2)*(1-ue4**2)*massterm_np,1.0) #Nue Disappearance

        else:
            P_np = np.where("numu" in label_np,1-sin22*massterm_np,1.0)
            if do_NueDisappearance:
                P_np = np.where("nue"  in label_np,1-4*(ue4**2)*(1-ue4**2)*massterm_np,1.0) #Nue Disappearance

    elif (filetag in ["fullosc"]):
        if do_NueAppearance:
            P_np = 4*(ue4**2)*(um4**2)*massterm_np
        else:
            P_np = P_np*0



    P_l = P_np.tolist()
    return P_l

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

def printDiags(cov):
    print("Printing Diagonal:")
    for i in range(cov.shape[0]):
        print(i,cov[i,i])
    print()


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

# cov[j][j] = cov_cnp(obs[j],pred[j]) + stkerr[j]**2
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

def makeCovHist(cov,histName,maxmin = -1.0):
    import ROOT
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetPalette(104)
    hist = ROOT.TH2D(histName,histName,19,250,1200,19,250,1200)
    canv = ROOT.TCanvas("can","can",1200,1000)
    canv.SetLeftMargin(0.15)
    canv.SetRightMargin(0.15)
    canv.SetTopMargin(0.15)
    canv.SetBottomMargin(0.15)
    for i in range(cov.shape[0]):
        for j in range(cov.shape[1]):
            hist.SetBinContent(i+1,j+1,cov[i,j])

    max_min = 0.005
    if maxmin != -1.0:
        max_min = maxmin
    hist.SetMaximum(max_min)
    hist.SetMinimum(-1.0*max_min)
    hist.Draw("COLZ")
    canv.SaveAs(histName+".png")

def getCorr(cov):
    # D = sqrt(diag(S));
    # DInv = inv(D);
    # R = DInv * S * Dinv; /** correlation matrix **/
    diag = np.diag(cov)
    diag_m = np.zeros(cov.shape)
    for i in range(cov.shape[0]):
        diag_m[i,i] = diag[i]
    sqrt_diag = np.sqrt(diag_m)
    inv_sqrt_diag = np.linalg.inv(sqrt_diag)
    corr = inv_sqrt_diag * cov * inv_sqrt_diag
    return corr

# main function
def main():

    sys_dir         = "/home/jmills/workdir/Disappearance/NueAppearanceStudy/Systematics/"
    det_sys_mine_m        = np.nan_to_num(np.loadtxt(sys_dir+"detsys_Enu_1m1p_run13_cov_19binMod.csv",delimiter=','))
    det_sys_davio_m       = np.nan_to_num(np.loadtxt(sys_dir+"detsys_frac_davioThesis_apr7.txt",delimiter=' '))
    mine_minus_davio_m = det_sys_mine_m-det_sys_davio_m

    corr_mine_m = getCorr(det_sys_mine_m)
    corr_davio_m = getCorr(det_sys_davio_m)
    maxmin = 10.0
    makeCovHist(corr_mine_m,"DetSys_CORRJoshu",maxmin)
    makeCovHist(corr_davio_m,"DetSys_CORRDavio",maxmin)

    makeCovHist(det_sys_mine_m,"DetSys_COVJoshu",0.025)
    makeCovHist(det_sys_davio_m,"DetSys_COVDavio",0.025)
    makeCovHist(mine_minus_davio_m,"DetSys_COV_Josh_Minus_Davio",0.025)

    flx_xsc_mine_m        = np.nan_to_num(np.loadtxt(sys_dir+"frac_covar_sel_total_withPi0Weights__nu_energy_reco_19binMod.txt",delimiter=','))
    combined_mine_m = flx_xsc_mine_m + det_sys_mine_m
    makeCovHist(det_sys_davio_m,"Total_COVJoshu",0.025)

    cv_spec = [84.3134, 547.328, 816.636, 1037.69, 1006.43, 1238.81, 1311.03, 1196.64, 1138.31, 985.403, 913.959, 783.934, 632.163, 565.869, 443.254, 329.496, 268.552, 218.284, 191.611]
    cv_sum_sq = np.sum(cv_spec)*np.sum(cv_spec)

    norm_unc_flux_xsec = 0
    norm_unc_detsys    = 0
    for i in range(19):
        for j in range(19):
            norm_unc_flux_xsec += flx_xsc_mine_m
            norm_unc_detsys    +=

    assert 1==2

    # First define all parameters
    #Binary Settings
    sys     = False
    usedata = False
    shapeonly = False
    makeCutData = False
    ntestbins = 25

    #which of Nick's selections do we use?
    finalPred = False
    fset = 0 # 0 = open data, 1-7 = fake datasets, 8 = High E far sideband, 9 = Low BDT far sideband, 10 = Full near sideband
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
    tag='17November2021_jmills_1m1pFullSel'
    filetaglist = {1:['ext','overlay','intrinsics','ncpi0','ccpi0'],
                   2:['ext','overlay','intrinsics'],
                   3:['ext','overlay','intrinsics','ncpi0','ccpi0']}
    df = {1:{},2:{},3:{}}
    psavedir = 'Parquets/'+tag+'/'
    for r in [1,2,3]:
        for filetag in filetaglist[r]:
            pstring = 'SelectionMode%i_%s_fset%i_run%i.parquet.gzip'%(cutMode,filetag,fset,r)
            try:
                print("Adding ", r, filetag)
                df[r][filetag] = pd.read_parquet(psavedir+pstring)
            except:
                print("Didn't find",r, filetag)
                pass
    # use run 3 ext for run 2
    df[2]['ext'] = df[3]['ext'].copy()
    print("Loaded data frames")

    #BDT Settings
    bdtmode = 'avgscore'
    nBDTs = 10
    bdtpower = 0
    # bdtcutRange = (0.5,1.0)

    runs_to_plot = [[1,2,3]]

    POTdict = {1:{ft:DAT_POT1 for ft in filetaglist[1]},
               2:{ft:DAT_POT2 for ft in filetaglist[2]},
               3:{ft:DAT_POT3 for ft in filetaglist[3]}}

    # make lists of sin22theta, m41, chi, nullspectrum
    sin22theta_list = np.logspace(-2, 0, ntestbins, endpoint=True)
    m41_list = np.logspace(-2, 2, ntestbins, endpoint=True)
    chi2_list = np.zeros((ntestbins, ntestbins))
    bestm   = -999
    bests   = -999
    bestchi = 100000000

    # now loop to get oscillated spectra for each m,sin
    m_idx = -1
    times = []
    for m in np.arange(ntestbins):
        m_idx +=1
        sin_idx = -1
        m41=m41_list[m]

        for s in np.arange(ntestbins):
            start_time = time.time()
            sin_idx +=1
            print()
            print(m, s)
            sin41=sin22theta_list[s]
            um4 =.135
            ue4 = sqrt(sin41/(4*um4**2))
            # print statement to track progress
            print("m41: ",m41, " sin2: ", sin41, 'ue4:', ue4)
            # setup bins and dfs
            varName = 'Enu_1m1p'
            bins  = [250+50*i for i in range(20)]
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
            if False:
                # Print Keys
                for k in df[1]['overlay'].keys():
                    print(k)
                assert 1==2

            # loop over all three runs
            for r in runs_to_plot[0]:

                FullPOT += POTdict[r]['overlay']
                if r == 1: POT1 += POTdict[r]['overlay']
                elif r in [2,3]: POT23 += POTdict[r]['overlay']

                #loop to make cuts and set event weights
                for filetag in df[r].keys():
                    dfc = df[r][filetag].copy()
                    # make sure the file cuts are right and drop duplicates
                    dfc['sigprob'] = dfc['sigprobavg']
                    dfc['bdtweight'] = np.where(np.array(dfc['sigprob'])>.5,1,0)
                    dfc.sort_values(by=['run','subrun','event','sigprob'],ascending=False,inplace=True)
                    len_before = len(dfc)
                    if r==2 and filetag=='overlay':
                        dfc.drop_duplicates(subset=['run','subrun','event','EnuTrue'],inplace=True)
                    else:
                        dfc.drop_duplicates(subset=['run','subrun','event'],inplace=True)
                    len_after = len(dfc)
                    if len_before != len_after:
                        print(len_before,len_after,filetag,r)
                        assert 1==2
                    # add labels based on event type
                    # numu_l = np.core.defchararray.find(np.array(dfc['label'],dtype=np.string_),'m')!=-1
                    # labels = np.where(np.logical_and(numu_l,dfc['scedr']>vtxCut),'offvtx',dfc['label'])
                    # dfc['label'] = labels

                    # get weights from oscillation probabilities
                    oscprob=oscweights(dfc,ue4,um4,m41,filetag,numuMode=True,sin22=sin41)
                    oscprob_v2=oscweights_v2(dfc,ue4,um4,m41,filetag,numuMode=True,sin22=sin41)
                    for i in range(len(oscprob)):
                        try:
                            assert oscprob[i] == oscprob_v2[i]
                        except:
                            print(filetag,i,oscprob[i],oscprob_v2[i])
                            assert oscprob[i] == oscprob_v2[i]


                    dfc['oscprob']=oscprob
                    weights = dfc['GenieWeight'] * dfc['oscprob'] * dfc['POTweight']**(-1)
                    weights*= dfc['bdtweight'] * dfc['sigprob']**bdtpower
                    weights*=dfc['sigprob']>0.5
                    weights*=dfc['sigprob']<=1.0
                    weights*=dfc['Enu_1m1p']<high
                    weights*=dfc['Enu_1m1p']>low
                    # add in weights for different binning
                    # binweights = GetBinWeights(dfc,bins)
                    # weights*=binweights
                    dfc['weights'] = weights*POTdict[r][filetag]

                    #now get null weights
                    # CV Spectra Weights
                    nullweights = dfc['GenieWeight'] *  dfc['POTweight']**(-1)
                    nullweights*= dfc['bdtweight'] * dfc['sigprob']**bdtpower
                    nullweights*=dfc['sigprob']>0.5
                    nullweights*=dfc['sigprob']<=1.0
                    nullweights*=dfc['Enu_1m1p']<high
                    nullweights*=dfc['Enu_1m1p']>low
                    # add in weights for different binning
                    # binweights = GetBinWeights(dfc,bins)
                    # nullweights*=binweights
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
            oschist= plt.hist(dftot['Enu_1m1p'],bins=bins,range=(low,high),weights=dftot['weights'])
            nullhist= plt.hist(dftot['Enu_1m1p'],bins=bins,range=(low,high),weights=dftot['nullweights'])
            nullhist_e= plt.hist(dftot_e['Enu_1m1p'],bins=bins,range=(low,high),weights=dftot_e['nullweights'])
            nullhist_mu= plt.hist(dftot_mu['Enu_1m1p'],bins=bins,range=(low,high),weights=dftot_mu['nullweights'])

            xbins  = [(bins[i]+bins[i+1])/2.0 for i in range(numbins)]
            xbin_edges = bins
            pred = np.array(nullhist[0])
            pred_e = np.array(nullhist_e[0])
            pred_mu = np.array(nullhist_mu[0])
            obs = np.array(oschist[0])
            valerrs = poisson_errors(obs)
            binwid = [50 for i in range(numbins)]

            #mcstat error (very tiny)
            tothist= plt.hist(dftot.query('nullweights > 0')['Enu_1m1p'],bins=bins,range=(low,high))
            stkerr = [1.0/sqrt(float(i)) for i in tothist[0]]

            # start of cov matrix and chi2 calculations
            cov = np.zeros((numbins,numbins))

            for j in range(numbins):
                # add cnp errors to cov matrix
                cov[j][j] = cov_cnp(obs[j],pred[j]) + stkerr[j]**2

            # add in detector, flux< and xsec
            rwt_sys_m,det_sys_m = getSysErrors()
            # Comment this out, not sure what Katie is doing here: det_sys_m = getDetSysTot(det_sys_m,pred_e,pred_mu)
            cov += det_sys_m * np.outer(pred,pred)

            # write detvar to a text file
            if(m==0 and s==0):
                file = open("Systematics/fracdetvar_19bins.txt","w")
                x=det_sys_m * np.outer(pred,pred)
                for i in x:
                    for j  in i:
                        file.write(str(j)+'\n')
                file.close()

            cov += rwt_sys_m * np.outer(pred,pred)

            mask1D = np.where(pred==0,False,True)
            mask2D = np.outer(mask1D,mask1D)#np.where(cov==0,False,True)

            Del = (obs - pred)[mask1D]
            cov = cov[mask2D].reshape((len(Del),len(Del)))

            chi2 = np.matmul(np.matmul(Del,np.linalg.inv(cov)),Del)
            print("chi2:",chi2)
            chi2_list[s,m]=(chi2)
            if(chi2<bestchi):
                bestchi=chi2
                bestm=m41_list[m]
                bests=sin22theta_list[s]
            time_taken = time.time() - start_time
            times.append(time_taken)
            print("Time for This Pass:      ", time_taken)
            print("Average Time per Pass:   ", np.mean(times))
            # end of loop over runs
        #end of loop over s
    # end of loop over m
    print()
    # save results to text file for easy loading '\n'.join(mylist)
    file = open("gridsearch_MCsens_allosc.txt","w")
    file.write('\n'.join(map(str,m41_list)))
    file.write('\n')
    file.write('\n'.join(map(str,sin22theta_list)))
    file.write('\n')
    for i in range(len(chi2_list)):
        file.write('\n'.join(map(str,chi2_list[i])))
        file.write('\n')
    file.write((str(bestm)))
    file.write('\n')
    file.write((str(bests)))
    file.write('\n')
    file.write((str(bestchi)))
    file.close()

    print(bestm,bests)




# call main function
if __name__ == "__main__":
    main()
