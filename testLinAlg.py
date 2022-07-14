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

# get the sys cov matrices
def getSysErrors(get20 = False):
    sys_dir         = "/home/jmills/workdir/Disappearance/NueAppearanceStudy/Systematics/"
    flux_xsex_sys_m = np.nan_to_num(np.loadtxt(sys_dir+"frac_covar_sel_total_withPi0Weights__nu_energy_reco_19binMod.txt",delimiter=','))
    det_sys_m       = np.nan_to_num(np.loadtxt(sys_dir+"detsys_Enu_1m1p_run13_cov_19binMod.csv",delimiter=','))
    if get20:
        flux_xsex_sys_m = np.nan_to_num(np.loadtxt(sys_dir+"frac_covar_sel_total_withPi0Weights__nu_energy_reco.txt",delimiter=','))
        det_sys_m       = np.nan_to_num(np.loadtxt(sys_dir+"detsys_Enu_1m1p_run13_cov.csv",delimiter=','))
    return flux_xsex_sys_m, det_sys_m

def cov_cnp(M,mu):
    cov=0
    if mu > 0:
        if M != 0:
            cov = (3 / (2 / mu + 1 / M))
        else:
            cov = mu/2
    return cov

def main():

    truedata = [73.0912, 459.729, 703.92, 865.677, 799.659, 1062.61, 1171.13, 1031.6, 910.838, 755.541, 711.276, 631.343, 534.493, 504.713, 408.018, 312.531, 256.266, 207.206, 179.879]
    fakedata = [52, 378, 658, 757, 801, 1040, 1147, 1103, 909, 732, 674, 676, 634, 569, 350, 347, 255, 191, 185]
    print("\n\n\n")
    print("Identity")
    onesDiag = np.zeros((19,19))
    for i in range(19):
        onesDiag[i,i] = 1

    invOnesDiag = np.linalg.inv(onesDiag)
    chisq = 0
    for i in range(19):
        for j in range(19):
            chisq += (truedata[i] - fakedata[i])*invOnesDiag[i,j]*(truedata[j]-fakedata[j])
    print("ChiSq",chisq)

    print("\n\n\n")
    print("5xIdentity")

    onesDiag = np.zeros((19,19))
    for i in range(19):
        onesDiag[i,i] = 5

    invOnesDiag = np.linalg.inv(onesDiag)
    chisq = 0
    for i in range(19):
        for j in range(19):
            chisq += (truedata[i] - fakedata[i])*invOnesDiag[i,j]*(truedata[j]-fakedata[j])
    print("ChiSq",chisq)

    print("\n\n\n")
    print("Rands")
    sys1, sys2 = getSysErrors()
    rands = sys1+sys2 #np.random.rand(19,19)
    # for i in range(5):
    #     rands[i,i] = 5

    invOnesDiag = np.linalg.inv(rands)
    chisq = 0
    for i in range(19):
        for j in range(19):
            chisq += (truedata[i] - fakedata[i])*invOnesDiag[i,j]*(truedata[j]-fakedata[j])
    val1 = chisq
    print("ChiSq",chisq/val1)

    print("\n\n\n")
    print("5xRands")
    rands = 5*rands
    # for i in range(5):
    #     rands[i,i] = 5

    invOnesDiag = np.linalg.inv(rands)
    chisq = 0
    for i in range(19):
        for j in range(19):
            chisq += (truedata[i] - fakedata[i])*invOnesDiag[i,j]*(truedata[j]-fakedata[j])
    print("ChiSq",chisq/val1)
    print("\n\n\n")




    # # start of cov matrix and chi2 calculations
    # pred   = np.array([29.87467642,182.12288997,265.48755242,359.95383653,335.23421611,404.21765838,436.59580894,405.02981761,365.45004252,331.29406276,302.74911042,253.16113734,216.91284118,191.56492456,140.66893367,111.77782016,83.55767308,70.95802436,63.90498322])
    # stkerr = np.array([0.08873565094161139, 0.03456505649101418, 0.028665182485640765, 0.025474090105633382, 0.026462806201248155, 0.023722272266019126, 0.023044900641260895, 0.023966279201637065, 0.024776471541572212, 0.026407386650992314, 0.027451402562301407, 0.029424494316824982, 0.03251280443811776, 0.03537745688386125, 0.040555355282690636, 0.045980048987170286, 0.052342392259021375, 0.05496497099293127, 0.05716619504750295])
    # obs    = np.array([-10.617459319074978,-12.647952076739976,-91.1666831761966,-89.85956459630236,-113.20323111053546,-77.95043922313238,2.253509189845829,49.470849599977555,74.01108301107507,92.36224381993446,111.33849259251302,109.09815457757324,105.45281583291779,105.91330579594339,83.92959012368416,69.43615562178661,53.03835851803535,45.406622102086374,43.9114412873285])
    # mask1D = np.where(pred==0,False,True)
    # mask2D = np.outer(mask1D,mask1D)#np.where(cov==0,False,True)
    #
    #
    # numbins = 19
    # cov = np.zeros((numbins,numbins))
    #
    # for j in range(numbins):
    #     # add cnp errors to cov matrix
    #     cov[j][j] = cov_cnp(obs[j],pred[j]) + stkerr[j]**2
    #
    # # add in detector, flux< and xsec
    # rwt_sys_m,det_sys_m = getSysErrors()
    # # Comment this out, not sure what Katie is doing here: det_sys_m = getDetSysTot(det_sys_m,pred_e,pred_mu)
    # cov += det_sys_m * np.outer(pred,pred)
    # # write detvar to a text file
    #
    #
    # cov += rwt_sys_m * np.outer(pred,pred)
    #
    # cov = cov[mask2D].reshape((numbins,numbins))
    # print(cov)
    # Del = (obs - pred)[mask1D]
    # chi2 = np.matmul(np.matmul(Del,np.linalg.inv(cov)),Del)
    # print()
    # print("Chi2")
    # print(chi2)
    # print("\n\n\n\nTry20bins")
    #
    # pred   = np.array([0,29.87467642,182.12288997,265.48755242,359.95383653,335.23421611,404.21765838,436.59580894,405.02981761,365.45004252,331.29406276,302.74911042,253.16113734,216.91284118,191.56492456,140.66893367,111.77782016,83.55767308,70.95802436,63.90498322])
    # stkerr = np.array([0.12,0.08873565094161139, 0.03456505649101418, 0.028665182485640765, 0.025474090105633382, 0.026462806201248155, 0.023722272266019126, 0.023044900641260895, 0.023966279201637065, 0.024776471541572212, 0.026407386650992314, 0.027451402562301407, 0.029424494316824982, 0.03251280443811776, 0.03537745688386125, 0.040555355282690636, 0.045980048987170286, 0.052342392259021375, 0.05496497099293127, 0.05716619504750295])
    # obs    = np.array([-10,-10.617459319074978,-12.647952076739976,-91.1666831761966,-89.85956459630236,-113.20323111053546,-77.95043922313238,2.253509189845829,49.470849599977555,74.01108301107507,92.36224381993446,111.33849259251302,109.09815457757324,105.45281583291779,105.91330579594339,83.92959012368416,69.43615562178661,53.03835851803535,45.406622102086374,43.9114412873285])
    # mask1D = np.where(pred==0,False,True)
    # mask2D = np.outer(mask1D,mask1D)#np.where(cov==0,False,True)
    #
    #
    # numbins = 20
    # cov = np.zeros((numbins,numbins))
    #
    # for j in range(numbins):
    #     # add cnp errors to cov matrix
    #     cov[j][j] = cov_cnp(obs[j],pred[j]) + stkerr[j]**2
    #
    # # add in detector, flux< and xsec
    # rwt_sys_m,det_sys_m = getSysErrors(True)
    # # Comment this out, not sure what Katie is doing here: det_sys_m = getDetSysTot(det_sys_m,pred_e,pred_mu)
    # cov += det_sys_m * np.outer(pred,pred)
    # # write detvar to a text file
    #
    #
    # cov += rwt_sys_m * np.outer(pred,pred)
    # print(cov.shape)
    # print(mask2D.shape)
    # Del = (obs - pred)[mask1D]
    # cov = cov[mask2D].reshape((len(Del),len(Del)))
    # print(cov)
    # chi2 = np.matmul(np.matmul(Del,np.linalg.inv(cov)),Del)
    # print()
    # print("Chi2")
    # print(chi2)



# call main function
if __name__ == "__main__":
    main()
