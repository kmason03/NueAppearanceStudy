from __future__ import print_function
import numpy as np
import os
import matplotlib
matplotlib.use('pdf')
import matplotlib.colors as colors
import matplotlib.pyplot as plt

def main():
    bins = np.loadtxt("bins_10.txt")
    m41_list        = np.loadtxt("bins_10.txt")[0,:].tolist() #bin edges
    sin22theta_list = np.loadtxt("bins_10.txt")[1,:].tolist() #bin edges
    nm41 = len(m41_list)-1
    nsin22 = len(sin22theta_list)-1
    nUniverse = 1000

    outDir = "Plots/"

    universeChisFiles_list = ["chis_40x_all_1000.txt"]
    TestChisFiles_list =     ["chis_40x_nonfreq_shaperate.txt"]

    y,x = np.meshgrid(m41_list,sin22theta_list)
    xmod = np.zeros((nsin22,nm41))
    ymod = np.zeros((nsin22,nm41))
    for i in range(xmod.shape[0]):
        for j in range(xmod.shape[1]):
            xmod[i,j] = np.sqrt(x[i,j]*x[i+1,j+1])
            ymod[i,j] = np.sqrt(y[i,j]*y[i+1,j+1])


    # Specific to One Test:
    test_idx = 0
    universeDeltaChis_Flat = np.loadtxt(universeChisFiles_list[test_idx])
    universeDeltaChis_Arranged = np.zeros((nsin22,nm41,nUniverse))
    critChisMap = np.zeros((nsin22,nm41))
    for s in range(nsin22):
        for m in range(nm41):
            thisPTSorted = np.sort(universeDeltaChis_Flat[m*nm41+s,:])
            universeDeltaChis_Arranged[s,m,:] = thisPTSorted
            critChisMap[s,m] = thisPTSorted[int(nUniverse*0.9)]

    # Make Plot of Critical Chi2s
    fig,ax = plt.subplots(figsize=(10,8))
    # critChisMap[critChisMap > 4.6] = 4.6
    plt.pcolormesh(x,y,critChisMap)
    cbar = plt.colorbar()
    cbar.set_label(r'$ \chi^2_c$',rotation=0,fontsize=20)
    plt.title(r'$\chi^2_C$ for 90% CL',fontsize=35)
    plt.xlabel(r"$sin^2$(2$\theta_{\mu \mu}$)",fontsize=30)
    plt.ylabel(r"$\Delta m^2_{41}$",fontsize=30)
    plt.xscale("log")
    plt.yscale("log")
    plt.savefig(outDir+"shaperate_critchi2map_freq_90_40x.png")
    # Make plot showing fractional difference from Wilks' Theorem
    fracFromWilksMap = (critChisMap-4.6)/4.6
    fig,ax = plt.subplots(figsize=(10,8))
    plt.pcolormesh(x, y, fracFromWilksMap)
    cbar = plt.colorbar()
    cbar.set_label(r'$ (\chi^2_c - 4.6)/4.6$',rotation=0,fontsize=20,labelpad=40)
    plt.title(r'Frac Diff from 2DoF 90% CL',fontsize=35)
    plt.xlabel(r"$sin^2$(2$\theta_{\mu \mu}$)",fontsize=30)
    plt.ylabel(r"$\Delta m^2_{41}$",fontsize=30)
    plt.xscale("log")
    plt.yscale("log")
    plt.savefig(outDir+"shaperate_fracdiff_critchi2_freq_90_40x.png")
    # These are the chis2 (not delta chi2!) Between the grid point and the CV
    TestChis_Flat = np.loadtxt(TestChisFiles_list[test_idx])
    TestChis_Arranged = np.zeros((nsin22,nm41))
    freqExclusion = np.zeros((nsin22,nm41))
    for s in range(nsin22):
        for m in range(nm41):
            TestChis_Arranged[s,m] = TestChis_Flat[m*nm41+s]
            freqExclusion[s,m] = (TestChis_Arranged[s,m] > critChisMap[s,m])
    # freqExclusion = np.where((TestChis_Arranged > critChisMap),1,0)
    print(np.max(TestChis_Arranged))

    fig,ax = plt.subplots(figsize=(10,8))
    plt.pcolormesh(x,y,TestChis_Arranged)
    cbar = plt.colorbar()
    contours = plt.contour(xmod, ymod, TestChis_Arranged,[4.6], colors='white'); #Where above 4.6 exclude
    contours2 = plt.contour(xmod, ymod, freqExclusion,[0.9], colors='red'); #Where above frequentist limits exclude

    cbar.set_label(r'$ \chi^2$',rotation=0,fontsize=20)
    plt.title(r'$\chi^2$ for Osc vs CV',fontsize=35)
    plt.xlabel(r"$sin^2$(2$\theta_{\mu \mu}$)",fontsize=30)
    plt.ylabel(r"$\Delta m^2_{41}$",fontsize=30)
    plt.xscale("log")
    plt.yscale("log")
    plt.savefig(outDir+"shaperate_chi2map_Wilks_90_40x.png")


    universeDeltaChis_Flat_40x = np.loadtxt("chis_40x_all_1000.txt")
    universeDeltaChis_Arranged_40x = np.zeros((nsin22,nm41,nUniverse))
    critChisMap_40x = np.zeros((nsin22,nm41))
    for s in range(nsin22):
        for m in range(nm41):
            thisPTSorted = np.sort(universeDeltaChis_Flat_40x[m*nm41+s,:])
            universeDeltaChis_Arranged_40x[s,m,:] = thisPTSorted
            critChisMap_40x[s,m] = thisPTSorted[int(nUniverse*0.9)]

    log_absDiff = np.zeros((nsin22,nm41))
    for s in range(nsin22):
        for m in range(nm41):
            log_absDiff[s,m] = np.log(abs(critChisMap_40x[s,m] - critChisMap[s,m]))

    fig,ax = plt.subplots(figsize=(10,8))
    plt.pcolormesh(x,y,log_absDiff)
    cbar = plt.colorbar()
    # cbar.set_label(r'$log(abs(\chi^2_40x - \chi^2_40x))',rotation=0,fontsize=20)
    plt.title(r'$Log of Abs Diff 40x CritChi - 1x CritChi',fontsize=35)
    plt.xlabel(r"$sin^2$(2$\theta_{\mu \mu}$)",fontsize=30)
    plt.ylabel(r"$\Delta m^2_{41}$",fontsize=30)
    plt.xscale("log")
    plt.yscale("log")
    plt.savefig(outDir+"log_absDiff_40x_40x_critchimaps.png")

# call main function
if __name__ == "__main__":
    main()
