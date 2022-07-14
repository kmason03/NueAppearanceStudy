from __future__ import print_function
import numpy as np
import os
import matplotlib
matplotlib.use('pdf')
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import ROOT

def main():
    covDir = "/home/jmills/workdir/Disappearance/NueAppearanceStudy/fracCov/"
    fracCOVflux_np   = np.loadtxt(covDir+"fraccovar_flux_19bins.txt",delimiter=',')
    fracCOVxsec_np   = np.loadtxt(covDir+"fraccovar_xsec_19bins.txt",delimiter=',')
    fracCOVreint_np  = np.loadtxt(covDir+"fraccovar_reinteraction_19bins.txt",delimiter=',')
    fracCOVdetsys_np  = np.loadtxt(covDir+"fracdetvar_19bins.txt").reshape((19,19))
    np.savetxt('fracdetvar_np_19bins.csv', fracCOVdetsys_np,delimiter=',')
    fracCOVTotNew_np = fracCOVdetsys_np+fracCOVreint_np+fracCOVxsec_np+fracCOVflux_np
    fracCOVTotOld_np = np.loadtxt("fracdetvar_19bins.txt").reshape((19,19)) + np.loadtxt("fracCov_lauren.txt").reshape((19,19))


    covNames = \
    ["Flux   ",
    "XSec   ",
    "ReInter",
    "DetSys ",
    "TotNew ",
    "TotOld "]

    covMats  = \
    [fracCOVflux_np,
    fracCOVxsec_np,
    fracCOVreint_np,
    fracCOVdetsys_np,
    fracCOVTotNew_np,
    fracCOVTotOld_np]

    # Null Spectra
    trueSpec_l = \
    [[29.9266, 187.287, 270.795, 365.819, 335.556, 406.876, 430.191, 401.403, 362.286, 324.868, 305.201, 257.914, 214.036, 185.429, 147.378, 112.063, 87.9462, 70.7294, 62.1602]]
    # [[1175.67, 7357.57, 10638.2, 14371.2, 13182.3, 15984.1, 16900.1, 15769.1, 14232.4, 12762.5, 11989.8, 10132.2, 8408.44, 7284.59, 5789.75, 4402.42, 3454.97, 2778.61, 2441.97],]
    fnormsquared = []


    specIDX = 0
    for cidx in range(len(covNames)):
        thisFracCOV_np = covMats[cidx]
        thisCOVName    = covNames[cidx]
        scaledCOV_np = np.zeros((19,19))
        for i in range(19):
            for j in range(19):
                scaledCOV_np[i,j] += thisFracCOV_np[i,j]*trueSpec_l[specIDX][i]*trueSpec_l[specIDX][j]
                # if i==j:
                #     if trueSpec_l[specIDX][i] > 0 and fakeSpec_l[specIDX][i] > 0:
                #         scaledCOV_np[i,j] += 3.0 / ((1.0/fakeSpec_l[specIDX][i]) + (2.0/trueSpec_l[specIDX][i]))
                #     else:
                #         scaledCOV_np[i,j] += trueSpec_l[specIDX][i] / 2.0

        thisNum = scaledCOV_np.sum()*1.0
        thisDen = np.sum(trueSpec_l)**2*1.0
        thisfnormsquared = thisNum/thisDen
        fnormsquared.append(thisfnormsquared)
        thisfnorm        = np.sqrt(thisfnormsquared)
        print()
        print(thisCOVName, thisfnormsquared, thisfnorm)


    sysCovFrac_h = ROOT.TH2D("Fractional Total Systematic Covariance Matrix","Fractional Total Systematic Covariance Matrix", 19,250,1200,19,250,1200)

    sysCovScaled_h = ROOT.TH2D("Scaled Systematic Covariance Matrix","Scaled Systematic Covariance Matrix", 19,250,1200,19,250,1200)
    sysCovScaled_shapeonly_h = ROOT.TH2D("Scaled Systematic Covariance Matrix Shape Only","Scaled Systematic Covariance Matrix Shape Only", 19,250,1200,19,250,1200)

    sysCovScaled_cnp_h = ROOT.TH2D("With CNP Diagonal Term","With CNP Diagonal Term", 19,250,1200,19,250,1200)

    sysCovFrac_DetSys_h = ROOT.TH2D("Fractional Detector Systematics Covariance Matrix","Fractional Detector Systematics Covariance Matrix", 19,250,1200,19,250,1200)
    sysCovFrac_Flux_h = ROOT.TH2D("Fractional Flux Systematic Covariance Matrix","Fractional Flux Systematics Covariance Matrix", 19,250,1200,19,250,1200)
    sysCovFrac_XSec_h = ROOT.TH2D("Fractional Cross-Section Systematic Covariance Matrix","Fractional Cross-Section Systematic Covariance Matrix", 19,250,1200,19,250,1200)
    sysCovFrac_Reint_h = ROOT.TH2D("Fractional ReInteraction Systematic Covariance Matrix","Fractional ReInteraction Systematic Covariance Matrix", 19,250,1200,19,250,1200)

    # fracCOVdetsys_np+fracCOVreint_np+fracCOVxsec_np+fracCOVflux_np

    for i in range(19):
        for j in range(19):
            thisTerm = fracCOVTotNew_np[i,j]*trueSpec_l[specIDX][i]*trueSpec_l[specIDX][j]
            shapeonlyterm = thisTerm - fnormsquared[4]*trueSpec_l[specIDX][i]*trueSpec_l[specIDX][j]
            # if i==j:
            #     if trueSpec_l[specIDX][i] > 0 and fakeSpec_l[specIDX][i] > 0:
            #         scaledCOV_np[i,j] += 3.0 / ((1.0/fakeSpec_l[specIDX][i]) + (2.0/trueSpec_l[specIDX][i]))
            #     else:
            #         scaledCOV_np[i,j] += trueSpec_l[specIDX][i] / 2.0
            sysCovScaled_h.SetBinContent(i+1,j+1,thisTerm)
            sysCovFrac_h.SetBinContent(i+1,j+1,fracCOVTotNew_np[i,j])
            sysCovScaled_shapeonly_h.SetBinContent(i+1,j+1,shapeonlyterm)

            sysCovFrac_DetSys_h.SetBinContent(i+1,j+1,fracCOVdetsys_np[i,j])
            sysCovFrac_Flux_h.SetBinContent(i+1,j+1,fracCOVflux_np[i,j])
            sysCovFrac_XSec_h.SetBinContent(i+1,j+1,fracCOVxsec_np[i,j])
            sysCovFrac_Reint_h.SetBinContent(i+1,j+1,fracCOVreint_np[i,j])


    sysCovScaled_h.SetXTitle("Neutrino Energy")
    sysCovScaled_h.SetYTitle("Neutrino Energy")
    ROOT.gStyle.SetOptStat(0)
    canv = ROOT.TCanvas("can","can",1200,1000)
    canv.SetRightMargin(0.15)
    canv.SetLeftMargin(0.15)
    sysCovScaled_h.Draw("COLZ")
    canv.SaveAs(covDir+"ScaledCovTot.png")


    sysCovFrac_h.SetXTitle("Neutrino Energy (MeV)")
    sysCovFrac_h.SetYTitle("Neutrino Energy (MeV)")
    ROOT.gStyle.SetOptStat(0)
    canv = ROOT.TCanvas("can","can",1200,1000)
    canv.SetRightMargin(0.15)
    canv.SetLeftMargin(0.15)
    sysCovFrac_h.Draw("COLZ")
    canv.SaveAs(covDir+"FracCovSysTot.png")

    sysParts = [sysCovFrac_DetSys_h,sysCovFrac_Flux_h,sysCovFrac_XSec_h,sysCovFrac_Reint_h]
    sysPartsSaves = [covDir+"FracCovSysDetsys.png",covDir+"FracCovSysFlux.png",covDir+"FracCovSysXSec.png",covDir+"FracCovSysReint.png"]
    for i in range(len(sysParts)):
        thissys = sysParts[i]
        save = sysPartsSaves[i]
        thissys.SetXTitle("Neutrino Energy")
        thissys.SetYTitle("Neutrino Energy")
        ROOT.gStyle.SetOptStat(0)
        canv = ROOT.TCanvas("can","can",1200,1000)
        canv.SetRightMargin(0.15)
        canv.SetLeftMargin(0.15)
        thissys.Draw("COLZ")
        canv.SaveAs(save)



    sysCovScaled_shapeonly_h.SetXTitle("Neutrino Energy")
    sysCovScaled_shapeonly_h.SetYTitle("Neutrino Energy")
    sysCovScaled_shapeonly_h.Draw("COLZ")
    canv.SaveAs(covDir+"ScaledCovTot_ShapeOnly.png")

# call main function
if __name__ == "__main__":
    main()
