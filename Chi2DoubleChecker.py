from __future__ import print_function
import numpy as np
import os
import matplotlib
matplotlib.use('pdf')
import matplotlib.colors as colors
import matplotlib.pyplot as plt

def main():
    fracCOV_np = np.loadtxt("fracdetvar_19bins.txt").reshape((19,19)) + np.loadtxt("fracCov_lauren.txt").reshape((19,19))
    eigVal, eigVec = np.linalg.eig(fracCOV_np)
    print("Printing", np.all(np.linalg.eigvals(fracCOV_np) > 0))
    for e in eigVal:
        print(e)
    # for i in range(19):
    #     for j in range(19):
    #         print(i,j,fracCOV_np[i,j])


    shapeOnly = False
    trueSpec_l = \
    # Null Spectra
    [[1175.67, 7357.57, 10638.2, 14371.2, 13182.3, 15984.1, 16900.1, 15769.1, 14232.4, 12762.5, 11989.8, 10132.2, 8408.44, 7284.59, 5789.75, 4402.42, 3454.97, 2778.61, 2441.97],
    [],
    [],]
    # Oscillated Spectra
    # [[891.23, 5699.63, 7660.71, 10519.1, 9408.99, 11655.7, 12879, 12453.7, 11433.9, 10529.1, 10133.4, 8672.97, 7319.37, 6406.8, 5161.78, 3970.42, 3129.81, 2528.59, 2235.75],

    # Thrown Universe
    fakeSpec_l = \
    [[879, 4803, 6633, 9482, 7658, 9859, 11415, 11075, 10428, 10036, 8057, 6552, 6672, 5946, 4296, 2916, 2072, 1953, 2020],
    [],
    [],]

    specIDX = 0
    scaledCOV_np = np.zeros((19,19))
    if shapeOnly:
        predIntegral = 0.0
        obsIntegral  = 0.0
        covIntegral  = 0.0
        for i in range(19):
            predIntegral += trueSpec_l[specIDX][i]
            obsIntegral  += fakeSpec_l[specIDX][i]
            for j in range(19):
                covIntegral += fracCOV_np[i,j]*trueSpec_l[specIDX][i]*trueSpec_l[specIDX][j]
        fnorm = covIntegral/(predIntegral*predIntegral)
        print("FNORM:",fnorm)
        for i in range(19):
            trueSpec_l[specIDX][i] = trueSpec_l[specIDX][i]*(obsIntegral/predIntegral)
        for i in range(19):
            for j in range(19):
                scaledCOV_np[i,j] = (fracCOV_np[i,j]-fnorm)*trueSpec_l[specIDX][i]*trueSpec_l[specIDX][j]
                if i==j:
                    print(i,j,fracCOV_np[i,j]-fnorm)
                if i==j:
                    if trueSpec_l[specIDX][i] > 0 and fakeSpec_l[specIDX][i] > 0:
                        scaledCOV_np[i,j] += 3.0 / ((1.0/fakeSpec_l[specIDX][i]) + (2.0/trueSpec_l[specIDX][i]))
                        # print(i,j,3.0 / ((1.0/fakeSpec_l[specIDX][i]) + (2.0/trueSpec_l[specIDX][i])))
                    else:
                        scaledCOV_np[i,j] += trueSpec_l[specIDX][i] / 2.0
                        # print(i,j,trueSpec_l[specIDX][i] / 2.0)

    else:
        for i in range(19):
            for j in range(19):
                # if i==j:
                #     print(i,j,scaledCOV_np[i,j],"Before")
                scaledCOV_np[i,j] += fracCOV_np[i,j]*trueSpec_l[specIDX][i]*trueSpec_l[specIDX][j]
                # if i==j:
                #     print(i,j,fracCOV_np[i,j]*trueSpec_l[specIDX][i]*trueSpec_l[specIDX][j], fracCOV_np[i,j], trueSpec_l[specIDX][i], trueSpec_l[specIDX][j])
                #     print(i,j,scaledCOV_np[i,j],"After")
                if i==j:
                    if trueSpec_l[specIDX][i] > 0 and fakeSpec_l[specIDX][i] > 0:
                        scaledCOV_np[i,j] += 3.0 / ((1.0/fakeSpec_l[specIDX][i]) + (2.0/trueSpec_l[specIDX][i]))
                    else:
                        scaledCOV_np[i,j] += trueSpec_l[specIDX][i] / 2.0
    # for i in range(19):
    #     for j in range(19):
    #         print(i,j,scaledCOV_np[i,j])
    # for x in range(19):
    #     print(x,scaledCOV_np[x,x])
    print()

    eigVal, eigVec = np.linalg.eig(scaledCOV_np)
    print("Printing", np.all(np.linalg.eigvals(scaledCOV_np) > 0))
    for e in eigVal:
        print(e)
    return 0
    invCOV_np = np.linalg.inv(scaledCOV_np)
    chi2Test = 0
    print("\n\nHere\n\n")
    for i in range(19):
        for j in range(19):
            chi2Test += (fakeSpec_l[specIDX][i] - trueSpec_l[specIDX][i])*invCOV_np[i,j]*(fakeSpec_l[specIDX][j] - trueSpec_l[specIDX][j])
            # print(i,j,(fakeSpec_l[specIDX][i] - trueSpec_l[specIDX][i])*invCOV_np[i,j]*(fakeSpec_l[specIDX][j] - trueSpec_l[specIDX][j]))
            # if i==j:
                # print(i,j,  invCOV_np[i,j])
            # print(fakeSpec_l[specIDX][i],trueSpec_l[specIDX][i],invCOV_np[i,j],fakeSpec_l[specIDX][j],trueSpec_l[specIDX][j])
    print()
    print("chi2Test: ", chi2Test)
    print("ShapeOnly:",shapeOnly)
# call main function
if __name__ == "__main__":
    main()
