from numpy import mean,asarray,matmul
from math import sqrt,acos,cos,sin,pi,exp,log,isnan,atan2

def NewAng(th,ph):
    a = 0.9999
    b = 0.003
    g = 0.0015

    thp = acos(-1.0*b*sin(th)*cos(ph)-g*sin(th)*sin(ph)+a*cos(th))
    php = atan2(a*sin(th)*sin(ph)+g*cos(th),a*sin(th)*cos(ph)+b*cos(th))

    return thp,php

def VtxInSimpleFid(vtxX,vtxY,vtxZ,edgeCut=10):

    xmin =  0      + edgeCut
    xmax =  256.25 - edgeCut
    ymin = -116.5  + edgeCut
    ymax =  116.5  - edgeCut
    zmin =  0      + edgeCut
    zmax =  1036.8 - edgeCut

    if vtxX > xmax or vtxX < xmin or vtxY > ymax or vtxY < ymin or vtxZ > zmax or vtxZ < zmin:
        return False
    else:
        return True

def VtxInFid(x,y,z):

    if x < 15:        return False
    if x > 256-15:    return False
    if y < -116.5+15: return False
    if y > 116.5-25:  return False
    if z < 15:        return False
    if z > 1000:      return False

    if z > 700 and z < 740: return False
    if z > 50  and z < 58:  return False
    if z > 90  and z < 98:  return False
    if z > 118 and z < 125: return False
    if z > 244 and z < 250: return False
    if z > 285 and z < 292: return False
    if z > 397 and z < 402: return False
    if z > 415 and z < 421: return False
    if z > 807 and z < 813: return False
    if z > 818 and z < 824: return False
    if z > 872 and z < 880: return False

    m = 1/sqrt(3)

    if y <  m*z - 180 and y >  m*z - 200: return False
    if y <  m*z - 370 and y >  m*z - 380: return False
    if y <  m*z - 345 and y >  m*z - 350: return False
    if y <  m*z - 549 and y >  m*z - 555: return False
    if y <  m*z - 605 and y >  m*z - 600: return False
    if y <  m*z - 630 and y >  m*z - 625: return False
    if y < -m*z + 435 and y > -m*z + 415: return False
    if y < -m*z + 615 and y > -m*z + 605: return False
    if y < -m*z + 160 and y > -m*z + 155: return False
    if y < -m*z + 235 and y > -m*z + 231: return False

    if y > m*z -117: return False

    return True

def GetPhiT(El,Ep,Thl,Thp,Phl,Php,pid='electron'):

    EMass = 0.511
    MuMass = 105.6584
    PMass  = 938.673

    if pid == 'electron':
        Pl  = sqrt((El+EMass)**2  - EMass**2)
    elif pid == 'muon':
        Pl  = sqrt((El+MuMass)**2  - MuMass**2)
    Pp  = sqrt((Ep+PMass)**2 - PMass**2)

    Plt = [Pl*sin(Thl)*cos(Phl),Pl*sin(Thl)*sin(Phl),0]
    Ppt = [Pp*sin(Thp)*cos(Php),Pp*sin(Thp)*sin(Php),0]

    PltM = sqrt(Plt[0]**2+Plt[1]**2)
    PptM = sqrt(Ppt[0]**2+Ppt[1]**2)

    try:
        phit = acos(-1.0*(Plt[0]*Ppt[0]+Plt[1]*Ppt[1])/(PltM*PptM))
    except:
        phit = -999

    return phit

def pTrans(El,Ep,Thl,Thp,Phl,Php,pid='electron'):

    EMass = 0.511
    MuMass = 105.6584
    PMass  = 938.673

    if pid == 'electron':
        Pl  = sqrt((El+EMass)**2  - EMass**2)
    elif pid == 'muon':
        Pl  = sqrt((El+MuMass)**2  - MuMass**2)
    Pp  = sqrt((Ep+PMass )**2 - PMass**2 )

    Plt = [Pl*sin(Thl)*cos(Phl),Pl*sin(Thl)*sin(Phl),Pl*cos(Thl)]
    Ppt = [Pp*sin(Thp)*cos(Php),Pp*sin(Thp)*sin(Php),Pp*cos(Thp)]

    Pt  = [Ppt[0]+Plt[0],Ppt[1]+Plt[1],0]

    PtMag = sqrt(Pt[0]**2 + Pt[1]**2)

    return PtMag


def pTransRat(El,Ep,Thl,Thp,Phl,Php,pid='electron'):

    EMass = 0.511
    MuMass = 105.6584
    PMass  = 938.673

    if pid == 'electron':
        Pl  = sqrt((El+EMass)**2  - EMass**2)
    elif pid == 'muon':
        Pl  = sqrt((El+MuMass)**2  - MuMass**2)
    Pp  = sqrt((Ep+PMass )**2 - PMass**2 )

    Plt = [Pl*sin(Thl)*cos(Phl),Pl*sin(Thl)*sin(Phl),Pl*cos(Thl)]
    Ppt = [Pp*sin(Thp)*cos(Php),Pp*sin(Thp)*sin(Php),Pp*cos(Thp)]

    Pt  = [Ppt[0]+Plt[0],Ppt[1]+Plt[1],0]

    PtMag = sqrt(Pt[0]**2 + Pt[1]**2)

    return PtMag / sqrt(Pt[0]**2+Pt[1]**2+(Plt[2]+Ppt[2])**2)


def alphaT(El,Ep,Thl,Thp,Phl,Php,pid='electron'):

    MuMass = 105.6584
    EMass = 0.511
    PMass  = 938.2721

    if pid == 'electron':
        Pl  = sqrt((El+EMass)**2  - EMass**2)
    elif pid == 'muon':
        Pl  = sqrt((El+MuMass)**2  - MuMass**2)
    Pp  = sqrt((Ep+PMass)**2 - PMass**2)

    Plt = [Pl*sin(Thl)*cos(Phl),Pl*sin(Thl)*sin(Phl),0]
    Ppt = [Pp*sin(Thp)*cos(Php),Pp*sin(Thp)*sin(Php),0]

    PltM = sqrt(Plt[0]**2+Plt[1]**2)
    PptM = sqrt(Ppt[0]**2+Ppt[1]**2)

    Pt  = [Ppt[0]+Plt[0],Ppt[1]+Plt[1],0]
    PtMag = sqrt(Pt[0]**2 + Pt[1]**2)

    if PltM == 0 or PptM == 0:
        return 9999

    try:
        alphat = acos(-1.0*(Plt[0]*Pt[0]+Plt[1]*Pt[1])/(PtMag*PltM))
    except:
        alphat = -999

    return alphat

def ECCQE(KE,theta,pid="muon",B=29.5):

    Mn  = 939.5654
    Mp  = 938.2721
    Mmu = 105.66
    Me  = 0.511

    try:
        if pid == "muon":
            Muon_theta = theta
            Muon_KE    = KE
            EnuQE=  0.5*( (2*(Mn-B)*(Muon_KE+Mmu)-((Mn-B)*(Mn-B)+Mmu*Mmu-Mp*Mp  ))/( (Mn-B)-(Muon_KE+Mmu)+sqrt((((Muon_KE+Mmu)*(Muon_KE+Mmu))-(Mmu*Mmu)))*cos(Muon_theta  ) ) );

        if pid == "electron":
            Electron_theta = theta
            Electron_KE    = KE
            EnuQE=  0.5*( (2*(Mn-B)*(Electron_KE+Me)-((Mn-B)*(Mn-B)+Me*Me-Mp*Mp  ))/( (Mn-B)-(Electron_KE+Me)+sqrt((((Electron_KE+Me)*(Electron_KE+Me))-(Me*Me)))*cos(Electron_theta  ) ) );

        elif pid == "proton":
            Proton_theta = theta
            Proton_KE    = KE
            EnuQE = 0.5*( (2*(Mn-B)*(Proton_KE+Mp) -((Mn-B)*(Mn-B)+Mp*Mp  -Me*Me))/( (Mn-B)-(Proton_KE+Mp) +sqrt((((Proton_KE+Mp) *(Proton_KE+Mp) )-(Mp *Mp )))*cos(Proton_theta) ) );
    except:
        EnuQE = -1000000

    return EnuQE

def ECal(KEp,KEmu,pid="electron",B=29.5):

    Mn  = 939.5654
    Mmu = 105.66
    Me  = 0.511
    Mp  = 938.2721

    if pid == "electron":
        EnuCal = KEp+KEmu+B+Me+(Mn-Mp)
    if pid == "muon":
        EnuCal = KEp+KEmu+B+Mmu+(Mn-Mp)

    return EnuCal

def Q2(Enu,El,theta,pid='muon'):

    EMass = 0.511
    MuMass = 105.6584
    PMass  = 938.673

    if pid == 'electron':
        Pl  = sqrt((El+EMass)**2  - EMass**2)
        return -1.0*EMass**2 + 2*Enu*(El + EMass - Pl*cos(theta))
    elif pid == 'muon':
        Pl  = sqrt((El+MuMass)**2  - MuMass**2)
        return -1.0*MuMass**2 + 2*Enu*(El + MuMass - Pl*cos(theta))

def OpenAngle(th1,th2,phi1,phi2):

    try:
        return acos( cos(th1)*cos(th2)+sin(th1)*sin(th2)*cos(phi1-phi2) )
    except:
        return -9999

def PhiDiff(phi1,phi2):

    bigPhi = max([phi1,phi2])
    lilPhi = min([phi1,phi2])

    return bigPhi - lilPhi

def edgeCut(wallDistVec):

    vtxEdge = 5
    trkEdge = 15

    if abs(wallDistVec[0]-wallDistVec[1]) < 5:
        if min(wallDistVec) < vtxEdge:
            return False
        else:
            return True
    else:
        if min(wallDistVec) < trkEdge:
            return False
        else:
            return True


def ECCQE_mom(px,py,pz,pid="muon",B = 29.5,n=[0,0,1]):

    Mn  = 939.5654
    Mp  = 938.272
    Mmu = 105.6584
    Me  = 0.511

    if pid == 'proton':

        E  = sqrt(px**2+py**2+pz**2 + Mp**2)
        pL = (px*n[0]+py*n[1]+pz*n[2])/sqrt(n[0]**2+n[1]**2+n[2]**2)

        num = 2*(Mn - B)*E - ((Mn-B)**2 + Mp**2 - Mmu**2)
        den = (Mn-B) - E + pL
        EnuQE = 0.5*num/den

    elif pid == 'muon':

        E = sqrt(px**2+py**2+pz**2 + Mmu**2)
        pL = (px*n[0]+py*n[1]+pz*n[2])/sqrt(n[0]**2+n[1]**2+n[2]**2)

        num = 2*(Mn - B)*E - ((Mn-B)**2 + Mmu**2 - Mp**2)
        den = (Mn-B) - E + pL
        EnuQE = 0.5*num/den

    elif pid == 'electron':

        E = sqrt(px**2+py**2+pz**2 + Me**2)
        pL = (px*n[0]+py*n[1]+pz*n[2])/sqrt(n[0]**2+n[1]**2+n[2]**2)

        num = 2*(Mn - B)*E - ((Mn-B)**2 + Me**2 - Mp**2)
        den = (Mn-B) - E + pL
        EnuQE = 0.5*num/den

    return EnuQE

def GetCCQEDiff(lE,pE,lTh,pTh,lPh,pPh,pid='electron',B=29.5):

    Mn  = 939.5654
    Mp  = 938.2721
    Ml  = 0.511

    pP = sqrt((pE+Mp)**2 - Mp**2)
    lP = sqrt((lE+Ml)**2 - Ml**2)

    pPx = pP*sin(pTh)*cos(pPh)
    pPy = pP*sin(pTh)*sin(pPh)
    pPz = pP*cos(pTh)

    lPx = lP*sin(lTh)*cos(lPh)
    lPy = lP*sin(lTh)*sin(lPh)
    lPz = lP*cos(lTh)

    ecal = pE + Mp + lE + Ml - (Mn - B)
    elqe = ECCQE_mom(lPx,lPy,lPz,pid,B,[0,0,1])

    return abs(ecal - elqe)/(ecal+elqe)

def SensibleMinimize(lE,pE,lTh,pTh,lPh,pPh,pid,B=25.9):

    vars = [0.05*i for i in range(100)]
    bestEDiff = 99999999

    for x in vars:

        var_lE = lE*x
        thisEDiff = GetCCQEDiff(var_lE,pE,lTh,pTh,lPh,pPh,pid,B)

        if thisEDiff < bestEDiff:
            bestVar = x
            bestEDiff = thisEDiff

    return bestVar


def Boost(Pfx,Pfy,Pfz,w,x,y,z,B=25.9):

    # okay. let's get in the weeds with this
    _pn = sqrt(Pfx**2+Pfy**2+Pfz**2)
    _MAr = 37211.0 #MeV
    _Mn  = 939.5654
    _KEf = sqrt((_MAr + B - _Mn)**2 - _pn**2) - _MAr - B + _Mn
    _En = _Mn - B - _KEf
    _beta = _pn / _En
    _betax = Pfx/_En
    _betay = Pfy/_En
    _betaz = Pfz/_En
    _gamma = 1.0/sqrt(1.0-pow(_beta,2))
    _k = (_gamma - 1.0)/pow(_beta,2)

    lorMat = [

      [  _gamma      , -_gamma*_betax      , -_gamma*_betay     , -_gamma*_betaz     ],

      [  -_gamma*_betax  , 1+_k*_betax**2  , _k*_betax*_betay   , _k*_betax*_betaz   ],

      [  -_gamma*_betay  , _k*_betax*_betay    , 1+_k*_betay**2 , _k*_betay*_betaz   ],

      [  -_gamma*_betaz  , _k*_betax*_betaz    , _k*_betay*_betaz   , 1+_k*_betaz**2 ]
    ]

    bV = matmul(lorMat,[w,x,y,z])

    return bV[0],bV[1],bV[2],bV[3]


def BoostTracks(lE,pE,lTh,pTh,lPh,pPh,pid='muon',B=29.5):

    Mn  = 939.5654
    Mp  = 938.2721
    Mm  = 105.6584
    Me  = 0.511
    Ml  = -999

    if pid == 'muon':
        Ml = Mm

    if pid == 'electron':
        Ml = Me

    pP = sqrt((pE+Mp)**2 - Mp**2)
    lP = sqrt((lE+Ml)**2 - Ml**2)

    pPx = pP*sin(pTh)*cos(pPh)
    pPy = pP*sin(pTh)*sin(pPh)
    pPz = pP*cos(pTh)

    lPx = lP*sin(lTh)*cos(lPh)
    lPy = lP*sin(lTh)*sin(lPh)
    lPz = lP*cos(lTh)

    labEventPz = lPz+pPz

    ####
    pE0,pPx0,pPy0,pPz0 = Boost(pPx+lPx,pPy+lPy,0,pE,pPx,pPy,pPz)
    lE0,lPx0,lPy0,lPz0 = Boost(pPx+lPx,pPy+lPy,0,lE,lPx,lPy,lPz)
    nE0,nPx0,nPy0,nPz0 = Boost(pPx+lPx,pPy+lPy,0,1,0,0,1)

    ecal = pE0 + Mp + lE0 + Ml - (Mn - B)
    epqe = ECCQE_mom(pPx0,pPy0,pPz0,"proton",B,n=[nPx0,nPy0,nPz0])
    elqe = ECCQE_mom(lPx0,lPy0,lPz0,"muon",B,n=[nPx0,nPy0,nPz0])

    thisSph = sqrt((ecal-epqe)**2+(ecal-elqe)**2+(elqe-epqe)**2)
    ####

    pE,pPx,pPy,pPz = Boost(pPx+lPx,pPy+lPy,0,pE+Mp,pPx,pPy,pPz)
    lE,lPx,lPy,lPz = Boost(pPx+lPx,pPy+lPy,0,lE+Ml,lPx,lPy,lPz)
    nE,nPx,nPy,nPz = Boost(pPx+lPx,pPy+lPy,0,0,0,0,1)  # z unit vector needs to be boosted
    nE,xPx,xPy,xPz = Boost(pPx+lPx,pPy+lPy,0,0,1,0,0)  # x unit vector needs to be boosted

    pP = sqrt(pPx**2+pPy**2+pPz**2)
    lP = sqrt(lPx**2+lPy**2+lPz**2)

#    newPTh = acos(pPz/pP)
#    newPPh = atan2(pPy,pPx)
#    newLTh = acos(lPz/lP)
#    newLPh = atan2(lPy,lPx)

    pPdn = (nPx*pPx+nPy*pPy+nPz*pPz)/(sqrt(nPx**2+nPy**2+nPz**2)*sqrt(pPx**2+pPy**2+pPz**2))
    lPdn = (nPx*lPx+nPy*lPy+nPz*lPz)/(sqrt(nPx**2+nPy**2+nPz**2)*sqrt(lPx**2+lPy**2+lPz**2))
    newPTh = acos(pPdn)
    newLTh = acos(lPdn)

    ptPdx = (xPx*pPx+xPy*pPy)/(sqrt(xPx**2+xPy**2+xPz**2)*sqrt(pPx**2+pPy**2))
    ltPdx = (xPx*lPx+xPy*lPy)/(sqrt(xPx**2+xPy**2+xPz**2)*sqrt(lPx**2+lPy**2))
    magPPh = acos(ptPdx)
    magLPh = acos(ltPdx)
    newPPh = magPPh if pPy > 0 else -1.0*magPPh
    newLPh = magLPh if lPy > 0 else -1.0*magLPh

    newPE  = pE - Mp
    newLE  = lE - Ml

    ecal = pE + lE - (Mn - B)
    epqe = ECCQE_mom(pPx,pPy,pPz,"proton",B,n=[nPx,nPy,nPz])
    elqe = ECCQE_mom(lPx,lPy,lPz,pid,B,n=[nPx,nPy,nPz])

    return newPE,newLE,newPTh,newLTh,newPPh,newLPh,ecal,epqe,elqe,thisSph

def Getpz(El,Ep,lTh,pTh,pid='electron'):

    EMass = 0.511
    MuMass = 105.6584
    PMass  = 938.673

    if pid == 'electron':
        Pl  = sqrt((El+EMass)**2  - EMass**2)
    elif pid == 'muon':
        Pl  = sqrt((El+MuMass)**2  - MuMass**2)
    Pp  = sqrt((Ep+PMass)**2 - PMass**2)

    Ppz = Pp*cos(pTh)
    Plz = Pl*cos(lTh)

    pz  = Ppz + Plz

    return pz

def Getq3q0(Ep,El,pTh,lTh,pPh,lPh,pid='electron',B=29.5):

    NMass = 939.5654
    PMass  = 938.673
    EMass = 0.511
    MuMass = 105.6584

    if pid == 'electron':
        Pl  = sqrt((El+EMass)**2  - EMass**2)
        Ml = EMass
    elif pid == 'muon':
        Pl  = sqrt((El+MuMass)**2  - MuMass**2)
        Ml = MuMass
    Pp  = sqrt((Ep+PMass)**2 - PMass**2)

    Ppx = Pp*sin(pTh)*cos(pPh)
    Ppy = Pp*sin(pTh)*sin(pPh)
    Ppz = Pp*cos(pTh)

    Plx = Pl*sin(lTh)*cos(lPh)
    Ply = Pl*sin(lTh)*sin(lPh)
    Plz = Pl*cos(lTh)

    ecal = Ep + PMass + El + Ml - (NMass - B)

    q0 = ecal - (El+Ml)

    pnu = Plz + Ppz

    q3 = sqrt( (-Plx)**2 + (-Ply)**2 + (pnu - Plz)**2 )

    return q3,q0


def GetTotPE(coincidenceThresh, flashes):

    totPE       = 0
    flash_found = 0
    flash_bins  = []
    for x in xrange(len(flashes)):

        if flashes[x] > coincidenceThresh and flash_found == 1:
            flash_bins.append(x)
            totPE+= flashes[x]

        if flashes[x] < coincidenceThresh and flash_found == 1:
            break

        if flashes[x] > coincidenceThresh and flash_found == 0:
            totPE+= flashes[x]
            flash_bins.append(x)
            flash_found = 1

    return totPE,flash_bins



def CorrectionFactor(x,y,z,theta,phi,L,calibmap_v):        #  assumes straight line

    dr    = 0.5
    steps = int(L / dr)
    dx    = sin(theta)*cos(phi)*dr
    dy    = sin(theta)*sin(phi)*dr
    dz    = cos(theta)*dr

    sumFac = 0
    for i in range(steps):
        thisBin  = calibmap_v[0].FindBin(x+i*dx,y+i*dy,z+i*dz)
        corrFac0 = calibmap_v[0].GetBinContent(thisBin)
        thisBin  = calibmap_v[1].FindBin(x+i*dx,y+i*dy,z+i*dz)
        corrFac1 = calibmap_v[1].GetBinContent(thisBin)
        thisBin  = calibmap_v[2].FindBin(x+i*dx,y+i*dy,z+i*dz)
        corrFac2 = calibmap_v[2].GetBinContent(thisBin)

        sumFac+=(corrFac0+corrFac1+corrFac2)/3.0

    avgCorrFac = sumFac/steps

    return avgCorrFac

def CorrectionFactorPoint(x,y,z):

    thisBin  = calibMap_v[0].FindBin(x,y,z)
    corrFac0 = calibMap_v[0].GetBinContent(thisBin)
    thisBin  = calibMap_v[1].FindBin(x,y,z)
    corrFac1 = calibMap_v[1].GetBinContent(thisBin)
    thisBin  = calibMap_v[2].FindBin(x,y,z)
    corrFac2 = calibMap_v[2].GetBinContent(thisBin)

    avgFac=(corrFac0+corrFac1+corrFac2)/3.0

    return avgFac
