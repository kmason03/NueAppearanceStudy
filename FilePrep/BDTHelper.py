from __future__ import print_function
from xgboost import XGBClassifier
import numpy as np
from math import sqrt
import random
from joblib import dump, load
import ROOT
import pickle

from SelectionDefs import NewAng, VtxInSimpleFid, VtxInFid, GetPhiT, pTrans,pTransRat
from SelectionDefs import alphaT, ECCQE, ECal, Q2, OpenAngle, PhiDiff, edgeCut
from SelectionDefs import ECCQE_mom, Boost, BoostTracks, Getpz, Getq3q0

#   HELPTER FUNCTION DEFINITIONS

m_e = 0.511 #MeV
m_p = 938.272 #MeV
m_n = 939.565 #MeV
BE = 29.5 #MeV
#newCalib_m = 0.01319672
newCalib_m = 0.01255796
newCalib_b = 0

def LEEweights(Earr):
		
		W = np.ones_like(Earr)
		bounds = [200,250,300,350,400,450,500,600,800]
		weights = [6.37441,5.6554,3.73055,1.50914,1.07428,0.754093,0.476307,0.152327]
		for i in range(len(weights)):
			W += np.where(np.logical_and(Earr>bounds[i],Earr<bounds[i+1]),weights[i],0)
		return W

def GetShCons(evt):

    EU = evt.shower1_sumQ_U*0.0155481
    EV = evt.shower1_sumQ_V*0.01586385
    EY = evt.shower1_sumQ_Y*0.01319672

    return sqrt((EU-EV)**2 + (EU-EY)**2 + (EY-EV)**2)/(EY+1e-6)


def getBjXY(x,Enu_1e1p,Electron_Edep):
            Q2cal_1e1p             = Q2(Enu_1e1p,Electron_Edep,x.Lepton_ThetaReco,'electron')
            EHad_1e1p              = (Enu_1e1p - Electron_Edep - .511)
            x_1e1p                 = Q2cal_1e1p/(2*939.5654*EHad_1e1p)
            y_1e1p                 = EHad_1e1p/Enu_1e1p
            return x_1e1p,y_1e1p

def getBjXYboost(x,Electron_Edep):
    try:
        (pEB_1e1p,eEB_1e1p,pThB_1e1p,
         eThB_1e1p,pPhB_1e1p,ePhB_1e1p,
         EcalB_1e1p,EpCCQEB_1e1p,
         EeCCQEB_1e1p,sphB_1e1p) = BoostTracks(Electron_Edep,x.Proton_Edep,
                                               x.Lepton_ThetaReco,x.Proton_ThetaReco,
                                               x.Lepton_PhiReco,x.Proton_PhiReco,'electron')
        Q2calB_1e1p          = Q2(EcalB_1e1p,eEB_1e1p,eThB_1e1p)
        EHadB_1e1p           = (EcalB_1e1p - eEB_1e1p - .511)
        yB_1e1p              = EHadB_1e1p/EcalB_1e1p
        xB_1e1p              = Q2calB_1e1p/(2*939.5654*EHadB_1e1p)
        return xB_1e1p,yB_1e1p
    except:
        return -9999,-9999

def getSphB_boost(x,Electron_Edep):
    try:
        (pEB_1e1p,eEB_1e1p,pThB_1e1p,
         eThB_1e1p,pPhB_1e1p,ePhB_1e1p,
         EcalB_1e1p,EpCCQEB_1e1p,
         EeCCQEB_1e1p,sphB_1e1p) = BoostTracks(Electron_Edep,x.Proton_Edep,
                                               x.Lepton_ThetaReco,x.Proton_ThetaReco,
                                               x.Lepton_PhiReco,x.Proton_PhiReco,'electron')
        dEp_1e1p            = EpCCQEB_1e1p - EcalB_1e1p
        dEe_1e1p            = EeCCQEB_1e1p - EcalB_1e1p
        dEep_1e1p           = EpCCQEB_1e1p - EeCCQEB_1e1p
        return sqrt(dEp_1e1p**2+dEe_1e1p**2+dEep_1e1p**2)
    except:
        return -9999

def getNewShowerCalibTrainingVarbs(x,newCalib=True):

    if newCalib:
        Electron_Edep       = x.shower1_sumQ_Y*newCalib_m + newCalib_b
        Enu_1e1p            = ECal(Electron_Edep,x.Proton_Edep,'electron',B=BE)
        PT_1e1p             = pTrans(Electron_Edep,x.Proton_Edep,
                                     x.Lepton_ThetaReco,x.Proton_ThetaReco,
                                     x.Lepton_PhiReco,x.Proton_PhiReco,'electron')
        AlphaT_1e1p         = alphaT(Electron_Edep,x.Proton_Edep,
                                     x.Lepton_ThetaReco,x.Proton_ThetaReco,
                                     x.Lepton_PhiReco,x.Proton_PhiReco,'electron')
        PzEnu_1e1p          = Getpz(Electron_Edep,x.Proton_Edep,
                                    x.Lepton_ThetaReco,x.Proton_ThetaReco,'electron') - Enu_1e1p
        Q3_1e1p,Q0_1e1p     = Getq3q0(x.Proton_Edep,Electron_Edep,
                                      x.Proton_ThetaReco,x.Lepton_ThetaReco,
                                      x.Proton_PhiReco,x.Lepton_PhiReco,'electron',B=BE)
        pTRat_1e1p          = pTransRat(Electron_Edep,x.Proton_Edep,
                                        x.Lepton_ThetaReco,x.Proton_ThetaReco,
                                        x.Lepton_PhiReco,x.Proton_PhiReco,'electron')
        BjXB_1e1p,BjYB_1e1p    = getBjXYboost(x,Electron_Edep)
    else:
        Electron_Edep       = x.Electron_Edep
        Enu_1e1p            = x.Enu_1e1p
        PT_1e1p             = x.PT_1e1p
        AlphaT_1e1p         = x.AlphaT_1e1p
        PzEnu_1e1p          = x.PzEnu_1e1p
        Q3_1e1p,Q0_1e1p     = x.Q3_1e1p,x.Q0_1e1p
        pTRat_1e1p          = x.PTRat_1e1p
        BjXB_1e1p,BjYB_1e1p   = x.BjXB_1e1p,x.BjYB_1e1p

    EpCCQE              = ECCQE(x.Proton_Edep,x.Proton_ThetaReco,pid="proton",B=BE)
    EeCCQE              = ECCQE(Electron_Edep,x.Lepton_ThetaReco,pid="electron",B=BE)
    SphB_1e1p           = getSphB_boost(x,Electron_Edep) 



    #Standard varbs
    training_varbs = [Enu_1e1p, Electron_Edep, PT_1e1p, AlphaT_1e1p, SphB_1e1p, PzEnu_1e1p, x.ChargeNearTrunk, 
                      Q0_1e1p, Q3_1e1p, x.Thetas, x.Phis, pTRat_1e1p, x.Proton_TrackLength, x.Lepton_TrackLength, 
                      x.Proton_ThetaReco, x.Proton_PhiReco, x.Lepton_ThetaReco, x.Lepton_PhiReco, max(x.MinShrFrac,-1),
                      max(x.MaxShrFrac,-1), x.shower1_smallQ_Y/(x.shower1_sumQ_Y+1e-6), BjXB_1e1p, BjYB_1e1p]
            
    return training_varbs

def getNewShowerCalibDPFvarbs(x,newCalib=True):
    
    if newCalib:
        Electron_Edep       = x.shower1_sumQ_Y*newCalib_m + newCalib_b
        Enu_1e1p            = ECal(Electron_Edep,x.Proton_Edep,'electron',B=BE)
        PT_1e1p             = pTrans(Electron_Edep,x.Proton_Edep,
                                     x.Lepton_ThetaReco,x.Proton_ThetaReco,
                                     x.Lepton_PhiReco,x.Proton_PhiReco,'electron')
        AlphaT_1e1p         = alphaT(Electron_Edep,x.Proton_Edep,
                                     x.Lepton_ThetaReco,x.Proton_ThetaReco,
                                     x.Lepton_PhiReco,x.Proton_PhiReco,'electron')
        PzEnu_1e1p          = Getpz(Electron_Edep,x.Proton_Edep,
                                    x.Lepton_ThetaReco,x.Proton_ThetaReco,'electron') - Enu_1e1p
        Q3_1e1p,Q0_1e1p     = Getq3q0(x.Proton_Edep,Electron_Edep,
                                      x.Proton_ThetaReco,x.Lepton_ThetaReco,
                                      x.Proton_PhiReco,x.Lepton_PhiReco,'electron',B=BE)
        pTRat_1e1p          = pTransRat(Electron_Edep,x.Proton_Edep,
                                        x.Lepton_ThetaReco,x.Proton_ThetaReco,
                                        x.Lepton_PhiReco,x.Proton_PhiReco,'electron')
        BjXB_1e1p,BjYB_1e1p = getBjXYboost(x,Electron_Edep)
    else:
        Electron_Edep       = x.Electron_Edep
        Enu_1e1p            = x.Enu_1e1p
        PT_1e1p             = x.PT_1e1p
        AlphaT_1e1p         = x.AlphaT_1e1p 
        PzEnu_1e1p          = x.PzEnu_1e1p
        Q3_1e1p,Q0_1e1p     = x.Q3_1e1p,x.Q0_1e1p
        pTRat_1e1p          = x.PTRat_1e1p
        BjXB_1e1p,BjYB_1e1p = x.BjXB_1e1p,x.BjYB_1e1p
    
    EpCCQE              = ECCQE(x.Proton_Edep,x.Proton_ThetaReco,pid="proton",B=BE)
    EeCCQE              = ECCQE(Electron_Edep,x.Lepton_ThetaReco,pid="electron",B=BE)
    SphB_1e1p           = getSphB_boost(x,Electron_Edep) 
    
    
    
    
    DPFvarbs  = [Enu_1e1p,x.Eta,PT_1e1p,AlphaT_1e1p,SphB_1e1p,
                 PzEnu_1e1p,x.ChargeNearTrunk,Q0_1e1p,Q3_1e1p,
                 x.Thetas,x.Phis,pTRat_1e1p,x.Proton_ThetaReco,
                 x.Proton_PhiReco,max(x.MinShrFrac,-1),max(x.MaxShrFrac,-1),
                 BjXB_1e1p,BjYB_1e1p,x.Proton_Edep,Electron_Edep,
                 x.Lepton_ThetaReco,x.Lepton_PhiReco,x.OpenAng,
                 x.Xreco,x.Yreco,x.Zreco,x.MuonPID_int_v[2],
                 x.ProtonPID_int_v[2],x.EminusPID_int_v[2],
                 (x.shower1_smallQ_Y/(x.shower1_sumQ_Y+1e-6) if x.shower1_sumQ_Y != 0 else -1),
                 GetShCons(x),EeCCQE,EpCCQE,x.Proton_TrackLength, x.Lepton_TrackLength]
    
    return DPFvarbs

def precuts(x):
  if x.PassSimpleCuts == 0: return False
  if x.PassShowerReco ==0: return False
  if (x.TotPE < 20 or x.PorchTotPE > 20): return False
  if x.Proton_Edep < 60 or x.Electron_Edep < 35: return False
  if max(x.MaxShrFrac,-1) < 0.2: return False
  if x.FailedBoost_1e1p: return False
  return True

def postcuts(x):
  if x.GammaPID_pix_v[2]/(x.EminusPID_pix_v[2]+0.0001) > 2: return False
  if x.Electron_Edep > 100 and x.MuonPID_int_v[2] > 0.2: return False
  if x._pi0mass > 50: return False
  if x.Proton_ThetaReco > np.pi/2: return False
  if GetShCons(x) > 2: return False
  if x.OpenAng < 0.5: return False
  return True

def signal(x,addPostcuts):
  if not precuts(x): return -1
  if addPostcuts and not postcuts(x): return -1
  if abs(x.MC_parentPDG) != 12: return 0
  return 1




def getDataFromFVV(filename,traintestsplit,
                   newCalib=False,addPostcuts=False,Ecut=None,
                   applyGoodReco=True,shuffle=True,trainAgainstNues=False,
                   treestr=None,useEnu=True):

  f = ROOT.TFile(filename,'READ')
  if treestr is not None: t = f.Get(treestr)
  else: t = f.Get('dlana/FinalVertexVariables') 

  xv = []
  yv = []
  rsev = []

  for i,x in enumerate(t):
    if i%100==0: print('%3.3f%% done' % (100*float(i)/float(t.GetEntries())),end='\r')
    y = signal(x,addPostcuts)
    if y<0: continue
    varbs = getNewShowerCalibTrainingVarbs(x,newCalib=newCalib)
    tE = x.MC_energyInit
    if y==1:
      if x.interactionType != 1001: continue
      if not x.is1l1p0pi: continue
      if x.MC_scedr > 5: continue 
      if applyGoodReco and abs(varbs[0]-tE)/tE > 0.2: continue
    if Ecut is not None and varbs[0]> Ecut: continue
    if useEnu: xv.append(varbs) 
    else: xv.append(varbs[1:]) 
    yv.append(y)
    rsev.append(tuple((x.run,x.subrun,x.event)))
  
  if shuffle:
    z = list(zip(xv,yv,rsev))
    random.shuffle(z)
    xv,yv,rsev = zip(*z)
  
  pivot = int(traintestsplit*len(xv))
  
  return xv[:pivot],yv[:pivot],rsev[:pivot],xv[pivot:],yv[pivot:],rsev[pivot:]
      
  

#    MAIN BDT CLASS

# WARNING: don't run for 18 threads on your personal computer if you don't have them, it will crash your computer

class BDT():

  def initialize(self,tag,n_est=3000,depth=12,gamma=3,n_thr=18,v=1,obj='binary:logistic'):
    self.model = XGBClassifier(verbosity=v, 
                               objective=obj,
                               n_estimators=n_est,
                               max_depth=depth,
                               gamma=gamma,
                               nthread = n_thr)
    self.tag = tag

  def trainModel(self,xt,yt,xv,yv):
    eval_set = [(xv,yv)] 
    self.model.fit(xt,yt,early_stopping_rounds=10, eval_metric='logloss',eval_set=eval_set,verbose=True)

  def saveModel(self,savedir):
    pickle.dump(self.model,open(savedir+'/BDT_1e1p_'+tag+'.pickle.dat'))

  def infer(self,x):
    return self.model.predict(x)

  def getModel(self):
    return self.model
