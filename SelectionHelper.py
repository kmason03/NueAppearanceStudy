from __future__ import print_function
import numpy as np
from math import sqrt,cos,acos,pi,exp,sin,atan2,log
import pandas as pd
import pickle
from xgboost import XGBClassifier

from SelectionDefs import NewAng, VtxInSimpleFid, VtxInFid, GetPhiT, pTrans,pTransRat
from SelectionDefs import alphaT, ECCQE, ECal, Q2, OpenAngle, PhiDiff, edgeCut
from SelectionDefs import ECCQE_mom, Boost, BoostTracks, Getpz, Getq3q0

from BDTHelper import getNewShowerCalibTrainingVarbs,getNewShowerCalibDPFvarbs

# Load a bunch of text files, set variable strings, define directories, etc...

weightdir = '/home/kmason/fullosc/NicksCode/1e1pdata/aux/'

dpf_varb_names = ['Enu_1e1p','Eta','PT_1e1p','AlphaT_1e1p',
                  'SphB_1e1p','PzEnu_1e1p','ChargeNearTrunk',
                  'Q0_1e1p','Q3_1e1p','Thetas','Phis','PTRat_1e1p',
                  'Proton_ThetaReco','Proton_PhiReco',
                  'MinShrFrac','MaxShrFrac',
                  'BjXB_1e1p','BjYB_1e1p','Proton_Edep',
                  'Electron_Edep',
                  'Lepton_ThetaReco','Lepton_PhiReco',
                  'OpenAng','Xreco','Yreco','Zreco',
                  'MPIDY_muon','MPIDY_proton','MPIDY_eminus',
                  'shower_fraction','Shower_Consistency',
                  'EnuQE_lepton','EnuQE_proton','Proton_TrackLength','Lepton_TrackLength',
                  'EnuTrue','BDTscore_1e1p','ccnc','interactionType','nlepton','nproton','scedr',
                  'POTweight','GenieWeight','LEEweight','label',
                  'newpi0flag','oldpi0flag','datarun','filetag','cutLevel']

efficiency_names = ['EnuTrue','ccnc','scedr',
                    'POTweight','GenieWeight','LEEweight','label',
                    'newpi0flag','datarun','filetag','cutLevel']

sys_varb_names = ['Enu_1e1p','Eta','PT_1e1p','AlphaT_1e1p',
                  'SphB_1e1p','PzEnu_1e1p','ChargeNearTrunk',
                  'Q0_1e1p','Q3_1e1p','Thetas','Phis','PTRat_1e1p',
                  'Proton_ThetaReco','Proton_PhiReco',
                  'MinShrFrac','MaxShrFrac',
                  'BjXB_1e1p','BjYB_1e1p','Proton_Edep',
                  'Electron_Edep',
                  'Lepton_ThetaReco','Lepton_PhiReco',
                  'OpenAng','Xreco','Yreco','Zreco',
                  'MPIDY_muon','MPIDY_proton','MPIDY_eminus',
                  'shower_fraction','Shower_Consistency',
                  'EnuQE_lepton','EnuQE_proton','Proton_TrackLength','Lepton_TrackLength',
                  'Etrue','sample']

train_varb_names = ['Enu_1e1p', 'Electron_Edep', 'PT_1e1p', 'AlphaT_1e1p', 'SphB_1e1p', 'PzEnu_1e1p', 'ChargeNearTrunk',
                    'Q0_1e1p', 'Q3_1e1p', 'Thetas', 'Phis', 'pTRat_1e1p', 'Proton_TrackLength', 'Lepton_TrackLength',
                    'Proton_ThetaReco', 'Proton_PhiReco', 'Lepton_ThetaReco', 'Lepton_PhiReco', 'MinShrFrac',
                    'MaxShrFrac','shower_fraction' , 'BjXB_1e1p', 'BjYB_1e1p']

rsev_names = ['run','subrun','event','vtxid']

OVR_POT1  = 4.71579e+20
OVR_POT1rr  = 1.33681555217E+21
NUE_POT1  = 9.80259e+22
NUE_POT1rr  = 1.15690338134E+23
LOWE_POT1 = 6.05398370114e+23
LOWM_POT1 = 1.63103516179e+21
DIRT_POT1 = 1.0
OPEN_POT1 = 4.403e19
EXT_POT1  = 22474918.0 / 9776965.0 * OPEN_POT1
NCPi0_POT1 = 2.90763E+21
CCPi0_POT1 = 6.91412E+20
FULLOSC_POT1 = 2.71612e+20


OVR_POT2  = 4.08963968669e+20
OVR_POT2rr  = 1.29919202522E+21
NUE_POT2  = 9.2085012316e+22
LOWE_POT2 = 7.497617079E+23
LOWM_POT2 = 2.0290756497E+21
DAT_POT2  = 0
FULLOSC_POT2 = 2.74298e+20

OVR_POT3  = 8.98773223801e+20
NUE_POT3  = 4.70704675581e+22
LOWE_POT3 = 5.97440749241e+23
LOWM_POT3 = 1.51234621011e+21
DIRT_POT3 = 1.0
OPEN_POT3 = 8.786e+18
EXT_POT3  = 39566274.0 / 2263559.0 * OPEN_POT3
NCPi0_POT3 = 2.22482E+21 + 2.64479e+20
CCPi0_POT3 = 5.91343E+20
FULLOSC_POT3 = 2.74298e+20

# Final BDT files correction
NUE_POT1 *= 190064./207548.
EXT_POT1 *= 64907./63119.
OVR_POT3 *= 314302./318046.
NUE_POT3 *= 100438./100186.


POTDICT = {1:{},2:{},3:{}}
POTDICT[1]['open'] = OPEN_POT1
POTDICT[1]['ext'] = EXT_POT1
POTDICT[1]['overlay'] = OVR_POT1rr
POTDICT[1]['overlay_lowE'] = LOWM_POT1
POTDICT[1]['intrinsics'] = NUE_POT1rr
POTDICT[1]['intrinsics_lowE'] = LOWE_POT1
POTDICT[1]['ncpi0'] = NCPi0_POT1 + OVR_POT1rr
POTDICT[1]['ccpi0'] = CCPi0_POT1 + OVR_POT1rr
POTDICT[1]['fullosc'] = FULLOSC_POT1
POTDICT[2]['overlay'] = OVR_POT2rr
POTDICT[2]['overlay_lowE'] = LOWM_POT2
POTDICT[2]['intrinsics'] = NUE_POT2
POTDICT[2]['intrinsics_lowE'] = LOWE_POT2
POTDICT[2]['fullosc'] = FULLOSC_POT2
POTDICT[3]['open'] = OPEN_POT3
POTDICT[3]['ext'] = EXT_POT3
POTDICT[3]['overlay'] = OVR_POT3
POTDICT[3]['overlay_lowE'] = LOWM_POT3
POTDICT[3]['intrinsics'] = NUE_POT3
POTDICT[3]['intrinsics_lowE'] = LOWE_POT3
POTDICT[3]['ncpi0'] = NCPi0_POT3 + OVR_POT3
POTDICT[3]['ccpi0'] = CCPi0_POT3 + OVR_POT3
POTDICT[3]['fullosc'] = FULLOSC_POT3


goodruns1 = np.loadtxt(weightdir+'pass_r1.txt')
goodruns2 = np.loadtxt(weightdir+'pass_r2.txt')
goodruns3 = np.loadtxt(weightdir+'pass_r3.txt')
goodruns = np.concatenate((goodruns1,goodruns2,goodruns3))
goodruns_dict = {1:goodruns1,2:goodruns2,3:goodruns3}

newflag = {}
newflag[1]=np.loadtxt(weightdir+'newpi0flag/2021Feb4_newhaspi0_run1_1mil.txt',delimiter=',')[:,:3].astype(int)
newflag[2]=np.loadtxt(weightdir+'newpi0flag/2021Jan26_newhaspi0_run2.txt',delimiter=',')[:,:4]
newflag[3]=np.loadtxt(weightdir+'newpi0flag/2021Jan26_newhaspi0_run3.txt',delimiter=',')[:,:3].astype(int)
newflag_low = {}
newflag_low[1]=np.loadtxt(weightdir+'newpi0flag/2021Feb05_newhaspi0_run1_lowE.txt',delimiter=',')[:,:3].astype(int)
newflag_low[2]=np.loadtxt(weightdir+'newpi0flag/2021Feb05_newhaspi0_run2_lowE.txt',delimiter=',')[:,:4]
newflag_low[3]=np.loadtxt(weightdir+'newpi0flag/2021Feb05_newhaspi0_run3_lowE.txt',delimiter=',')[:,:3].astype(int)
newflag_nue = {}
newflag_nue[1]=np.loadtxt(weightdir+'newpi0flag/2021June02_newhaspi0_run1_nue.txt',delimiter=',')[:,:3].astype(int)
newflag_nue[2]=np.loadtxt(weightdir+'newpi0flag/2021June02_newhaspi0_run2_nue.txt',delimiter=',')[:,:3].astype(int)
newflag_nue[3]=np.loadtxt(weightdir+'newpi0flag/2021June02_newhaspi0_run3_nue.txt',delimiter=',')[:,:3].astype(int)

# Helpful functions
    
def newhaspi0flag(test_r, test_s, test_e, test_enu, run, lowE=False):

    if run==2:
        if lowE:
            return np.any((np.abs(newflag_low[2]-np.array([float(test_r),float(test_s),float(test_e),test_enu]))<1e-1).all(1))
        else:
            return np.any((np.abs(newflag[2]-np.array([float(test_r),float(test_s),float(test_e),test_enu]))<1e-1).all(1))
    else:
        if lowE:
            return np.any(np.equal(newflag_low[run],[test_r,test_s,test_e]).all(1))
        else:
            return np.any(np.equal(newflag[run],[test_r,test_s,test_e]).all(1))

def haspi0(x,run,filetag):
  nuType = x.MC_parentPDG
  if newhaspi0flag(x.run, x.subrun, x.event, x.MC_energyInit, run, lowE='lowE' in filetag):
    if nuType==14 or (nuType==-14 and x.ccnc) or (abs(nuType)==12 and x.ccnc):
      return True
  return False

def truthCuts(x,filetag,lowEpatch):
  nuType = x.MC_parentPDG
  if lowEpatch and not 'lowE' in filetag:
    if x.MC_energyInit<400: return False
  if 'overlay' in filetag:
    if abs(nuType) == 12 and not x.ccnc: return False
  return True


def GetShCons(evt):
    
    EU = evt.shower1_sumQ_U*0.0155481
    EV = evt.shower1_sumQ_V*0.01586385
    EY = evt.shower1_sumQ_Y*0.01319672
    
    #EU = evt.shower1_sumQ_U*0.0139 + 31.5
    #EV = evt.shower1_sumQ_V*0.0143 + 35.7
    #EY = evt.shower1_sumQ_Y*0.0125 + 13.8
    
    return sqrt((EU-EV)**2 + (EU-EY)**2 + (EY-EV)**2)/(EY+1e-6)

def precuts(x,run,cutMode):
    if x.PassSimpleCuts == 0: return False
    if x.PassShowerReco ==0: return False
    if (x.TotPE < 20 or x.PorchTotPE > 20): return False
    if x.Proton_Edep < 50 or x.Electron_Edep < 35: return False
    if max(x.MaxShrFrac,-1) < 0.2: return False
    if GetShCons(x) > 2: return False
    if x.OpenAng < 0.5: return False
    if x.FailedBoost_1e1p: return False
    if x.Proton_ThetaReco > np.pi/2: return False
    if not x.run in goodruns: return False

    # High Energy cut, turn on/off as necessary
    #if x.Enu_1e1p > 1200: return False
        
    if cutMode==1:
        if x.Enu_1e1p < 200 or x.Enu_1e1p > 1200: return False
        if x.Lepton_PhiReco < np.pi/2 + 0.25 and x.Lepton_PhiReco > np.pi/2 - 0.25: return False
        if x.Thetas < 0.75 or x.Thetas > 2.5: return False
        if x.Q0_1e1p > 350: return False
        if x.Q3_1e1p < 250 or x.Q3_1e1p > 850: return False
        if x.ProtonPID_int_v[2]<0.1: return False  
    if cutMode==2:
        if x.Enu_1e1p < 700: return False
        #if x.Enu_1e1p > 1200: return False
    if cutMode==3:
        #if x.BDTscore_1e1p > 0.7 or x.BDTscore_1e1p < 0.01: return False
        if x.ProtonPID_int_v[2] < 0.0: return False
        if abs(x.Lepton_PhiReco-np.pi/2) < 0.25: return False
        
    return True

def postcuts(x):
    if x.GammaPID_pix_v[2]/(x.EminusPID_pix_v[2]+0.0001) > 2: return False
    if x.Electron_Edep > 100 and x.MuonPID_int_v[2] > 0.2: return False
    if x._pi0mass > 50: return False
    return True


def getLabel(x,filetag):
    
    if 'ext' in filetag: return 'EXTBNB'
    if 'data' in filetag: return 'data'
    
    nuType = x.MC_parentPDG
    mode = x.interactionType
    
    if abs(nuType)==12: 
        label = 'nue_'
        if mode in [1001]: label += 'ccqe'
        else: label += 'other'
        return label
    else: label = 'numu_'
    
    if mode in [1001]:
        label += 'ccqe'
    elif mode in [1000] and x.genieMode==10:
        label += 'mec'
    elif mode in [1003,1005,1007,1009,1010,1012,1014,1016,1017,1021,1028,1032,1079,1085]:
        label += 'pipm'
    elif mode in [1004,1006,1008,1011,1013,1015,1080,1086,1090]:
        label += 'pi0'
    else:
        label += 'other'
    
    return label

def getTopLabel(x,filetag):
    
    if 'ext' in filetag: return 'EXTBNB'
    if 'data' in filetag: return 'data'
    
    nuType = x.MC_parentPDG
    mode = x.interactionType

    label = '%i'%(x.nlepton)
    
    if abs(nuType)==12: 
        label += 'e'
        if x.nproton==1: label += '1p'
        else: label += 'X'
        return label
    else: label += 'm'
    
    if x.npi0>=1:
        label += 'Npi0'
    elif x.nproton==1:
        label += '1p'
    elif x.nproton>1:
        label += 'Np'
    else:
        label += 'X'
    
    return label


# main BDT ensemble class


class BDTensemble:

    def __init__(self,tag,BDTnumlist,oldBDT=False):

        self.tag = tag
        self.newCalib = 'newShowerCalib' in tag
        self.useEnu = not 'noEnu' in tag
        self.BDTnumlist = BDTnumlist
        self.nBDTs = len(BDTnumlist)
        bdtsavedir = '/home/kmason/fullosc/NicksCode/1e1pdata/aux/BDTWeights/'
        
        self.bdt = {1:{},2:{},3:{}}
        self.trainrse = {1:{},2:{},3:{}}
        self.valrse = {1:{},2:{},3:{}}
        self.valweight = {1:{},2:{},3:{}}

        for r in [1,2,3]:
          for b in BDTnumlist:
              try:
                  print(bdtsavedir+'BDTweights_R%i_%i.pickle'%(r,b),'rb')
                  self.bdt[r][b] = pickle.load(open(bdtsavedir+'BDTweights_R%i_%i.pickle'%(r,b),'rb'))
              except:
                  pass
              self.valweight[r][b] = {}
              self.trainrse[r][b] = {}
              self.valrse[r][b] = {}
              for filetag in ['intrinsics',
                              'overlay',
                              'ext',
                              'intrinsics_lowE',
                              'overlay_lowE',
                              'ccpi0',
                              'ncpi0',
                              'fullosc',
                              'moot']:
                  try:
                      self.trainrse[r][b][filetag] = np.loadtxt(bdtsavedir+'TrainSample_R%i_%i_%s.txt'%(r,b,filetag),dtype=int)
                      self.valrse[r][b][filetag] = np.loadtxt(bdtsavedir+'ValSample_R%i_%i_%s.txt'%(r,b,filetag),dtype=int)
                      print(len(self.valrse[r][b][filetag]),len(self.trainrse[r][b][filetag]))
                      if len(self.valrse[r][b][filetag])!=0:
                          self.valweight[r][b][filetag] = float(len(self.trainrse[r][b][filetag])+len(self.valrse[r][b][filetag])) \
                                                        / float(len(self.valrse[r][b][filetag]))
                      else:
                          self.valweight[r][b][filetag] = 1.0
                  except:
                      print("Exception")
                      self.trainrse[r][b][filetag] = np.zeros([1,3])
                      self.valrse[r][b][filetag] = np.zeros([1,3])
                      self.valweight[r][b][filetag] = 1.0

    def inference(self,tvdf,dpfdf,run,filetag):
        
        if filetag=='data': filetag='moot'
        if filetag=='fullosc': filetag='moot'
        training_varbs = tvdf.values
        rse = dpfdf[['run','subrun','event']].values
        print("length test",len(rse))
        dpfdf['nBDTs'] = int(self.nBDTs)*np.ones(len(rse))
       
        
        for b in self.BDTnumlist:
            tvweight = np.ones(len(rse))
            intrain = np.array([np.equal(x,self.trainrse[run][b][filetag]).all(1).any() for x in rse])
            inval = np.array([np.equal(x,self.valrse[run][b][filetag]).all(1).any() for x in rse])
            tvweight = np.where(intrain,0.0,tvweight)
            tvweight = np.where(inval,self.valweight[run][b][filetag],tvweight)
            sigprob = self.bdt[run][b].predict_proba(training_varbs)[:,1]
            tvweightkey = 'tvweight%i'%b
            sigprobkey = 'sigprob%i'%b
            dpfdf[tvweightkey] = tvweight
            dpfdf[sigprobkey] = sigprob

    def MakeBDTcut(self,idf,sigcut,mode,r2overlay=False):

        # Conglemerate BDT scores and weights based on strategy
        # To be run after inference
      
        bdtweight = np.zeros(idf.shape[0])
        sigprobmax = np.zeros(idf.shape[0]) 
        sigprobavg = np.zeros(idf.shape[0])
        sigprobmedian = np.zeros(idf.shape[0])
        sigproblist = np.zeros((idf.shape[0],self.nBDTs))
        notintrain = np.zeros((idf.shape[0],self.nBDTs),dtype=bool)
        numnottrain = np.zeros(idf.shape[0])
        for b in self.BDTnumlist:
            sp = idf['sigprob%i'%b]
            tvw = idf['tvweight%i'%b]
            sigprobmax = np.where(np.logical_and(tvw>0,sp>sigprobmax),sp,sigprobmax) # cut on the maximum non-train score in ensemble
            if mode == 'fracweight': 
                #bdtweight += np.where(sp>sigcut,tvw/float(self.nBDTs),0)
                bdtweight += np.where((tvw>0.1) & (sp>sigcut),1.0,0.0)
            sigprobavg += np.where(tvw>0.1,sp,0)
            numnottrain += np.where(tvw>0.1,1,0)
            sigproblist[:,b] = sp
            notintrain[:,b] = tvw > 0.1
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



def selection(t,cutMode,filetag,run,lowEpatch,ensemble,POT=0,genieDict=None,leeDict=None,verbose=True):

  
  if ensemble.useEnu: tdf = pd.DataFrame(columns=train_varb_names)
  else: tdf = pd.DataFrame(columns=train_varb_names[1:])
  dpfdf = pd.DataFrame(columns=dpf_varb_names)
  rsevdf = pd.DataFrame(columns=rsev_names)

  for i,x in enumerate(t):
    if verbose and i%100==0: print('%2.1f%% done'%(100*float(i)/float(t.GetEntries())),end='\r')
    cutLevel = 0
    if not truthCuts(x,filetag,lowEpatch): continue 
    if not precuts(x,run,cutMode): 
      if cutMode!=5: continue
    else: cutLevel = 1
    if cutMode in [0,2,5] and cutLevel>=1:
      if not postcuts(x):
        if cutMode != 5: continue
      else: cutLevel=2
    

    idxv = [x.run,x.subrun,x.event,x.vtxid]
    w = 1.0
    lw = 0.0
    if genieDict is not None: 
      for idx in [tuple((x.run,x.subrun,x.event,round(x.MC_energyInit,0))),
                  tuple((x.run,x.subrun,x.event,round(x.MC_energyInit,0)-1)),
                  tuple((x.run,x.subrun,x.event,round(x.MC_energyInit,0)+1))]:
        if idx in genieDict:
          gotIdx = True
          w = genieDict[idx]
          if leeDict is not None: lw = leeDict[idx]
          continue
          
   
    pi0flag = False
    intType = -9999
    nlepton = 0
    nproton = 0
    if filetag=='data':
      POTweight = POT
    else:
      POTweight = POTDICT[run][filetag]
      if 'overlay' in filetag: pi0flag = haspi0(x,run,filetag)
      if filetag!='ext': 
            nlepton = x.nlepton
            nproton = x.nproton
            intType = x.interactionType

    label = getLabel(x,filetag)
    
    if pi0flag:
      #label = '%i'%(1-x.ccnc)
      #if abs(x.MC_parentPDG) == 12: label += 'eX'
      #else: label += 'mNpi0'
      #if abs(x.MC_parentPDG) == 12: label = 'nue_other'
      #else: label = 'numu_pi0'
      if run in [1,3]:
        if x.ccnc: POTweight = POTDICT[run]['ncpi0']
        else: POTweight = POTDICT[run]['ccpi0']
 
    if cutMode==5:
      if cutLevel>=2:
        tvarb = pd.Series(getNewShowerCalibTrainingVarbs(x,newCalib=ensemble.newCalib,useEnu=ensemble.useEnu),index=train_varb_names)
      else: tvarb = pd.Series(np.zeros(len(train_varb_names)),index=train_varb_names) 
        
      dpfvarb = pd.Series([x.MC_energyInit,x.ccnc,x.MC_scedr,POTweight,w,lw,label,pi0flag,run,filetag,cutLevel],index=efficiency_names)
    
    else:
      if ensemble.useEnu: tvarb = pd.Series(getNewShowerCalibTrainingVarbs(x,newCalib=ensemble.newCalib),index=train_varb_names)
      else: tvarb = pd.Series(getNewShowerCalibTrainingVarbs(x,newCalib=ensemble.newCalib)[1:],index=train_varb_names[1:])
      dpfvarb = pd.Series(getNewShowerCalibDPFvarbs(x,newCalib=ensemble.newCalib) + [x.MC_energyInit,x.BDTscore_1e1p,x.ccnc,intType,nlepton,nproton,x.MC_scedr,POTweight,w,lw,label,pi0flag,x.haspi0,run,filetag,cutLevel],index=dpf_varb_names)
    
    rsev = pd.Series(idxv,index=rsev_names)
    
    tdf = tdf.append(tvarb,ignore_index=True)
    dpfdf = dpfdf.append(dpfvarb,ignore_index=True)
    rsevdf = rsevdf.append(rsev,ignore_index=True)

  print("length check",len(rsevdf))
  print("length check",len(dpfdf))
    
  dpfdf = pd.concat([rsevdf,dpfdf],axis=1)
  ensemble.inference(tdf,dpfdf,run,filetag)
  return dpfdf

def sysselection(t,cutMode,run,sample,ensemble,overlay=False,verbose=True):

  tdf = pd.DataFrame(columns=train_varb_names)
  dpfdf = pd.DataFrame(columns=dpf_varb_names)
  rsevdf = pd.DataFrame(columns=rsev_names)

  for i,x in enumerate(t):
    if verbose and i%100==0: print('%2.1f%% done'%(100*float(i)/float(t.GetEntries())),end='\r')
    if overlay and not truthCuts(x,'overlay',False): continue 
    if not precuts(x,run,cutMode): continue
    if cutMode in [0,2]:
      if not postcuts(x): continue
    
    idx = tuple((x.run,x.subrun,x.event))
    idxv = [x.run,x.subrun,x.event,x.vtxid]

    tvarb = pd.Series(getNewShowerCalibTrainingVarbs(x,newCalib=ensemble.newCalib),index=train_varb_names)
    dpfvarb = pd.Series(getNewShowerCalibDPFvarbs(x,newCalib=ensemble.newCalib) + [x.MC_energyInit,sample],index=sys_varb_names)
    rsev = pd.Series(idxv,index=rsev_names)
    
    tdf = tdf.append(tvarb,ignore_index=True)
    dpfdf = dpfdf.append(dpfvarb,ignore_index=True)
    rsevdf = rsevdf.append(rsev,ignore_index=True)

  dpfdf = pd.concat([rsevdf,dpfdf],axis=1)
  ensemble.inference(tdf,dpfdf,run,'moot')
  return dpfdf


