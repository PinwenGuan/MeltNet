import numpy as np
import pymatgen as mg
from calf.constants import *
from pymatgen.ext.matproj import MPRester
from scipy import interpolate 
import torch 
from torch.autograd import Variable 
import torch.nn.functional as F 
import matplotlib.pyplot as plt 

"""
Function for generating data from a system
liq: the file containing liquidus data, the name should be like Al-Fe.liq
dx: composition interval
dE: if energy above convex hull is larger than dE, discard this formation energy
Return:
x:
    0. Composition, 
    1. MP (Materials Project) enthalpy, 
    2. MP convex hull enthalpy, 
    3. Miedema enthalpy, 
    4. mechanical fusion entropy, 
    5. mixing entropy, 
    6. linear melting point,
    7. fusion entropy weighted melting point
    8. VEC
    9. electronegativity difference
    10. atomic size difference
out:
    0. true melting point 
    1. true melting point-linear melting point 
    2. true melting point-fusion entropy weighted melting point
    3. equivalent enthalpy from true melting point
"""
def gen(liq,dx=0.01,dE=0.3):
    species=liq.strip('liq').strip('.').split('_')[0].split('-')
    with MPRester("QToSVCwdOf1L9pua") as m:
        q_all = m.query(criteria={"elements": {"$all": species},"nelements":2}, properties=['formula','formation_energy_per_atom','e_above_hull'])
        q=[]
        for i in range(len(q_all)):
            try:
                if q_all[i]['e_above_hull']<dE:
                    q.append(q_all[i])
            except:
                q.append(q_all[i])
        cs=[0];hms=[0]
        for i in range(len(q)):
            cs.append(q[i]['formula'][species[0]]/(q[i]['formula'][species[0]]+q[i]['formula'][species[1]]))
            hms.append(q[i]['formation_energy_per_atom']*1.6*1e-19*6.02*1e20)
        cs.append(1); hms.append(0)
        chm={}
        for i in range(len(cs)):
            chm[cs[i]]=[]
        for i in range(len(cs)):
            chm[cs[i]].append(hms[i])
        csn=list(chm.keys())
        hmsn=[min(chm[i]) for i in chm.keys()]
        hm_fit = interpolate.interp1d(csn, hmsn)
        q_0=[i for i in q_all if i['e_above_hull']==0]
        cs_0=[0];hms_0=[0]
        for i in range(len(q_0)):
            cs_0.append(q_0[i]['formula'][species[0]]/(q_0[i]['formula'][species[0]]+q_0[i]['formula'][species[1]]))
            hms_0.append(q_0[i]['formation_energy_per_atom']*1.6*1e-19*6.02*1e20)
        cs_0.append(1); hms_0.append(0)
        chm_0={}
        for i in range(len(cs_0)):
            chm_0[cs_0[i]]=[]
        for i in range(len(cs_0)):
            chm_0[cs_0[i]].append(hms_0[i])
        csn_0=list(chm_0.keys())
        hmsn_0=[min(chm_0[i]) for i in chm_0.keys()]
        hm_fit_0 = interpolate.interp1d(csn_0, hmsn_0)
    a=np.loadtxt(liq)
    a=np.array([a[i,:] for i in range(a.shape[0]) if abs((a[i,0]*1e9)%(dx*1e9))<1e-9]) 
    r={}
    chi={}
    for i in range(len(species)):
        rr=covalent_radius[species[i]]
        if type(rr)==dict:
            rr=np.average(list(rr.values()))
        r[species[i]]=rr
        chi[species[i]]=mg.Element(species[i]).X
    c={}
    c[species[0]]=a[:,0]
    c[species[1]]=1-a[:,0]
    y=a[:,1]
    y=y[1:-1]
    for i in species:
        c[i]=c[i][1:-1]
    x=[[] for i in y]
    z=[0 for i in y]
    out=[[] for i in y]
    for i in list(range(int(len(x)/2),-1,-1))+list(range(int(len(x)/2)+1,len(x))):
        vec=0
        chi_mean=0
        dchi2=0
        r_mean=0
        d2=0
        sm=0
        ds=c[species[0]][i]*fs[species[0]]+c[species[1]][i]*fs[species[1]]
        dh=c[species[0]][i]*fs[species[0]]*a[-1,1]+c[species[1]][i]*fs[species[1]]*a[0,1]
        tav1=c[species[0]][i]*a[-1,1]+c[species[1]][i]*a[0,1]   
        tav2=dh/ds
        yr1=y[i]-tav1
        yr2=y[i]-tav2
        hm_miedema=4*c[species[0]][i]*c[species[1]][i]*Hm[','.join(sorted(species))]
        hm_mp=hm_fit(c[species[0]][i])
        hm_mp_0=hm_fit_0(c[species[0]][i])
        for j in species:
            vec=vec+c[j][i]*VEC[j]
            chi_mean=chi_mean+c[j][i]*chi[j]
            r_mean=r_mean+c[j][i]*r[j]
            sm=sm-8.314*c[j][i]*np.log(c[j][i])
        for j in species:
            dchi2=dchi2+c[j][i]*(chi[j]-chi_mean)**2
            d2=d2+c[j][i]*(1-r[j]/r_mean)**2
        dchi=np.sqrt(dchi2)
        d=100*np.sqrt(d2)
        x[i]=[c[species[0]][i],hm_mp,hm_mp_0,hm_miedema,ds,sm/(8.314*np.log(2)),tav1,tav2,vec,dchi,d/100]
        if c[species[0]][i]!=0 and c[species[0]][i]!=1:
            z[i]=((tav2-y[i])*ds)/(4*c[species[0]][i]*c[species[1]][i])/1e3
        if c[species[0]][i]==0:
            z[i]=z[i+1]
        if c[species[0]][i]==1:
            z[i]=z[i-1]
        out[i]=[y[i],yr1,yr2,z[i]]
    x=np.array(x)
    out=np.array(out)
    return x,out

"""
Neural network class
n_feature: number of features
n_hidden: number of nodes in each hidden layer
n_output: number of nodes in the output layer
NL_HIDDEN: number of hidden layers
batch_normalization: whether do batch normalization
ACTIVATION: activation function
mom: momentum
"""
class Net(torch.nn.Module): 
  def __init__(self, n_feature, n_hidden, n_output, NL_HIDDEN,batch_normalization=False,ACTIVATION = torch.tanh,mom=0.5): 
    super(Net, self).__init__()
    self.do_bn = batch_normalization
    self.fcs = []   
    self.bns = []
    self.bn_input = torch.nn.BatchNorm1d(n_feature, momentum=mom)  
    self.NL_HIDDEN=int(NL_HIDDEN)
    self.ACTIVATION=ACTIVATION
    if type(n_hidden)!=list:
        n_hidden=[int(n_hidden) for i in range(self.NL_HIDDEN)]
    else:
        n_hidden=[int(i) for i in n_hidden]
    n_feature=int(n_feature);n_output=int(n_output)
    for i in range(self.NL_HIDDEN):               
      input_size = n_feature if i == 0 else n_hidden[i-1]
      fc = torch.nn.Linear(input_size, n_hidden[i])
      setattr(self, 'fc%i' % i, fc)                       
      self.fcs.append(fc)
      if self.do_bn:
        bn = torch.nn.BatchNorm1d(n_hidden[i], momentum=mom)
        setattr(self, 'bn%i' % i, bn)   
        self.bns.append(bn)
    self.predict = torch.nn.Linear(n_hidden[-1], n_output)         
  def forward(self, x):
    pre_activation = [x]
    if self.do_bn: x = self.bn_input(x)    
    layer_input = [x]
    for i in range(self.NL_HIDDEN):
      x = self.fcs[i](x)
      pre_activation.append(x)    
      if self.do_bn: x = self.bns[i](x)  
      x = self.ACTIVATION(x)
      layer_input.append(x)       
    out = self.predict(x)
    return out
