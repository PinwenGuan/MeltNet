### Script for data generation

import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from ml import *
import numpy as np

n=400 # upper limit of total number of systems  
de=0.5 # upper limit of formation enthalpy (above convex hull) included, in eV/atom
ts=0.2 # fraction of test set in whole set
dir='cv/' # folder to save data of input features
liqdir='liq' # folder storing liquidus data

os.chdir(liqdir)
liq=sorted(glob.glob('*.liq'))
os.chdir('..')
liq=liq[:n]
try:
    os.mkdir(dir)
except:
    pass

kf = KFold(n_splits=int(1/ts),shuffle=True, random_state=42)

liq_train_all=[]
liq_test_all=[]
for i in kf.split(liq):
    liq_train_all.append(i[0])
    liq_test_all.append(i[1])

aliq={}
os.chdir(liqdir)
for j in range(len(liq)):
    aliq[liq[j]]=list(gen(liq[j],dE=de))

os.chdir('..')

for j in range(int(1/ts)):
    liq_train=[liq[i] for i in liq_train_all[j]]
    liq_test=[liq[i] for i in liq_test_all[j]]
    a=[]
    for i in range(len(liq_train)):
        aa=aliq[liq_train[i]]
        aa[0]=np.hstack([aa[0],liq_train_all[j][i]*np.ones([aa[0].shape[0],1])])
        aa=tuple(aa)
        a.append(aa)
    a_test=[]
    for i in range(len(liq_test)):
        aa=aliq[liq_test[i]]
        aa[0]=np.hstack([aa[0],liq_test_all[j][i]*np.ones([aa[0].shape[0],1])])
        aa=tuple(aa)
        a_test.append(aa)
    x0=a[0][0]
    y0=a[0][1]
    for i in range(1,len(a)):
        x0=np.vstack((x0,a[i][0]))
        y0=np.vstack((y0,a[i][1]))
    x0_test=a_test[0][0]
    y0_test=a_test[0][1]
    for i in range(1,len(a_test)):
        x0_test=np.vstack((x0_test,a_test[i][0]))
        y0_test=np.vstack((y0_test,a_test[i][1]))
    np.savetxt(dir+'x0-dE'+str(de)+'-ts'+str(ts)+'-'+str(j),x0)
    np.savetxt(dir+'x0_test-dE'+str(de)+'-ts'+str(ts)+'-'+str(j),x0_test)
    np.savetxt(dir+'y0-dE'+str(de)+'-ts'+str(ts)+'-'+str(j),y0)
    np.savetxt(dir+'y0_test-dE'+str(de)+'-ts'+str(ts)+'-'+str(j),y0_test)
