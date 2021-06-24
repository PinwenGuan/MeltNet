### Script for emsemble prediction using MelNet
### Output files with four kinds of prefix:
### xyp-: input features, true value of output and prediction of output
### lsys: system-specific test errors
### lt-: training error of each epoch in the training process
### l-: test error of each epoch in the training process

import glob
import os
import torch 
from torch.autograd import Variable 
import torch.nn.functional as F 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from ml import *
import torch.utils.data as Data
import numpy as np

n=400 # upper limit of total number of systems  
de=0.5 # upper limit of formation enthalpy included, in eV/atom
ts=0.2 # fraction of test set in whole set, for cross validation, this means 1/ts=5 folds
dir='cv/' # folder of training data, which is also where the results of training are stored
nens=100 # size of the emsemble
n_epoch=10 # number of training epochs

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

for j in range(int(1/ts)):
    ensdir=dir+'ens'+str(j)+'/'
    try:
        os.mkdir(ensdir)
    except:
        pass
    liq_train=[[] for jj in range(nens)]
    x00=np.loadtxt(dir+'x0-dE'+str(de)+'-ts'+str(ts)+'-'+str(j))
    x0_test=np.loadtxt(dir+'x0_test-dE'+str(de)+'-ts'+str(ts)+'-'+str(j))
    y00=np.loadtxt(dir+'y0-dE'+str(de)+'-ts'+str(ts)+'-'+str(j))
    y0_test=np.loadtxt(dir+'y0_test-dE'+str(de)+'-ts'+str(ts)+'-'+str(j))
    for jj in range(0,nens):
        liq_train[jj] = train_test_split(liq_train_all[j], test_size=0.25, random_state=jj)[0]
        g=[]
        for i in liq_train[jj]:
            g=g+(x00[:,-1] == i).nonzero()[0].tolist()
        x0=x00[g,:]
        y0=y00[g,:]
        """
        x_selection:
        0. c, 
        1. MP enthalpy, 
        2. MP convex hull enthalpy, 
        3. Miedema enthalpy, 
        4. mechanical fusion entropy, 
        5. mixing entropy, 
        6. linear melting point,
        7. fusion entropy weighted melting point
        8. VEC
        9. electronegativity difference
        10. atomic size difference
        """
        x_selection=[1,4,5,7,8,9,10]
        x=x0[:,x_selection]
        x_test=x0_test[:,x_selection]
        """
        y_selection:
        0. true melting point 
        1. true melting point-linear melting point 
        2. true melting point-fusion entropy weighted melting point
        3. equivalent enthalpy from true melting point
        """
        y_selection=[1]
        y=y0[:,y_selection]
        y_test=y0_test[:,y_selection]
        x0=torch.Tensor(x0)
        x=torch.Tensor(x)
        y0=torch.Tensor(y0)
        y=torch.Tensor(y)
        #y=torch.unsqueeze(y, dim=1)
        x0, x, y0, y= Variable(x0), Variable(x), Variable(y0), Variable(y)
        x0_test=torch.Tensor(x0_test)
        x_test=torch.Tensor(x_test)
        y0_test=torch.Tensor(y0_test)
        y_test=torch.Tensor(y_test)
        x0_test, x_test, y0_test, y_test= Variable(x0_test), Variable(x_test), Variable(y0_test), Variable(y_test)
        net = Net(n_feature=len(x_selection),n_hidden=48,n_output=1,NL_HIDDEN=3,mom=0.5,batch_normalization=True,ACTIVATION=torch.tanh) 
        torch.manual_seed(1)   
        BATCH_SIZE = 32      
        torch_dataset = Data.TensorDataset(x0,y0,x,y)
        loader = Data.DataLoader(
            dataset=torch_dataset,      
            batch_size=BATCH_SIZE,      
            shuffle=True,               
            num_workers=2,
            drop_last=True,              
        )
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01, weight_decay=0.0006)  
        loss_function = torch.nn.L1Loss()
        params = list(net.parameters())
        niter=n_epoch
        lt=[]
        l=[]
        print('Cross validation '+str(j)+':')
        print('Emsemble '+str(jj)+':')
        for t in range(niter):
            for step, (batch_x0,batch_y0,batch_x, batch_y) in enumerate(loader): 
                prediction = net(batch_x)
                loss = loss_function(prediction, batch_y)
                optimizer.zero_grad() 
                loss.backward() 
                optimizer.step()
            prediction_test = net(x_test)
            loss_test = loss_function(prediction_test, y_test) 
            lt.append(float(loss_test.data.numpy()))
            prediction = net(x)
            loss = loss_function(prediction, y) 
            l.append(float(loss.data.numpy()))
            if t%10==0:
                print(float(loss_test.data.numpy()))
        fulld=np.hstack([x0_test.data.numpy(),y_test.data.numpy(),prediction_test.data.numpy()])
        np.savetxt(ensdir+'xyp-dE'+str(de)+'-ts'+str(ts)+'-'+str(jj),fulld)
        n=np.unique(x0_test[:,-1].round().numpy())
        g=[]
        for i in n:
            g.append((x0_test[:,-1] == i).nonzero())
        loss=[0 for i in n]
        for i in range(len(g)):
            loss[i]=float(loss_function(prediction_test[g[i]], y_test[g[i]]).data.numpy())
        with open(ensdir+'lsys'+str(de)+'-ts'+str(ts)+'-'+str(jj),'w') as f:
            for i in range(len(g)):
                f.write(liq[int(n[i])]+': '+str(loss[i])+'\n')
        np.savetxt(ensdir+'lt-dE'+str(de)+'-ts'+str(ts)+'-'+str(jj),np.array(lt))
        np.savetxt(ensdir+'l-dE'+str(de)+'-ts'+str(ts)+'-'+str(jj),np.array(l))

## post processing
#------------------ plot true vs prediction melting point
nliq_plot=liq_test_all[0][::7]
liq_plot=[liq[i] for i in nliq_plot]
j=0
w=3
xyp=np.loadtxt(dir+'xyp-dE'+str(de)+'-ts'+str(ts)+'-'+str(j))
#nens=100
ensdir=dir+'ens0/'
xypens=[[] for jj in range(nens)]
for jj in range(nens):
    xypens[jj]=np.loadtxt(ensdir+'xyp-dE'+str(de)+'-ts'+str(ts)+'-'+str(jj))

for i in range(len(liq_plot)):
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams['axes.linewidth'] = 2
    plt.rcParams["font.family"] = "Arial"
    plt.subplot(int(len(liq_plot)/w),w,i+1)
    g=(xyp[:,-3] == nliq_plot[i]).nonzero()[0]
    a=xyp[g,:]
    x=[0]+a[:,0].tolist()+[1]
    x=np.array(x)
    atrue=[0]+a[:,-2].tolist()+[0]
    atrue=np.array(atrue)
    apredict=[0]+a[:,-1].tolist()+[0]
    apredict=np.array(apredict)
    f=np.loadtxt(liq_plot[i])
    f1=f[(f[:,0] == x[0]).nonzero()[0],1].tolist()[0]
    f2=f[(f[:,0] == x[-1]).nonzero()[0],1].tolist()[0]
    ytrue=(1-x)*f1+x*f2+atrue
    ypredict=(1-x)*f1+x*f2+apredict
    gens=[(jj[:,-3] == nliq_plot[i]).nonzero()[0] for jj in xypens]
    aens=[xypens[jj][gens[jj],-1] for jj in range(len(gens))]
    yave=[0]+np.average(np.array(aens),axis=0).tolist()+[0]
    yave=np.array(yave)+(1-x)*f1+x*f2
    ydev=[0]+np.std(np.array(aens),axis=0).tolist()+[0]
    ydev=np.array(ydev)
    plt.fill_between(x, yave-ydev, yave+ydev,facecolor='r',alpha=0.4)
    plt.plot(x,ytrue,label='True',color = "g")
    plt.plot(x,ypredict,'b',label='Single prediction')
    plt.plot(x,yave,'b--',label='Ensemble average')
    plt.xlim(0,1)
    plt.tick_params(axis='both',which='major',labelsize=11,width=2)
    ax=plt.gca()
    if i<len(liq_plot)-w:
        ax.set_xticklabels([])

plt.legend(frameon=False,prop={'size': 11},loc=(0.2,0.1))
plt.subplots_adjust(left=0.05, right=0.98, bottom=0.05, top=0.99, wspace=0.2, hspace=0.1)
plt.show()

#------------------ True value, single prediction, emsemble prediction
j=0 # select the first subset out of five subsets in the cross validation for calculations 
xyp=np.loadtxt(dir+'xyp-dE'+str(de)+'-ts'+str(ts)+'-'+str(j))
#nens=100
ensdir=dir+'ens'+str(j)+'/'
xypens=[[] for jj in range(nens)]
for jj in range(nens):
    xypens[jj]=np.loadtxt(ensdir+'xyp-dE'+str(de)+'-ts'+str(ts)+'-'+str(jj))

a=xyp[:,:]
x=[0]+a[:,0].tolist()+[1]
x=np.array(x)
ytrue=[0]+a[:,-2].tolist()+[0]
ytrue=np.array(ytrue)
ypredict=[0]+a[:,-1].tolist()+[0]
ypredict=np.array(ypredict)
aens=[xypens[jj][:,-1] for jj in range(len(xypens))]
yave=[0]+np.average(np.array(aens),axis=0).tolist()+[0]
yave=np.array(yave)
ydev=[0]+np.std(np.array(aens),axis=0).tolist()+[0]
ydev=np.array(ydev)

etrue=np.average(abs(ytrue))
epredict=np.average(abs(ypredict-ytrue))
eens=np.average(abs(yave-ytrue))
devens=np.average(abs(ydev))
etrue,epredict,eens,devens