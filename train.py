### Script for training MelNet
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
n_epoch=500 # number of training epochs

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
    x0=np.loadtxt(dir+'x0-dE'+str(de)+'-ts'+str(ts)+'-'+str(j))
    x0_test=np.loadtxt(dir+'x0_test-dE'+str(de)+'-ts'+str(ts)+'-'+str(j))
    y0=np.loadtxt(dir+'y0-dE'+str(de)+'-ts'+str(ts)+'-'+str(j))
    y0_test=np.loadtxt(dir+'y0_test-dE'+str(de)+'-ts'+str(ts)+'-'+str(j))
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
    x0, x, y0, y= Variable(x0), Variable(x), Variable(y0), Variable(y)
    x0_test=torch.Tensor(x0_test)
    x_test=torch.Tensor(x_test)
    y0_test=torch.Tensor(y0_test)
    y_test=torch.Tensor(y_test)
    x0_test, x_test, y0_test, y_test= Variable(x0_test), Variable(x_test), Variable(y0_test), Variable(y_test)
    net = Net(n_feature=len(x_selection),n_hidden=48,n_output=1,NL_HIDDEN=3,mom=0.5,batch_normalization=True,ACTIVATION=torch.tanh) 
    #print(net)
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
    #print (len(params))
    #print (params[0].size())
    niter=n_epoch
    lt=[]
    l=[]
    prediction_test = net(x_test)
    loss_test = loss_function(prediction_test, y_test)
    lt.append(float(loss_test.data.numpy()))
    prediction = net(x)
    loss = loss_function(prediction, y)
    l.append(float(loss.data.numpy()))
    print(float(loss_test.data.numpy()))
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
        print(float(loss_test.data.numpy()))
    fulld=np.hstack([x0_test.data.numpy(),y_test.data.numpy(),prediction_test.data.numpy()])
    np.savetxt(dir+'xyp-dE'+str(de)+'-ts'+str(ts)+'-'+str(j),fulld)
    n=np.unique(x0_test[:,-1].round().numpy())
    g=[]
    for i in n:
        g.append((x0_test[:,-1] == i).nonzero())
    loss=[0 for i in n]
    for i in range(len(g)):
        loss[i]=float(loss_function(prediction_test[g[i]], y_test[g[i]]).data.numpy())
        print(liq[int(n[i])]+': '+str(loss[i])) 
    with open(dir+'lsys'+str(de)+'-ts'+str(ts)+'-'+str(j),'w') as f:
        for i in range(len(g)):
            f.write(liq[int(n[i])]+': '+str(loss[i])+'\n')
    np.savetxt(dir+'lt-dE'+str(de)+'-ts'+str(ts)+'-'+str(j),np.array(lt))
    np.savetxt(dir+'l-dE'+str(de)+'-ts'+str(ts)+'-'+str(j),np.array(l))
