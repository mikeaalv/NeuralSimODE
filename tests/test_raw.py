## this script load storage files of running model and test on continuing running
## Test: whether similar MSE can be produced
## for both cpu and gpu
import argparse
import os
import random
import shutil
import time
import warnings
import pickle
# import feather
import numpy as np
import math
import sys
import copy
import h5py
import re
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn.parallel
import torch.utils.data as utils
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
import torch.nn.functional as F
import mlp_struc as models
from train_mlp_full_modified import train, test, parse_func_wrap, batch_sampler_block, save_checkpoint
rowiseq=range(1,3)# +1 than number of folders
devstr="cpu"#'cuda:0' cpu
device=torch.device(devstr)#cuda:0
loc="mac"#mac sever
rowi=1
ntime=101

if loc=="mac":
    inputdir="/Users/yuewu/Dropbox (Edison_Lab@UGA)/Projects/Bioinformatics_modeling/nc_model/nnt/simulation/simulat.overfittest.corr/"
    with open(inputdir+"result/"+str(rowi)+"/pickle_dimdata.dat","rb") as f1:
        dimdict=pickle.load(f1)
    
    with open(inputdir+"result/"+str(rowi)+"/pickle_inputwrap.dat","rb") as f1:
        inputwrap=pickle.load(f1)
    
    loaddic=torch.load(inputdir+"result/"+str(rowi)+"/model_best_train.resnetode.tar",map_location=device)
    # loaddic=torch.load("model_best_train.resnetode.tar",map_location=device)
    f=h5py.File(inputdir+"data/sparselinearode_new.small.stepwiseadd16.mat",'r')
    data=f.get('inputstore')
    Xvar=np.array(data).transpose()
    
else:
    inputdir="/pylon5/mc5fscp/mikeaalv/test_run/simulat_overfittest_corr/"
    with open(inputdir+str(rowi)+"/pickle_dimdata.dat","rb") as f1:
        dimdict=pickle.load(f1)
    
    with open(inputdir+str(rowi)+"/pickle_inputwrap.dat","rb") as f1:
        inputwrap=pickle.load(f1)
    
    loaddic=torch.load(inputdir+str(rowi)+"/model_best_train.resnetode.tar",map_location=device)
    f=h5py.File(inputdir+"data/sparselinearode_new.small.stepwiseadd16.mat",'r')
    data=f.get('inputstore')
    Xvar=np.array(data).transpose()

loaddic['best_acctr']
loaddic['best_acc1']
args=loaddic["args_input"]
Xvarnorm=inputwrap["Xvarnorm"]
ResponseVar=inputwrap["ResponseVar"]
samplevec=inputwrap["samplevec"]
# train_in_ind=inputwrap["train_in_ind"]
# test_in_ind=inputwrap["test_in_ind"]
nsample=dimdict["nsample"][0]
sampleind=set(range(0,nsample))
samplevectrain=inputwrap["samplevectrain"]
samplevectest=inputwrap["samplevectest"]
if bool(re.search("[rR]es[Nn]et",args.net_struct)):
    model=models.__dict__[args.net_struct](ninput=dimdict["ntheta"][0],num_response=dimdict["nspec"][0],p=args.p,ncellscale=args.layersize_ratio)
else:
    model=models.__dict__[args.net_struct](ninput=dimdict["ntheta"][0],num_response=dimdict["nspec"][0],nlayer=args.num_layer,p=args.p,ncellscale=args.layersize_ratio,batchnorm_flag=(args.batchnorm_flag is 'Y'))

model=torch.nn.DataParallel(model)
model.load_state_dict(loaddic['state_dict'])
train_in_ind=inputwrap['train_in_ind']
test_in_ind=inputwrap['test_in_ind']
Xtensortrain=torch.Tensor(Xvarnorm[list(train_in_ind),:])
Resptensortrain=torch.Tensor(ResponseVar[list(train_in_ind),:])
Xtensortest=torch.Tensor(Xvarnorm[list(test_in_ind),:])
Resptensortest=torch.Tensor(ResponseVar[list(test_in_ind),:])
traindataset=utils.TensorDataset(Xtensortrain,Resptensortrain)
testdataset=utils.TensorDataset(Xtensortest,Resptensortest)
# train_sampler=torch.utils.data.distributed.DistributedSampler(traindataset)
nblocktrain=int(args.batch_size/ntime)
# nblocktest=int(args.test_batch_size/ntime)
train_sampler=batch_sampler_block(traindataset,samplevectrain,nblock=nblocktrain)
test_sampler=batch_sampler_block(testdataset,samplevectest,nblock=nblocktrain)
# traindataloader=utils.DataLoader(traindataset,batch_size=args.batch_size,
#     shuffle=(train_sampler is None),num_workers=args.workers,pin_memory=True,sampler=train_sampler)
#
# testdataloader=utils.DataLoader(testdataset,batch_size=args.test_batch_size,
#     shuffle=False,num_workers=args.workers,pin_memory=True,sampler=test_sampler)\
if devstr=='cpu':
    workid=0
else:
    workid=args.workers
    model.to(device)

traindataloader=utils.DataLoader(traindataset,num_workers=workid,pin_memory=True,batch_sampler=train_sampler)
testdataloader=utils.DataLoader(testdataset,num_workers=workid,pin_memory=True,batch_sampler=test_sampler)
optimizer=optim.Adam(model.parameters(),lr=args.learning_rate)
acc1=test(args,model,traindataloader,device,ntime)
acc2=test(args,model,testdataloader,device,ntime)
for epoch in range(1,args.epochs+1):
    acctr=train(args,model,traindataloader,optimizer,epoch,device,ntime)
    acc3=test(args,model,traindataloader,device,ntime)# to make this works as the real training mse model.eval need to be selected in train to close batchnorm
    acc4=test(args,model,testdataloader,device,ntime)
    if epoch==1:
        best_acc1=acc4
        best_train_acc=acctr
    
    # is_best=acc1>best_acc1
    is_best=acc4<best_acc1
    is_best_train=acctr<best_train_acc
    # best_acc1=max(acc1,best_acc1)
    best_acc1=min(acc4,best_acc1)
    best_train_acc=min(acctr,best_train_acc)
    save_checkpoint({
        'epoch': epoch,
        'arch': args.net_struct,
        'state_dict': model.state_dict(),
        'best_acc1': best_acc1,
        'best_acctr': best_train_acc,
        'optimizer': optimizer.state_dict(),
        'args_input': args,
    },is_best,is_best_train)
