##this script set up the original trained model
##load the model
##plot corresponding figures
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
sys.path.insert(1,'/Users/yuewu/Dropbox (Edison_Lab@UGA)/Projects/Bioinformatics_modeling/nc_model/nnt/model_training/')
import mlp_struc as models

# import the model from model script
inputdir="/Users/yuewu/Dropbox (Edison_Lab@UGA)/Projects/Bioinformatics_modeling/nc_model/nnt/simulation/simulat.linear.small.mutinnt/mlp/"
os.chdir(inputdir)
from train_mlp_full_modified import train, test, parse_func_wrap, batch_sampler_block
##load information table
infortab=pd.read_csv(inputdir+'submitlist.tab',sep="\t",header=0)
infortab=infortab.astype({"batch_size": int,"test_batch_size": int})
##plot time trajectory and residue for:
# samplecombselec=[5000,8000]
# samplewholeselec=list(range(9995,10000))
# sampseletrainind=[0,4999,7999]
# sampseletestind=[0,499,999,1499,1999]
## number of samples from each group
trainsamplen=2
testsamplen=2
specind=[0,1]#[0,1,2,3,4,5]#[0,3,5,7,9]
ntime=21#101
# testselec=samplecombselec+samplewholeselec
f=h5py.File(inputdir+"data/sparselinearode_new.small.mat",'r')
data=f.get('inputstore')
Xvar=np.array(data).transpose()
rowiseq=range(1,2)# +1 than number of folders
##plot time trajectory
for rowi in rowiseq:
    #load data
    with open(inputdir+"result/"+str(rowi)+"/pickle_dimdata.dat","rb") as f1:
        dimdict=pickle.load(f1)
    
    with open(inputdir+"result/"+str(rowi)+"/pickle_inputwrap.dat","rb") as f1:
        inputwrap=pickle.load(f1)
    
    device=torch.device('cpu')
    loaddic=torch.load(inputdir+"result/"+str(rowi)+"/model_best.resnetode.tar",map_location=device)
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
    trainstep=math.floor(np.unique(samplevectrain).size/trainsamplen)
    teststep=math.floor(np.unique(samplevectest).size/testsamplen)
    sampseletrainindind=[0+i*trainstep for i in range(trainsamplen)]
    sampseletestindind=[0+i*teststep for i in range(testsamplen)]
    # sampseletrainind=samplevectrain[sampseletrainindind]
    # sampseletestind=samplevectest[sampseletestindind]
    # testind=np.sort(np.array(list(sampleind.difference(set(trainind)))))
    testselec=np.append(np.unique(samplevectrain)[sampseletrainindind],np.unique(samplevectest)[sampseletestindind])
    labels=["train"]*len(sampseletrainindind) + ["test"]*len(sampseletestindind)
    fig,ax=plt.subplots()
    for eleind, elesampe in enumerate(testselec):
        label=labels[eleind]
        showele=np.sort(np.where(np.isin(samplevec,elesampe))[0])## for each block time is in order
        Xtensortest=torch.Tensor(Xvarnorm[list(showele),:])
        Xvartest=torch.Tensor(Xvar[list(showele),:])
        Resptensortest=torch.Tensor(ResponseVar[list(showele),:])
        testdataset=utils.TensorDataset(Xtensortest,Resptensortest)
        train_sampler=None
        curr_batch_size=len(showele)#as in plotting, each time trajectory are estimated togethter(as one batch), the mini batch size is the data length
        testdataloader=utils.DataLoader(testdataset,batch_size=curr_batch_size,shuffle=False,num_workers=args.workers,pin_memory=True)
        ninnersize=int(args.layersize_ratio*dimdict["ntheta"][0])
        if bool(re.search("[rR]es[Nn]et",args.net_struct)):
            model=models.__dict__[args.net_struct](ninput=dimdict["ntheta"][0],num_response=dimdict["nspec"][0],p=args.p,ncellscale=args.layersize_ratio)
        else:
            model=models.__dict__[args.net_struct](ninput=dimdict["ntheta"][0],num_response=dimdict["nspec"][0],nlayer=args.num_layer,p=args.p,ncellscale=args.layersize_ratio,batchnorm_flag=(args.batchnorm_flag is 'Y'))
        model=torch.nn.DataParallel(model)
        model.load_state_dict(loaddic['state_dict'])
        model.eval()
        [data,target]=next(iter(testdataloader))
        torch.no_grad()
        output=model(data)
        output=output.detach()
        residue=output-target
        time=np.array(Xvartest[:,-1])
        # timeind=np.argsort(time,kind='mergesort')
        # time=time[timeind]
        for specele in specind:
            targetvec=np.array(target[:,specele])
            outputvec=np.array(output[:,specele])
            # targetvec=targetvec[timeind]
            # outputvec=outputvec[timeind]
            # fig,ax=plt.subplots()
            line1,=ax.plot(time,targetvec,label='simualted value')
            line2,=ax.plot(time,outputvec,label='estimated value')
            ax.legend()
            plt.savefig(inputdir+"result/test_"+str(elesampe)+"_spec_"+str(specele)+"_"+str(rowi)+"_"+label+".pdf")
            # plt.close(fig)
            plt.cla()
            # fig,ax=plt.subplots()
            line,=ax.plot(time,targetvec-outputvec,label='residue')
            ax.legend()
            plt.savefig(inputdir+"result/test"+str(elesampe)+"_spec_"+str(specele)+"_"+str(rowi)+"_"+label+"_residue.pdf")
            # plt.close(fig)
            plt.cla()

##plot residule for each run(different structure and hyperparameters)
###ALL run(structure+hyperparameters)+ ALL sweepind+ ALL speceid
fig,ax=plt.subplots()
for rowi in rowiseq:
    # load the pickle
    name=infortab.iloc[rowi-1,0]
    with open(inputdir+"result/"+str(rowi)+"/pickle_dimdata.dat","rb") as f1:
        dimdict=pickle.load(f1)
    
    with open(inputdir+"result/"+str(rowi)+"/pickle_inputwrap.dat","rb") as f1:
        inputwrap=pickle.load(f1)
    
    loaddic=torch.load(inputdir+"result/"+str(rowi)+"/model_best.resnetode.tar",map_location=device)
    args=loaddic["args_input"]
    Xvarnorm=inputwrap["Xvarnorm"]
    ResponseVar=inputwrap["ResponseVar"]
    samplevec=inputwrap["samplevec"]
    Xtensortest=torch.Tensor(Xvarnorm)
    Resptensortest=torch.Tensor(ResponseVar)
    testdataset=utils.TensorDataset(Xtensortest,Resptensortest)
    # train_sampler=None
    test_sampler=batch_sampler_block(testdataset,samplevec,nblock=trainsamplen)
    testdataloader=utils.DataLoader(testdataset,shuffle=False,num_workers=args.workers,pin_memory=True,batch_sampler=test_sampler)
    ninnersize=int(args.layersize_ratio*dimdict["ntheta"][0])
    if bool(re.search("[rR]es[Nn]et",args.net_struct)):
        model=models.__dict__[args.net_struct](ninput=dimdict["ntheta"][0],num_response=dimdict["nspec"][0],p=args.p,ncellscale=args.layersize_ratio)
    else:
        model=models.__dict__[args.net_struct](ninput=dimdict["ntheta"][0],num_response=dimdict["nspec"][0],nlayer=args.num_layer,p=args.p,ncellscale=args.layersize_ratio,batchnorm_flag=(args.batchnorm_flag is 'Y'))
    model=torch.nn.DataParallel(model)
    model.load_state_dict(loaddic['state_dict'])
    model.eval()
    targetvec=np.empty([0,dimdict["nspec"][0]])
    outputvec=np.empty([0,dimdict["nspec"][0]])
    timevec=[]
    torch.no_grad()
    for data, target in testdataloader:
        output=model(data)
        output=output.detach()
        timevec=timevec+data[:,-1].tolist()
        targetvec=np.concatenate([targetvec,np.array(target)])
        outputvec=np.concatenate([outputvec,np.array(output)])
        # targetvec=targetvec+target.tolist()[0]
        # outputvec=outputvec+output.tolist()[0]
    
    # targetvec=np.array(targetvec)
    # outputvec=np.array(outputvec)
    residuevec=targetvec-outputvec
    timevec=np.array(timevec)
    residuemean=np.zeros(ntime)
    for indmean, timepoint in enumerate(np.sort(np.unique(timevec),kind='mergesort')):
        ind=np.where(timevec==timepoint)
        residuemean[indmean]=np.mean(residuevec[ind])
    
    line,=ax.plot(range(0,ntime),residuemean,label='residue')
    ax.legend()
    plt.savefig(inputdir+"result/test"+name+"residue.pdf")
    plt.cla()

##extrapolation
