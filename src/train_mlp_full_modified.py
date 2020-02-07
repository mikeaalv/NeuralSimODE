##this training script support
###training mlp version of resnet on one gpu node (multiple gpu might be involved)
###multiple node is not supported yet
###separation of training vs testing, and got minibatching is intended to be in blocks of each whole time-series
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
from nltk import flatten
import re

import torch
import torch.nn.parallel
import torch.utils.data as utils
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.sampler import Sampler

# sys.path.insert(1,'/Users/yuewu/Dropbox (Edison_Lab@UGA)/Projects/Bioinformatics_modeling/nc_model/nnt/model_training/')
# import resnet_mlp as models
# sys.path.insert(1,'/home/mikeaalv/method/nnt_str/nc_model/nnt/model_training/')
import mlp_struc as models

model_names=sorted(name for name in models.__dict__
    if name.endswith("_mlp") and callable(models.__dict__[name]))
# samplewholeselec=list(range(9995,10000))## the whole time series just for testing
##default parameters
args_internal_dict={
    "batch_size": (50000,int),
    "test_ratio": (0.2,float), # ratio of sample for test in each epoch
    "test_batch_size": (50000,int),
    "epochs": (10,int),
    "learning_rate": (0.01,float),
    "momentum": (0.5,float),
    "no_cuda": (False,bool),
    "seed": (1,int),
    "log_interval": (10,int),
    "net_struct": ("resnet18_mlp",str),
    "layersize_ratio": (1.0,float),#use the input vector size to calcualte hidden layer size
    "optimizer": ("adam",str),##adam
    "normalize_flag": ("Y",str),#whether the input data in X are normalized Y normalized N not
    "batchnorm_flag": ("Y",str),# whether batch normalization is used Y yes N no. Not working for resnet
    "num_layer": (0,int),#number of layer, not work for resnet
    "timetrainlen": (101,int), #the length of time-series to use in training
    "inputfile": ("sparselinearode_new.small.stepwiseadd.mat",str),## the file name of input data
     "p": (0.0,float),
     "gpu_use": (1,int)# whehter use gpu 1 use 0 not use
}
###fixed parameters: for communication related parameter within one node
fix_para_dict={#"world_size": (1,int),
               # "rank": (0,int),
               # "dist_url": ("env://",str),#"tcp://127.0.0.1:FREEPORT"
               "gpu": (None,int),
               # "multiprocessing_distributed": (False,bool),
               # "dist_backend": ("nccl",str), ##the preferred way approach of parallel gpu
               "workers": (1,int)
}
inputdir="../data/"
def train(args,model,train_loader,optimizer,epoch,device,ntime):
    model.train()
    trainloss=[]
    for batch_idx, (data, target) in enumerate(train_loader):
        # print("checkerstart")
        # if args.gpu is not None:
        #     data=data.cuda(args.gpu,non_blocking=True)
        #
        # target=target.cuda(args.gpu,non_blocking=True)
        data,target=data.to(device),target.to(device)
        output=model(data)
        # loss=F.nll_loss(output,target)
        loss=F.mse_loss(output,target,reduction='mean')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval==0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss(per sample): {:.6f}'.format(
                epoch,batch_idx*len(data),len(train_loader.dataset),
                100. * batch_idx*len(data)/len(train_loader.dataset),loss.item()*ntime))
                
        trainloss.append(loss.item())
    
    return (sum(trainloss)/len(trainloss))*ntime

def test(args,model,test_loader,device,ntime):
    model.eval()
    test_loss=[]
    with torch.no_grad():
        for data, target in test_loader:
            # if args.gpu is not None:
            #     data=data.cuda(args.gpu,non_blocking=True)
            # target=target.cuda(args.gpu,non_blocking=True)
            data,target=data.to(device),target.to(device)
            output=model(data)
            # test_loss += F.nll_loss(output,target,reduction='sum').item() # sum up batch loss
            test_loss.append(F.mse_loss(output,target,reduction='mean').item()) # sum
    test_loss_mean=sum(test_loss)/len(test_loss)
    print('\nTest set: Average loss (per sample): {:.4f}\n'.format(test_loss_mean*ntime))
    return test_loss_mean*ntime

def parse_func_wrap(parser,termname,args_internal_dict):
    commandstring='--'+termname.replace("_","-")
    defaulval=args_internal_dict[termname][0]
    typedef=args_internal_dict[termname][1]
    parser.add_argument(commandstring,type=typedef,default=defaulval,
                        help='input '+str(termname)+' for training (default: '+str(defaulval)+')')
    
    return(parser)

def save_checkpoint(state,is_best,is_best_train,filename='checkpoint.resnetode.tar'):
    torch.save(state,filename)
    if is_best:
        shutil.copyfile(filename,'model_best.resnetode.tar')
    
    if is_best_train:
        shutil.copyfile(filename,'model_best_train.resnetode.tar')

class batch_sampler_block(Sampler):
    """
    user defined sampler to make sure random sampling blocks
    default replacement=FALSE
    default drop_last=FALSE
    """
    def __init__(self,datasource,blocks,nblock=1):
        """
            datasource: data set
            blocks: block list
            nblocks: number of block for each batch

         EX code:
         datasource=np.array([0,1,2,3,4,5,6,7,8,9])
         blocks=np.array([0,0,1,1,2,2,3,3,4,4])
         nblocks=2
         list(batch_sampler_block(datasource,blocks,nblock=nblocks))
        """
        self.datasource=datasource
        self.blocks=blocks
        self.nblock=nblock
        self.uniblocks=np.unique(self.blocks)
        self.blocksize=int(len(self.datasource)/len(self.uniblocks))
    
    def __iter__(self):
        n=len(self.uniblocks)
        indreorder=torch.randperm(n).tolist()
        batch=[]
        for idx in indreorder:
            indnew=list(range(idx*self.blocksize,(idx+1)*self.blocksize))
            batch.append(indnew)
            if len(batch)==self.nblock: ##number of block in each minibatch
                yield flatten(batch)
                batch=[]
        
        if len(batch) > 0:
            yield flatten(batch)
    
    def __len__(self):
        return self.nblock*self.blocksize

def main():
    # Training settings load-in through command line
    parser=argparse.ArgumentParser(description='PyTorch Example')
    for key in args_internal_dict.keys():
        parser=parse_func_wrap(parser,key,args_internal_dict)
    
    for key in fix_para_dict.keys():
        parser=parse_func_wrap(parser,key,fix_para_dict)
    
    args=parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic=True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    
    # args.distributed=args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node=torch.cuda.device_count()
    # ngpus_per_node=1
    # Since we have ngpus_per_node processes per node, the total world_size
    # needs to be adjusted accordingly
    ## args.world_size=ngpus_per_node*args.world_size
    # Use torch.multiprocessing.spawn to launch distributed processes: the
    # main_worker process function
    ## arg_transf=copy.deepcopy(args)
    ## arg_transf.ngpus_per_node=ngpus_per_node
    ## mp.spawn(main_worker,nprocs=ngpus_per_node,args=(ngpus_per_node,args))
    main_worker(args.gpu,ngpus_per_node,args)

def main_worker(gpu,ngpus_per_node,args):
    global best_acc1
    # args.gpu=gpu
    # if args.gpu is not None:
    #     print("Use GPU: {} for training".format(args.gpu))
    
    # For multiprocessing distributed training, rank needs to be the
    # global rank among all the processes
    ## args.rank=args.rank*ngpus_per_node+gpu
    ## dist.init_process_group(backend=args.dist_backend,init_method="env://",#args.dist_url,
    ## world_size=args.world_size,rank=args.rank)
    
    ##read the matlab matrix as Xvar and ResponseVar
    inputfile=args.inputfile
    f=h5py.File(inputdir+inputfile,'r')
    data=f.get('inputstore')
    Xvar=np.array(data).transpose()
    data=f.get('outputstore')
    ResponseVar=np.array(data).transpose()
    data=f.get('samplevec')
    samplevec=np.array(data)
    samplevec=np.squeeze(samplevec.astype(int)-1)##block index
    data=f.get('parastore')
    parastore=np.array(data)##omega normalizer
    data=f.get('nthetaset')
    nthetaset=int(np.array(data)[0][0])##block number
    data=f.get('ntime')
    ntimetotal=int(np.array(data)[0][0])##time seq including [training part, extrapolation part]
    f.close()
    ntime=args.timetrainlen
    # ResponseVarnorm=(ResponseVar-ResponseVar.mean(axis=0))/ResponseVar.std(axis=0)
    ResponseVarnorm=ResponseVar## the response variable was originally scale by omega {scaling} but not centered. and not more normalization will be done
    ##separation of train and test set
    nsample=(Xvar.shape)[0]
    ntheta=(Xvar.shape)[1]
    nspec=(ResponseVarnorm.shape)[1]
    simusamplevec=np.unique(samplevec)
    ##test block number {each block is composed for multiple samples}
    numsamptest=math.floor((simusamplevec.__len__())*args.test_ratio)
    # testsize=numsamptest*ntimetotal##test sample number
    sampleind=set(range(0,nsample))
    simusampeind=set(range(0,nthetaset))
    ## a preset whole time range for just testing
    # samplewholeselec=np.sort(-np.unique(samplevec))[0:5]
    simusamplevec=random.sample(set(simusampeind),numsamptest)
    ##index or training and testing data
    testind=np.sort(np.where(np.isin(samplevec,simusamplevec)))[0]
    trainind=np.sort(np.array(list(sampleind.difference(set(testind)))))
    # simusamplevec=random.sample(set(sampleind)-set(testaddon),testsize-testaddon.__len__())
    # testaddon=np.where(np.isin(simusamplevec,samplewholeselec))[0]
    # testaddon=np.where(np.isin(samplevec,samplewholeselec))[0]
    # testrand=random.sample(set(sampleind)-set(testaddon),testsize-testaddon.__len__())
    # testind=set(testrand)|set(testaddon)
    ntrainset=nthetaset-numsamptest
    ntestset=numsamptest
    ##training block index (time range)
    traintimeind=np.tile(np.concatenate((np.repeat(1,ntime),np.repeat(0,ntimetotal-ntime))),ntrainset)
    testtimeind=np.tile(np.concatenate((np.repeat(1,ntime),np.repeat(0,ntimetotal-ntime))),ntestset)
    train_in_ind=trainind[traintimeind==1]
    test_in_ind=testind[testtimeind==1]
    train_extr_ind=trainind[traintimeind==0]
    test_extr_ind=testind[testtimeind==0]
    ##train and test block ind
    samplevectrain=samplevec[train_in_ind]
    samplevectest=samplevec[test_in_ind]
    
    Xvartrain=Xvar[list(trainind),:]
    Xvartest=Xvar[list(testind),:]
    Xvarnorm=np.empty_like(Xvar)
    if args.normalize_flag is 'Y':
        ##the normalization if exist should be after separation of training and testing data to prevent leaking
        ##normalization (X-mean)/sd
        ##normalization include time. Train and test model need to have at least same range or same mean&sd for time
        Xvartrain_norm=(Xvartrain-Xvartrain.mean(axis=0))/Xvartrain.std(axis=0)
        Xvartest_norm=(Xvartest-Xvartest.mean(axis=0))/Xvartest.std(axis=0)
        Xvarnorm[list(trainind),:]=np.copy(Xvartrain_norm)
        Xvarnorm[list(testind),:]=np.copy(Xvartest_norm)
    else:
        Xvartrain_norm=Xvartrain
        Xvartest_norm=Xvartest
        Xvarnorm=np.copy(Xvar)
    
    #samplevecXX repeat id vector, XXind index vector
    inputwrap={"Xvarnorm": (Xvarnorm),
        "ResponseVar": (ResponseVar),
        "trainind": (trainind),
        "testind": (testind),
        "train_in_ind": (train_in_ind),
        "test_in_ind": (test_in_ind),
        "train_extr_ind": (train_extr_ind),
        "test_extr_ind": (test_extr_ind),
        # "testrand": (testrand),
        # "testaddon": (testaddon),
        "samplevec": (samplevec),
        # "samplewholeselec": (samplewholeselec),
        "samplevectrain": (samplevectrain),
        "samplevectest": (samplevectest),
        # "Xvarmean": (Xvarmean),## these two value: Xvarmean, Xvarstd can be used for "new" test data not used in the original normalization
        # "Xvarstd": (Xvarstd),
        "train_extr_ind": (train_extr_ind), ## the extrapolation point index on testing set
        "test_extr_ind": (test_extr_ind), ## the extrapolation point index on training set
        "inputfile": (inputfile),
        "ngpus_per_node": (ngpus_per_node),## number of gpus
        "numsamptest": (numsamptest),#number of testing samples
        "traintimeind": (traintimeind),
        "testtimeind": (testtimeind)
    }
    with open("pickle_inputwrap.dat","wb") as f1:
        pickle.dump(inputwrap,f1,protocol=4)##protocol=4 if there is error: cannot serialize a bytes object larger than 4 GiB
    
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
    #     shuffle=False,num_workers=args.workers,pin_memory=True,sampler=test_sampler)
    traindataloader=utils.DataLoader(traindataset,num_workers=args.workers,pin_memory=True,batch_sampler=train_sampler)
    testdataloader=utils.DataLoader(testdataset,num_workers=args.workers,pin_memory=True,batch_sampler=test_sampler)
    ninnersize=int(args.layersize_ratio*ntheta)
    ##store data
    with open("pickle_traindata.dat","wb") as f1:
        pickle.dump(traindataloader,f1,protocol=4)
    
    with open("pickle_testdata.dat","wb") as f2:
        pickle.dump(testdataloader,f2,protocol=4)
    
    dimdict={
        "nsample": (nsample,int),
        "ntheta": (ntheta,int),
        "nspec": (nspec,int),
        "ninnersize": (ninnersize,int)
    }
    with open("pickle_dimdata.dat","wb") as f3:
        pickle.dump(dimdict,f3,protocol=4)
    
    ##free up some space (not currently set)
    ##create model
    if bool(re.search("[rR]es[Nn]et",args.net_struct)):
        model=models.__dict__[args.net_struct](ninput=ntheta,num_response=nspec,p=args.p,ncellscale=args.layersize_ratio)
    else:
        model=models.__dict__[args.net_struct](ninput=ntheta,num_response=nspec,nlayer=args.num_layer,p=args.p,ncellscale=args.layersize_ratio,batchnorm_flag=(args.batchnorm_flag is 'Y'))
    
    # model.eval()
    # if args.gpu is not None:
    #     torch.cuda.set_device(args.gpu)
    #     model.cuda(args.gpu)
    #     # When using a single GPU per process and per
    #     # DistributedDataParallel, we need to divide the batch size
    #     # ourselves based on the total number of GPUs we have
    #     args.batch_size=int(args.batch_size/ngpus_per_node)
    #     args.workers=int((args.workers+ngpus_per_node-1)/ngpus_per_node)
    #     model=torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    # else:
    # DistributedDataParallel will divide and allocate batch_size to all
    # available GPUs if device_ids are not set
    
    # model=torch.nn.DataParallel(model).cuda()
    model=torch.nn.DataParallel(model)
    if args.gpu_use==1:
        device=torch.device("cuda:0")#cpu
    else:
        device=torch.device("cpu")
    
    model.to(device)
    if args.optimizer=="sgd":
        optimizer=optim.SGD(model.parameters(),lr=args.learning_rate,momentum=args.momentum)
    elif args.optimizer=="adam":
        optimizer=optim.Adam(model.parameters(),lr=args.learning_rate)
    elif args.optimizer=="nesterov_momentum":
        optimizer=optim.SGD(model.parameters(),lr=args.learning_rate,momentum=args.momentum,nesterov=True)
    
    cudnn.benchmark=True
    ##model training
    for epoch in range(1,args.epochs+1):
        acctr=train(args,model,traindataloader,optimizer,epoch,device,ntime)
        acc1=test(args,model,testdataloader,device,ntime)
        # test(args,model,traindataloader,device,ntime) # to record the performance on training sample with model.eval()
        if epoch==1:
            best_acc1=acc1
            best_train_acc=acctr
        
        # is_best=acc1>best_acc1
        is_best=acc1<best_acc1
        is_best_train=acctr<best_train_acc
        # best_acc1=max(acc1,best_acc1)
        best_acc1=min(acc1,best_acc1)
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
        # device=torch.device('cpu')
        # # model=TheModelClass(*args, **kwargs)
        # model.load_state_dict(torch.load('./1/checkpoint.resnetode.tar',map_location=device))

if __name__ == '__main__':
    main()
