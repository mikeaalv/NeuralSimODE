#this is a reimplementation of a few MLP structure
# 1. the ResNet published in pytorch/vision. A MLP type of setting was fomulated rather than those with CONV layers
# 2. regular MLP structure
#Yue Wu
#UGA 09/06/2019
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# from .utils import load_state_dict_from_url

__all__=['ResNet_mlp','resnet10_mlp','resnet14_mlp','resnet18_mlp', 'resnet34_mlp', 'resnet50_mlp', 'resnet101_mlp','resnet152_mlp','resnet2x_mlp','wide_resnet50_2_mlp', 'wide_resnet101_2_mlp' 'mlp_mod' 'gru_mlp_rnn' 'gru_rnn' 'diffaddcell_rnn'] #'resnext50_32x4d', 'resnext101_32x8d',

##currently no convolution layers
# def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
#     """3x3 convolution with padding"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                      padding=dilation, groups=groups, bias=False, dilation=dilation)

def line1d(in_features,out_features):
    return nn.Linear(in_features,out_features,bias=False)

def line1dbias(in_features,out_features):
    return nn.Linear(in_features,out_features,bias=True)

##the block structure general for resnet
class BasicBlock(nn.Module):
    expansion=1
    __constants__=['downsample']

    def __init__(self,inplanes,planes,downsample=None,groups=1,
                 base_width=64,norm_layer=None,p=0.0):
        # inplanes: input size
        # planes: internal size
        # downsample: whehter downsample the idnetify mapping. default: None
        # groups: was used for "Aggregated Residual Transformation" (not used currently) Defualt 1
        # width_per_group: used for "Wide Residual Networks" and "Aggregated Residual Transformation" Defualt 64
        # norm_layer: used to specify batch normalization function. Default None
        # p: dropbout probability Default 0.0
        super(BasicBlock,self).__init__()
        if norm_layer is None:
            norm_layer=nn.BatchNorm1d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        self.fc1=line1d(inplanes,planes)
        self.bn1=norm_layer(planes)
        self.relu=nn.ReLU(inplace=True)
        self.fc2=line1d(planes,inplanes)
        self.bn2=norm_layer(inplanes)
        self.downsample=downsample
        self.p=p

    def forward(self,x):
        identity=x
        out=F.dropout(self.fc1(x),training=self.training,p=self.p)
        out=self.bn1(out)
        out=self.relu(out)
        out=F.dropout(self.fc2(out),training=self.training,p=self.p)
        out=self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out=self.relu(out)
        return out

##the block structure bottleneck  for resnet
class Bottleneck(nn.Module):
    expansion=4

    def __init__(self,inplanes,planes,downsample=None,groups=1,
                 base_width=64,norm_layer=None,p=0.0):
        # inplanes: input size
        # planes: output size
        # downsample: whehter downsample the idnetify mapping. default: None
        # groups: was used for "Aggregated Residual Transformation" (not used currently) Defualt 1
        # width_per_group: used for "Wide Residual Networks" and "Aggregated Residual Transformation" Defualt 64
        # norm_layer: used to specify batch normalization function. Default None
        # p: dropbout probability Default 0.0
        super(Bottleneck,self).__init__()
        if norm_layer is None:
            norm_layer=nn.BatchNorm1d
        width=int(planes*(base_width/64./self.expansion))*groups
        self.fc1=line1d(inplanes,width)
        self.bn1=norm_layer(width)
        self.fc2=line1d(width,width)
        self.bn2=norm_layer(width)
        self.fc3=line1d(width,inplanes)
        self.bn3=norm_layer(inplanes)
        self.relu=nn.ReLU(inplace=True)
        self.downsample=downsample
        self.p=p

    def forward(self, x):
        identity=x
        out=F.dropout(self.fc1(x),training=self.training,p=self.p)
        out=self.bn1(out)
        out=self.relu(out)
        out=F.dropout(self.fc2(out),training=self.training,p=self.p)
        out=self.bn2(out)
        out=self.relu(out)
        out=F.dropout(self.fc3(out),training=self.training,p=self.p)
        out=self.bn3(out)

        if self.downsample is not None:
            identity=self.downsample(x)

        out += identity
        out=self.relu(out)

        return out

##the block structure general for mlp
class BasicBlock_mlp(nn.Module):
    expansion=1
    __constants__=['downsample']

    def __init__(self,inplanes,planes,batchnorm_flag=True,p=0.0):
        # inplanes: input size
        # planes: internal size
        # norm_layer: used to specify batch normalization function. Default None
        # p: dropbout probability Default 0.0
        super(BasicBlock_mlp,self).__init__()
        self.fc=line1d(inplanes,planes)
        if batchnorm_flag is True:
            self.bn=nn.BatchNorm1d(planes)
        else:
            self.bn=None
        self.relu=nn.ReLU(inplace=True)
        self.p=p

    def forward(self,x):
        identity=x
        out=F.dropout(self.fc(x),training=self.training,p=self.p)
        if self.bn is not None:
            out=self.bn(out)
        out=self.relu(out)
        return out

###the whole residule network structure
class ResNet_mlp(nn.Module):

    def __init__(self,block,layers,ninput,num_response,ncellscale,zero_init_residual=False,
                 groups=1,width_per_group=64,norm_layer=None,p=0.0):
        # block: block structure
        # layers: #layers,
        # ninput: #input,
        # num_response: #response,
        # ncellscale: scale factor for hidden layer size
        # zero_init_residual: whether initialize weight as 0. default False,
        # groups: was used for "Aggregated Residual Transformation" (not used currently) Defualt 1
        # width_per_group: used for "Wide Residual Networks" and "Aggregated Residual Transformation" Defualt 64
        # norm_layer: used to specify batch normalization function. Default None
        # p: dropbout probability Default 0.0
        super(ResNet_mlp,self).__init__()
        if norm_layer is None:
            norm_layer=nn.BatchNorm1d
        self._norm_layer=norm_layer#passed to block function
        self.inplanes=ninput## the input layer size
        self.width=int(ninput*ncellscale)#the default hiddne layer #neuron (width) is input dimension * scale factor
        self.groups=groups
        self.base_width=width_per_group
        self.fc1=line1d(self.inplanes,self.width)
        self.bn1=norm_layer(self.width)
        self.relu=nn.ReLU(inplace=True)
        self.layer1=self._make_layer(block,self.width,self.width,layers[0])
        # self.avgpool=nn.AdaptiveAvgPool2d((1, 1))
        self.fcf=nn.Linear(self.width,num_response)
        self.p=p
        ##paramter initilaization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight,mode='fan_out')
            elif isinstance(m, (nn.BatchNorm1d,nn.GroupNorm)):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m,Bottleneck):
                    nn.init.constant_(m.bn3.weight,0)
                elif isinstance(m,BasicBlock):
                    nn.init.constant_(m.bn2.weight,0)
    
    def _make_layer(self,block,inputwidth,width,blocks,dilate=False):
        # block: block structure
        # width: #the default hiddne layer size
        # blocks: #blocks
        norm_layer=self._norm_layer
        downsample=None#downsampling between blocks wasn't added as abstraction wasn't expected here yet
        #reformualted for Linear layers
        #this block will formualte the downsampling for identity in resnet(the first block when changing to anoher plane structure)
        #currently, the block didn't change dimensions
        layers=[]
        ##block will pass the arguments to the two block types
        layers.append(block(inputwidth,width,downsample=downsample,groups=self.groups,
                            base_width=self.base_width,norm_layer=norm_layer))
        for _ in range(1,blocks):
            layers.append(block(inputwidth,width,groups=self.groups,
                                base_width=self.base_width,norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self,x):
        x=self.fc1(x)
        x=self.bn1(x)
        x=self.relu(x)
        # x = self.maxpool(x)
        x=self.layer1(x)
        # x = self.avgpool(x)
        x=torch.flatten(x,1)
        x=F.dropout(self.fcf(x),training=self.training,p=self.p)
        # print("dataparallel checker")
        return x

## a gru network with controled
class gru_mlp_cell(nn.Module):
    def __init__(self,input_size,hidden_size,numlayer=0,bias=True,p=0.0):
        super(gru_mlp_cell,self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.x2h=line1dbias(input_size,3*hidden_size)
        self.h2h=line1dbias(hidden_size,3*hidden_size)
        self.layer1=self._make_layer(hidden_size,numlayer=numlayer,p=p)
        # self.reset_parameters()
    
    def reset_parameters(self):
        std=1.0/math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std,std)
    
    def forward(self,input,hidden):
        ## a condensed reimplementation(approximation) for effieciency (as used in pytroch c++ and other python implementaion)
        gate_input=self.x2h(input)#input transformation
        gate_hidden=self.h2h(hidden)#hidden state transformation
        # print('gate_input{}'.format(gate_input.shape))
        # gate_input=gate_input.squeeze()
        # gate_hidden=gate_hidden.squeeze()
        i_r,i_i,i_n=gate_input.chunk(3,1)#To: reset gate,update gate, update h
        h_r,h_i,h_n=gate_hidden.chunk(3,1)#To: reset gate,update gate,update h
        resetgate=torch.sigmoid(i_r+h_r)
        updategate=torch.sigmoid(i_i+h_i)
        h_n_new=self.layer1(h_n)
        newh=torch.tanh(i_n+resetgate*h_n_new)
        hy=newh+updategate*(hidden-newh)
        # print('resetgate{} updategate{} hidden{}'.format(resetgate.shape,updategate.shape,hidden.shape))
        return hy
        
    def _make_layer(self,hidden_size,numlayer=0,p=0.0):
        layers=[]
        ##block will pass the arguments to the two block types
        for _ in range(0,numlayer):
            layers.append(line1dbias(hidden_size,hidden_size))
            layers.append(nn.Dropout(p=p))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

## a new deigned rnn cell
class diffadd_cell(nn.Module):
    def __init__(self,input_size,hidden_size,numlayer=0,bias=True,p=0.0):
        super(diffadd_cell,self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.layer1=self._make_layer(hidden_size+input_size,numlayer=numlayer,p=p)
        self.lltransf=line1dbias(hidden_size+input_size,hidden_size)
        # self.reset_parameters()
    
    def reset_parameters(self):
        std=1.0/math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std,std)
    
    def forward(self,input,hidden):
        # print('hidden {} input {}'.format(hidden.shape,input.shape))
        hiddeninput=torch.cat((hidden,input),1)
        sizes=hidden.shape
        delt=input[:,-1].view(sizes[0],-1)
        hiddeninput=self.layer1(hiddeninput)
        hidden_d=torch.tanh(self.lltransf(hiddeninput))
        # print('hidden {} hidden_d {} delt{}'.format(hidden.shape,hidden_d.shape,delt.shape))
        hy=hidden+hidden_d*delt
        return hy
        
    def _make_layer(self,hidden_size,numlayer=0,p=0.0):
        layers=[]
        ##block will pass the arguments to the two block types
        for _ in range(1,numlayer):
            layers.append(line1dbias(hidden_size,hidden_size))
            layers.append(nn.Dropout(p=p))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

## the wrapper for rnn model
class RNN_Model(nn.Module):
    def __init__(self,input_dim,output_dim,hidden_dim,input_dim_0,numlayer,type='gru',bias=True,p=0.0):
        ##initialvec initial condition and t0
        super(RNN_Model,self).__init__()
        self.hidden_dim=hidden_dim
        # print('{}\n'.format(type))
        if type=='gru':
            self.rnncell=nn.GRUCell(input_dim,hidden_dim,bias=True)
        elif type=='gru_mlp':
            self.rnncell=gru_mlp_cell(input_dim,hidden_dim,numlayer,p=p)
        elif type=='diffaddcell':
            self.rnncell=diffadd_cell(input_dim,hidden_dim,numlayer,p=p)
        
        self.inputlay=line1dbias(input_dim_0,hidden_dim)
        self.outputlay=line1dbias(hidden_dim,output_dim)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight,mode='fan_out')
        
    def forward(self,x,initialvec):
        ##initialvec: input1 [Y(t_0) t_0], initial condition
        ##x: input2  [theta, t_k, delta t_k], theta and time
        hiddeninput0=initialvec
        self.hiddeninput0=hiddeninput0
        h0=self.inputlay(self.hiddeninput0)
        outs=[]
        hn=h0
        for seq in range(x.size(1)):#time direction
        
            # a=x[:,seq,:]
            # print('{} {}'.format(a.shape,hn.shape))
            
            hn=self.rnncell(x[:,seq,:],hn)
            outs.append(self.outputlay(hn))
        # print('outs{}'.format(outs))
        outtensor=torch.cat(outs)
        return outtensor

### MLP sturcture with control on number of layer and existence of batchnormalization
###the whole residule network structure
class _mlp_mod(nn.Module):

    def __init__(self,layers,ninput,num_response,ncellscale,batchnorm_flag=True,zero_init_residual=False,p=0.0):
        # block: block structure
        # layers: #layers,
        # ninput: #input,
        # num_response: #response,
        # ncellscale: scale factor for hidden layer size
        # zero_init_residual: whether initialize weight as 0. default False,
        # batchnorm_flag: whether include batch normalization layer or not default True
        # p: dropbout probability Default 0.0
        super(_mlp_mod,self).__init__()
        self.inplanes=ninput## the input layer size
        self.width=int(ninput*ncellscale)#the default hiddne layer #neuron (width) is input dimension * scale factor
        self.fc1=line1d(self.inplanes,self.width)
        self.batchnorm_flag=batchnorm_flag
        if self.batchnorm_flag is True:
            self.bn=nn.BatchNorm1d(self.width)
        else:
            self.bn=None
        
        self.relu=nn.ReLU(inplace=True)
        block=BasicBlock_mlp
        self.layer1=self._make_layer(block,self.width,self.width,layers-2)#except the first and the last linear layer
        # self.avgpool=nn.AdaptiveAvgPool2d((1, 1))
        self.fcf=nn.Linear(self.width,num_response)
        self.p=p
        ##paramter initilaization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight,mode='fan_out')
            elif isinstance(m, (nn.BatchNorm1d,nn.GroupNorm)):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)

    def _make_layer(self,block,inputwidth,width,blocks,dilate=False):
        # width: #the default hiddne layer size
        # blocks: #blocks
        batchnorm_flag=self.batchnorm_flag
        #reformualted for Linear layers
        #this block will formualte the downsampling for identity in resnet(the first block when changing to anoher plane structure)
        #currently, the block didn't change dimensions
        layers=[]
        ##block will pass the arguments to the two block types
        layers.append(block(inputwidth,width,batchnorm_flag=batchnorm_flag))
        for _ in range(1,blocks):
            layers.append(block(inputwidth,width,batchnorm_flag=batchnorm_flag))

        return nn.Sequential(*layers)
    
    def forward(self,x):
        x=self.fc1(x)
        if self.bn is not None:
            x=self.bn(x)
        x=self.relu(x)
        x=self.layer1(x)
        x=torch.flatten(x,1)
        x=F.dropout(self.fcf(x),training=self.training,p=self.p)
        # print("dataparallel checker")
        return x

def _resnet(ninput,num_response,block,layers,pretrained,progress,ncellscale,**kwargs):
    # ninput: #input,
    # num_response: #response,
    # block: block structure,
    # layers: #layers,
    # pretrained: pretrained or not(not currently used)
    # progress: progress(not currently used),
    # ncellscale: scale factor for hidden layer size
    # **kwargs: to add other parameters
    model=ResNet_mlp(block,layers,ninput,num_response,ncellscale=ncellscale,**kwargs)
    if pretrained:
        print("no pretrained model currently")
    return model

def _rnnnet(ntheta,nspec,num_layer,ncellscale,type,**kwargs):
    # ninput: #input,
    # num_response: #response,
    # block: block structure,
    # layers: #layers,
    # pretrained: pretrained or not(not currently used)
    # progress: progress(not currently used),
    # ncellscale: scale factor for hidden layer size
    # **kwargs: to add other parameters
    input_dim=ntheta-nspec+1
    output_dim=nspec
    input_dim_0=nspec+1
    hidden_dim=int(input_dim_0*(ncellscale+1))
    # print('input_dim {} output_dim {} input_dim_0 {} hidden_dim {}'.format(input_dim,output_dim,input_dim_0,hidden_dim))
    model=RNN_Model(input_dim,output_dim,hidden_dim,input_dim_0,num_layer,type,**kwargs)
    return model

### resnet structure examples
# ninput: #input,
# num_response: #response,
# p: dropbout probability Default 0.0
# ncellscale: scale factor for hidden layer size Default 1.0
# pretrained: pretrained or not(not currently used) Default False
# progress: progress(not currently used) Default True
# **kwargs: to add other parameters

def resnet10_mlp(ninput,num_response,p=0.0,ncellscale=1.0,pretrained=False,progress=True,**kwargs):
    r"""ResNet-10 model adapted from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    kwargs['p']=p
    return _resnet(ninput,num_response,BasicBlock,[4],pretrained,progress,ncellscale,**kwargs)

def resnet14_mlp(ninput,num_response,p=0.0,ncellscale=1.0,pretrained=False,progress=True,**kwargs):
    r"""ResNet-14 model adapted from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    kwargs['p']=p
    return _resnet(ninput,num_response,BasicBlock,[6],pretrained,progress,ncellscale,**kwargs)


def resnet18_mlp(ninput,num_response,p=0.0,ncellscale=1.0,pretrained=False,progress=True,**kwargs):
    r"""ResNet-18 model adapted from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    kwargs['p']=p
    return _resnet(ninput,num_response,BasicBlock,[8],pretrained,progress,ncellscale,**kwargs)


def resnet34_mlp(ninput,num_response,p=0.0,ncellscale=1.0,pretrained=False,progress=True,**kwargs):
    r"""ResNet-34 model adapted from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    kwargs['p']=p
    return _resnet(ninput,num_response,BasicBlock,[16],pretrained,progress,ncellscale,**kwargs)


def resnet50_mlp(ninput,num_response,p=0.0,ncellscale=1.0,pretrained=False,progress=True,**kwargs):
    r"""ResNet-50 model adapted from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    kwargs['p']=p
    return _resnet(ninput,num_response,Bottleneck,[16],pretrained,progress,ncellscale,**kwargs)


def resnet101_mlp(ninput,num_response,p=0.0,ncellscale=1.0,pretrained=False,progress=True,**kwargs):
    r"""ResNet-101 model adapted from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    kwargs['p']=p
    return _resnet(ninput,num_response,Bottleneck,[33],pretrained,progress,ncellscale,**kwargs)


def resnet152_mlp(ninput,num_response,p=0.0,ncellscale=1.0,pretrained=False,progress=True,**kwargs):
    r"""ResNet-152 model adapted from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    kwargs['p']=p
    return _resnet(ninput,num_response,Bottleneck,[50],pretrained,progress,ncellscale,**kwargs)

def resnet2x_mlp(ninput,num_response,p=0.0,ncellscale=1.0,pretrained=False,progress=True,x=9,**kwargs):
    r"""ResNet-x model adapted from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    x can be any number and the number of layers will be 2x, default 9 that is resnet18
    use structure as resnet18, that is no bottleneck as in deeper networks
    """
    kwargs['p']=p
    return _resnet(ninput,num_response,BasicBlock,[2*x-2],pretrained,progress,ncellscale,**kwargs)

# def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
#     r"""ResNeXt-50 32x4d model from
#     `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
#
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     kwargs['groups'] = 32
#     kwargs['width_per_group'] = 4
#     return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
#                    pretrained, progress, **kwargs)
#
#
# def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
#     r"""ResNeXt-101 32x8d model from
#     `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
#
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     kwargs['groups'] = 32
#     kwargs['width_per_group'] = 8
#     return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
#                    pretrained, progress, **kwargs)


def wide_resnet50_2_mlp(ninput,num_response,p=0.0,ncellscale=1.0,pretrained=False,progress=True,**kwargs):
    r"""Wide ResNet-50-2 model adapted from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    """
    kwargs['width_per_group']=64*2
    kwargs['p']=p
    return _resnet(ninput,num_response,Bottleneck,[16],pretrained,progress,ncellscale,**kwargs)


def wide_resnet101_2_mlp(ninput,num_response,p=0.0,ncellscale=1.0,pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model adapted from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    """
    kwargs['width_per_group']=64*2
    kwargs['p']=p
    return _resnet(ninput,num_response,Bottleneck,[33],pretrained,progress,ncellscale,**kwargs)

def mlp_mod(ninput,num_response,nlayer,batchnorm_flag=True,p=0.0,ncellscale=1.0,pretrained=False,progress=True,**kwargs):
    r"""regular MLP with control on number of layer and whether batchnorm is inlcuded
    """
    kwargs['p']=p
    kwargs['batchnorm_flag']=batchnorm_flag
    return _mlp_mod(nlayer,ninput,num_response,ncellscale,**kwargs)

def gru_mlp_rnn(ntheta,nspec,num_layer,p=0.0,ncellscale=1.0,**kwargs):
    r"""rnn model adapted from
    a controled number of layer is added within gru
    """
    kwargs['p']=p
    type='gru_mlp'
    return _rnnnet(ntheta,nspec,num_layer,ncellscale,type,**kwargs)

def gru_rnn(ntheta,nspec,num_layer,p=0.0,ncellscale=1.0,**kwargs):
    r"""the original gru in pytorch
    """
    kwargs['p']=p
    type='gru'
    return _rnnnet(ntheta,nspec,num_layer,ncellscale,type,**kwargs)

def diffaddcell_rnn(ntheta,nspec,num_layer,p=0.0,ncellscale=1.0,**kwargs):
    r"""a new designed rnn structure
    """
    kwargs['p']=p
    type='diffaddcell'
    return _rnnnet(ntheta,nspec,num_layer,ncellscale,type,**kwargs)
