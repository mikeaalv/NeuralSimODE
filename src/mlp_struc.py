#this is a reimplementation of a few MLP structure
# 1. the ResNet published in pytorch/vision. A MLP type of setting was fomulated rather than those with CONV layers
# 2. regular MLP structure
#Yue Wu
#UGA 09/06/2019
import torch
import torch.nn as nn
import torch.nn.functional as F
# from .utils import load_state_dict_from_url

__all__=['ResNet_mlp','resnet10_mlp','resnet14_mlp','resnet18_mlp', 'resnet34_mlp', 'resnet50_mlp', 'resnet101_mlp','resnet152_mlp', 'wide_resnet50_2_mlp', 'wide_resnet101_2_mlp' 'mlp_mod'] #'resnext50_32x4d', 'resnext101_32x8d',

##currently no convolution layers
# def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
#     """3x3 convolution with padding"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                      padding=dilation, groups=groups, bias=False, dilation=dilation)

def line1d(in_features,out_features):
    return nn.Linear(in_features,out_features,bias=False)

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
