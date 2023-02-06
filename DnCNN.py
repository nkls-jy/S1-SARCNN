import math 
import torch.nn as nn

def conv_with_padding(in_planes, out_planes, kernelsize, stride=1, dilation=1, bias=False, padding = None):
    """
    in_planes: number of channels in the input image
    out_planes: number of channels produced by the convolution
    stride = 1 -> no steps
    dilation = 1 -> normal convolution (no pixels omitted)
    """
    if padding is None:
        padding = kernelsize//2
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernelsize, stride=stride, dilation=dilation, padding=padding, bias=bias)

def conv_init(conv, act='linear'):
    """
    initializes the Convolution layer
    """
    # kernel_size: [int, int] -> height & width of conv window
    n = conv.kernel_size[0] * conv.kernel_size[1] * conv.out_channels
    conv.weight.data.normal_(0, math.sqrt(2. /n))

def batchnorm_init(m, kernelsize=3):
    """
    initialized batchnorm
    """
    n = kernelsize**2 * m.num_features
    m.weight.data.normal_(0, math.sqrt(2. / (n)))
    m.bias.data.zero_()

def make_activation(act):
    """
    Which activation function to apply
    """
    if act is None:
        return None
    elif act == 'relu':
        return nn.ReLU(inplace=True)
    elif act == 'tanh':
        return nn.Tanh()
    elif act == 'leaky_relu':
        return nn.LeakyReLU(inplace=True)
    elif act == 'softmax':
        return nn.Softmax()
    elif act == 'linear':
        return None
    else:
        assert(False)

def make_net(nplanes_in, kernels, features, bns, acts, dilats, bn_momentum = 0.1, padding=None):
    """
    :param nplanes_in: number of of input feature channels
    :param kernels: list of kernel size for convolution layers
    :param features: list of hidden layer feature channels
    :param bns: list of whether to add batchnorm layers
    :param acts: list of activations
    :param dilats: list of dilation factors
    :param bn_momentum: momentum of batchnorm
    :param padding: integer for padding (None for same padding)
    """

    depth = len(features)
    assert(len(features) == len(kernels))

    layers = list()
    for i in range(0, depth):
        if i == 0:
            in_feats = nplanes_in
        else:
            in_feats = features[i-1]

        elem = conv_with_padding(in_feats, features[i], kernelsize=kernels[i], dilation=dilats[i], padding=padding, bias=not(bns[i]))
        # initialize conv layer
        conv_init(elem, act=acts[i])
        layers.append(elem)

        elem = make_activation(acts[i])
        if elem is not None:
            layers.append(elem)
        
        if bns[i]:
            elem = nn.BatchNorm2d(features[i], momentum = bn_momentum)
            batchnorm_init(elem, kernelsize=kernels[i])
            layers.append(elem)
        
        #print(f"depth % 3: {i % 3}")
        #print(i)
        #if i > 1 and i < depth-1 and i % 3 == 0:
        #    print(f"adding Dropout")
        #    layers.append(nn.Dropout2d(p=0.5))

    return nn.Sequential(*layers)


"""
Function to multiply output weights with image patches
"""

def mul_weights_patches(x, w, kernel, stride=1, padding=False):
    if padding:
        import torch.nn.functional as F
        pad_row = -(x.shape[2] - kernel) % stride
        pad_col = -(x.shape[3] - kernel) % stride
        x = F.pad(x, (pad_col // 2, pad_col - pad_col // 2, pad_row // 2, pad_row - pad_row // 2))

    # change dimension order of weight tensor
    w = w.permute(0, 2, 3, 1)
    w = w.view(w.shape[0], 1, w.shape[1], w.shape[2], kernel, kernel)

    patches = x.unfold(2, kernel, stride).unfold(3, kernel, stride)

    pad_row = (w.shape[2] - patches.shape[2]) // 2
    pad_col = (w.shape[3] - patches.shape[3]) // 2

    if (pad_row<0) or (pad_col<0):
        patches = patches[:,:,max(-pad_row,0):(patches.shape[2] - max(-pad_row,0)), max(-pad_col,0):(patches.shape[3] - max(-pad_col,0)),:,:]
    if (pad_row>0) or (pad_col>0):
        w = w[:,:,max(pad_row,0):(w.shape[2] - max(pad_row,0)), max(pad_col,0):(w.shape[3] - max(pad_col,0)),:,:]

    # prediction
    #y = patches * w
    #print(f"output pred shape: {y.shape}")

    y = (patches * w).sum((4, 5))

    return y

"""
Model Definition
"""

class NlmCNN(nn.Module):
    def __init__(self, network_weights, sizearea, padding = False):
        
        super(NlmCNN, self).__init__()
        
        self.network_weights = network_weights
        self.sizearea = sizearea
        self.padding = padding
        #self.sar_data = sar_data

    def forward_weigths(self, x, reshape=False):
        
        #x_in = x.abs().log() / 2.0

        #if self.sar_data:
        #    x_in = x.abs().log() / 2.0
        #else:
        #    x_in = x
        x_in = x
        
        w = self.network_weights(x_in)

        if reshape:
            w = w.permute(0, 2, 3, 1)
            w = w.view(w.shape[0], w.shape[1], w.shape[2], self.sizearea, self.sizearea)
            return w
        else:
            return w

    def forward(self, x):
        w = self.forward_weigths(x)
        y = mul_weights_patches(x, w, self.sizearea, stride=1, padding=self.padding)
        return y


"""
Function for backnet
"""
def make_backnet(nplanes_in, sizearea, bn_momentum=0.1, padding=False):
    #depth = 15
    depth = 12
    depth = 10 # for bigger Sizearea

    features = [441, 529, 625, 729, 841, 961, 1089, 1225, 1369, sizearea * sizearea] # features for big Sizearea and shallower network
    # features: large sizearea, deep network
    #features = [225, 256, 289, 324, 361, 441, 520, 625, 729, 841, 961, 1089, 1156, 1225, sizearea*sizearea]
    # features: default sizearea (25)
    #features = [169, 225, 289, 361, 441, 529, 625, 729, 841, 961, 1089, sizearea*sizearea]
    #features = [169, 225, 289, 361, 441, 529, 625, 729, 841, sizearea*sizearea]
    #kernels = [7, 5, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1]

    kernels = [7, 5, 3, 3, 3, 3, 3, 3, 3, 1]  # Kernels for big Sizearea and shallower network
    #kernels = [5, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1]
    #kernels = [5, 5, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1]
    #kernels = [5, 5, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1]
    
    # big kernels
    #kernels = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 1]
    
    dilats = [1, ] * depth
    acts = ['leaky_relu', ] * (depth-1) + ['softmax', ]
    bns = [False, ] + [True, ] * (depth-2) + [False, ]
    network_weights = make_net(nplanes_in, kernels, features, bns, acts, dilats=dilats, bn_momentum=bn_momentum, padding=None if padding else 0)
    return network_weights
