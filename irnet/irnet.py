import numpy as np
from collections import OrderedDict
import torch, torch.nn as nn

class IRNet(nn.Module):
    def __init__(self, channels = 32, doublings = 4, reps = 3):
        super(IRNet, self).__init__()
        # set layer sizes
        encoder_in, encoder_out, decoder_in, decoder_out_a = self.coders_gen( channels, doublings, reps )
        # create encoder layers
        self.elayers = nn.ModuleList()
        self.eskipflags = ( encoder_in > 1 ) & ( encoder_out > encoder_in )
        for i in range(len(encoder_in)):
            self.elayers.append( self.conv2d( encoder_in[i], encoder_out[i], maxpool=self.eskipflags[i] ) )

        # create albedo and gradient decoder layers
        self.alayers, self.askipflags, self.askiplayers = self.aglayers( decoder_in, decoder_out_a )
        # self.glayers, self.gskipflags, self.gskiplayers = self.aglayers( decoder_in, decoder_out_g )

    def conv2d( self, n_in, n_out, transpose=False, batchnorm=True, activation=True, maxpool=False, printv=True ):
        x = OrderedDict()
        if transpose:
            x[ 'conv' ] = nn.ConvTranspose2d( n_in, n_out, kernel_size=3, stride=2, padding=1, output_padding=1 )
        else:
            x[ 'conv' ] = nn.Conv2d( n_in, n_out, kernel_size=3, padding=1 )
        if batchnorm:
            x[ 'batch' ] = nn.BatchNorm2d( num_features=n_out, eps=1e-4, momentum=0.1, affine=True )
        if activation:
            x[ 'act' ] = nn.ReLU()
        if maxpool:
            x[ 'maxpool' ] = nn.MaxPool2d( kernel_size=2, stride=2 )
        return nn.Sequential( x )

    def coders_gen(self, channels, doublings, reps):
        tup = []
        for i in range(doublings):
            if i == 0:
                tup.append(channels)
            tup.append(2*tup[-1])
        encoder_out = np.repeat(tup, reps)
        tup = np.repeat(tup, reps)
        tup = np.insert(tup, 0, 1)
        encoder_in = tup[:-1]
        return encoder_in, encoder_out, encoder_out[::-1], encoder_in[::-1]

    def aglayers( self, n_in, n_out ):
        layers = nn.ModuleList()
        skipflags = n_out < n_in
        skipflags[-1] = False
        skiplayers = nn.ModuleList()

        for i in range(len(n_in)-1):
            if skipflags[i]:
                layers.append( self.conv2d( n_in[i], n_out[i], transpose=True ) )
                skiplayers.append( self.conv2d( n_out[i], n_out[i], printv=False ) )
            else:
                layers.append( self.conv2d( n_in[i], n_out[i] ) )
                skiplayers.append( nn.Module() )

        layers.append( self.conv2d( n_in[-1], n_out[-1], batchnorm=False, activation=False ) )
        skiplayers.append( nn.Module() )

        return layers, skipflags, skiplayers

    def forward(self, x):

        # *** normalize x

        # common layers
        conv_out_list = []
        for i, layer in enumerate( self.elayers ):
            if self.eskipflags[i]:
                conv_out_list.append( x )
            x = layer( x )

        # albedo layers
        aout = x
        cout = conv_out_list.copy()
        for i, layer in enumerate( self.alayers ):
            aout = layer( aout )
            if self.askipflags[i]:
                y = cout.pop()
                z = self.askiplayers[i]( y )
                aout = aout + z
        # *** check that skip layers are working

        # # gradient layers
        # gout = x
        # for i, layer in enumerate( self.glayer ):
        # 	gout = self.apply_layer( layer, gout )
        # 	if self.gskipflag[i]:
        # 		gout = gout + self.gskiplayer[i]( conv_out_list.pop() )

        # return torch.cat( ( aout, gout ), dim=1 )

        return aout

