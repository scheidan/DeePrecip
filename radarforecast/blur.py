## -------------------------------------------------------
##
## April 15, 2016 -- Andreas Scheidegger
## andreas.scheidegger@eawag.ch
## -------------------------------------------------------

import numpy as np
from chainer import cuda, Function, Variable, Link
import chainer.functions as F

from chainer.functions.connection import convolution_2d
from chainer import link

xp = cuda.cupy

class Gaussian_blur(Link):

    """Gaussian blurring as two-dimensional convolutional layer

    Attributes:
        sigma (~chainer.Variable): scale parameter


    """
    def __init__(self, ksize=7):

        assert np.mod(ksize,2) == 1, "'ksize' of filter must not be even!"

        self.ksize = ksize
        self.pad = (ksize-1)//2

        super(Gaussian_blur, self).__init__(logsigma=(1))
        self.logsigma.data = xp.asarray([0.0], dtype=xp.float32)

        r = range(-(ksize-1)//2, (ksize-1)//2+1)
        dist = xp.asarray([-(x**2.0 + y**2.0) for x in r for y in r], dtype=xp.float32)
        self.dist = Variable(xp.reshape(dist, (1,1,self.ksize, self.ksize)))


    def __call__(self, x):
        """(~chainer.Variable): Input image.
        """

        # build filter matrix

        sd_mat = F.broadcast_to(F.exp(self.logsigma), (1,1,self.ksize, self.ksize))
        W = F.exp(self.dist/sd_mat)
        W = W/xp.sum(W.data)

        return convolution_2d.convolution_2d(x, W, b=None, stride=1, pad=self.pad, use_cudnn=True)
