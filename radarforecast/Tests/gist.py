## -------------------------------------------------------
##
## File: gist.py
##
## February  9, 2016 -- Andreas Scheidegger
## andreas.scheidegger@eawag.ch
## -------------------------------------------------------

## -------------------------------------------------------
##
## Example spatial transformer
##
## See: Jaderberg, M., Simonyan, K., Zisserman, A., and Kavukcuoglu,
## K. (2015) Spatial Transformer Networks. arXiv:1506.02025
##
## February  8, 2016 -- Andreas Scheidegger
## andreas.scheidegger@eawag.ch
## -------------------------------------------------------


import numpy as np

from chainer import Variable, Chain, cuda, optimizers
import chainer.functions as F
import chainer.links as L
import chainer

import os
import sys
sys.path.append("{}/Dropbox/Projects/Nowcasting/Chainer/Modules".format(os.environ['HOME']))
import generators as g              # data generators
import SpatialTransformer as st

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


gpuID = 0           # -1 = CPU, 0 = GPU
if gpuID>=0:
    print(cuda.get_device(gpuID))
    cuda.get_device(gpuID).use()

# -----------
# --- create data

data_in = np.random.random((30, 50)).astype("float32")
data_in[15:22, 20:33] = 2.0
data_in[8:10, 5:10] = 4.0


U = Variable(data_in.astype("float32"))
if gpuID>=0:
    U.to_gpu()

# transform
A = Variable(np.asarray([[1, 0,  -0.2], [0,  0.9,  0]]).astype("float32"))
dimout = data_in.shape

G_target = Variable(st.expand_grid(dimout))
G = st.sampling_grid(A, G_target)
if gpuID>=0:
    G.to_gpu()

dd = st.interpolate(U, G)

dd.to_cpu()
data_out = np.reshape(dd.data, dimout)
print(np.sum(data_out))

# -----------
# --- model to learn the transformation matrix


class AffineMat(chainer.Link):
    """
    This link holds the transformation matrix as parameter.
    However, typically the transformation matrix would be
    provided by an localization network.
    """
    def __init__(self):
        super(AffineMat, self).__init__(
            A=(2, 3),
        )
        self.A.data[...] = np.asarray([[1, 0, 0], [0, 1, 0]]).astype("float32")

    def __call__(self):
        return self.A

class TargetGrid(chainer.Link):
    """
    grid...
    """
    def __init__(self, dimout):
        super(TargetGrid, self).__init__(
            g_target=(3, np.prod(dimout)),
        )
        self.g_target.data[...] = st.expand_grid(dimout)

    def __call__(self):
        return self.g_target


class Testmodel(Chain):
    def __init__(self, dimout):
        self.dimout = dimout
        super(Testmodel, self).__init__(
            A = AffineMat(),
            G_target = TargetGrid(dimout)
        )

    def transform(self, data_in, train=False):
        # G_sampling = F.matmul(self.A(), self.G_target())
        G_sampling = st.sampling_grid(self.A(), self.G_target())
        xpred = st.interpolate(F.reshape(data_in, self.dimout), G_sampling)
        return F.reshape(xpred, (1,)+self.dimout)

    def loss(self, data_in, data_out):
        data_trans = self.transform(data_in, train=False)
        return F.mean_squared_error(data_trans, data_out)


model = Testmodel(dimout)
optimizer = optimizers.MomentumSGD(lr=0.2, momentum=0.9)
optimizer.setup(model)

data_in_var = Variable(data_in[np.newaxis,:])
data_out_var = Variable(data_out[np.newaxis])

if gpuID>=0:
    model.to_gpu()
    data_in_var.to_gpu()
    data_out_var.to_gpu()

with PdfPages("test.pdf") as pdf:

    for epoch in range(50):
        print('epoch %d' % epoch)
        # plot
        if epoch % 2 == 0:
            fig, axes = plt.subplots(nrows=1, ncols=3)
            fig.tight_layout()
            axes[0].imshow(data_in, interpolation="none", cmap="cubehelix_r", vmin=0, vmax=5)
            axes[0].set_title("input")
            axes[1].imshow(data_out, interpolation="none", cmap="cubehelix_r", vmin=0, vmax=5)
            axes[1].set_title("label")
            pred = model.transform(data_in_var).data
            pred = cuda.to_cpu(pred)
            axes[2].imshow(pred[0,:], interpolation="none", cmap="cubehelix_r", vmin=0, vmax=5)
            axes[2].set_title("Epochs: {}".format(epoch))
            pdf.savefig()
            plt.close()
        # update
        model.zerograds()
        loss = model.loss(data_in_var, data_out_var)
        print("loss: {}".format(loss.data))
        loss.backward(retain_grad=True)
        optimizer.update()


print("Estimated transformation matrix:\n {}".format(model.A.A.data))
