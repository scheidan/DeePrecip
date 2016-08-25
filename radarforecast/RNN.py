#
# define RNN models
#
# ---------------------------------


import os
import sys

from chainer import Variable, Chain, cuda, optimizers
import chainer.functions as F
import chainer.links as L
import numpy as np

import dataimport as g # data generators

from . import spatialtransformer as st
from . import blur
from . import traininfo as ti

# import pdb
# pdb.set_trace()

gpuID = 0
# xp is either numpy or cuda.cupy depending on GPU id
xp = cuda.cupy if gpuID >= 0 else np


# -----------
# Abstract class with generic functions

class RNN(Chain):

    def __init__(self):
        self.training = ti.training()

    def predict_n_steps(self, state, x_last, nstep_ahead, train=False):
        x = x_last
        x_pred = []
        for p in range(nstep_ahead):
            state = self.forward_one_step(state, x, train=train)
            # prediction for next time step
            x = self.state_to_pred(state).data
            x_pred.append(x[0,0,:])
        return xp.rollaxis(xp.dstack(x_pred),2)

    # prediction with online updating
    def predict_n_steps_updating(self, state, x_last, x_new, nstep_ahead, online_optimizer):
        """Predict n steps ahead but first updating all parameter to minimize the loss
        between 'x_last' and 'x_last_pred' (the previous prediction for x_last)."""

        # compute loss for next time step
        x_batch = xp.asarray(np.stack((x_last, x_new)), dtype=np.float32)
        loss, _ = self.loss_series(state, x_batch, eps=1.0)

        # update parameters
        if not hasattr(online_optimizer, '_hooks'): # check if setup is required
            online_optimizer.setup(self)

        self.zerograds()
        loss.backward()
        loss.unchain_backward() # delete computational 'history'
        online_optimizer.update()

        # update state
        state = self.update_state(state, x_batch[xp.newaxis,1,:,:])

        # make prediction
        x_pred = self.predict_n_steps(state, x_batch[xp.newaxis, xp.newaxis,1,:,:], nstep_ahead)

        return x_pred


    def update_state(self, state, x_data, train=False):
        for t in range(0, x_data.shape[0]):
            x_last = x_data[t, np.newaxis, np.newaxis]
            state = self.forward_one_step(state, x_last, train=train)
        return state


    # makes a series of nstep_ahead predictions (without parameter updating)
    # shape(x_pred) = (N, nstep_ahead, xdim, ydim)
    def predict_n_steps_online_series(self, state, x_data, nstep_ahead, train=False, return_state=False):
        x_pred = []
        for t in range(0, x_data.shape[0]):
            xpr = self.predict_n_steps(state, x_data[t,:], nstep_ahead, train=train)
            x_pred.append(xpr[:, xp.newaxis])
            state = self.update_state(state, x_data[t,0,:], train=train)

        if return_state:
            return (xp.swapaxes(xp.hstack(x_pred), 0, 1), state)
        else:
            return xp.swapaxes(xp.hstack(x_pred), 0, 1)


    def loss(self, xtrue, xpred):
        """Compute loss between two images."""
        return F.mean_squared_error(xpred, xtrue)


    def loss_series(self, state, x_data, burn_in=0, eps=1.0, train=True):
        """Compute average loss over all 'x_data' with scheduled sampling.
        x.data.shape = (N, xdim, ydim)"""

        assert x_data.shape[0] - burn_in >= 2

        loss = 0
        for t in range(0, x_data.shape[0]-1):

            # Select randomly the real image or the model prediction, see:
            # Bengio, S., Vinyals, O., Jaitly, N., and Shazeer, N. (2015)
            # Scheduled sampling for sequence prediction with recurrent neural
            # networks.
            if np.random.random() < eps:
                x_last = x_data[t, np.newaxis, np.newaxis]
            else:
                x_last = self.state_to_pred(state).data

            # -- predict internal state
            # 'state' is updated with every new observation
            state = self.forward_one_step(state, x_last, train=train)

            if t>=burn_in:
                # prediction for next time step
                xn_pred = self.state_to_pred(state)
                xn_true = Variable(x_data[t+1, np.newaxis, np.newaxis], volatile=False)
                newloss = self.loss(xn_pred, xn_true)
                loss += newloss

        loss = loss / (x_data.shape[0] - burn_in -1) # average loss
        return loss, state


    def block_prediction(self, state,
                         datafiles, length, pred_horizon,
                         fun=lambda x: x,
                         batchsize=8, return_state=False):
        """Builds an array with the label image and predictions with
    different forecast horizons and applies 'fun' to it.

    Tries to minimize memory consumption, i.e. 'length' can be large.

    For every time step an array A_t is contructed:
    A_t.shape = (pred_horizon+1, ydim, xdim),
    where the label image is A_t[0,:,:], and
    the forcast made k time steps ahead is A_t[k,:,:]

    If return_state=False:
    return list L,
    L =  [fun(A_t) for all t=1...length]

    If return_state=True:
    return the tuple (L, state)
    """

        # define data generator
        lengthbuffer = pred_horizon - 1
        gen = g.nstep_sequence_multi_hdf5(datafiles, steps=length+lengthbuffer, batch_size=batchsize)

        # get first buffer-block
        buffer = np.empty((0, pred_horizon, self.dim_in[0], self.dim_in[1]), dtype=np.float32)
        while buffer.shape[0] < lengthbuffer+1:
            data = next(gen)
            data = xp.asarray(data[:, np.newaxis, np.newaxis], dtype=np.float32)
            pp, state = self.predict_n_steps_online_series(state, data,
                                                     nstep_ahead=pred_horizon,
                                                     return_state=True)
            pp = cuda.to_cpu(pp)
            buffer = np.append(buffer, pp, axis=0)


        buffer, data = np.split(buffer, [lengthbuffer], axis=0)
        data = data[:,0,:,:]

        result = []
        # loop over batches
        while True:
            data = xp.asarray(data[:, np.newaxis, np.newaxis], dtype=np.float32)

            pp, state = self.predict_n_steps_online_series(state, data,
                                                     nstep_ahead=pred_horizon,
                                                     return_state=True)
            pp = cuda.to_cpu(pp)

            predi = np.append(buffer, pp, axis=0)
            buffer = predi[-lengthbuffer:,:]

            # shift predictions
            for i in range(predi.shape[1]):
                predi[:,i] = np.roll(predi[:,i], i)

            # apply fun
            A =  np.append(cuda.to_cpu(data)[:,:,0,:,:], predi[lengthbuffer:,:], axis=1)
            res = [fun(a) for a in A]
            result.extend(res)

            # read new data
            data = next(gen, None)
            if data is None:
                break

        if return_state:
            return (result, state)
        else:
            return result


# -----------
# A convolutional two layer LSTM network with a spatial transformer network

class ConvLTSMSpatialTransformerDeconv(RNN):

    def __init__(self, dim_in=(50,50), nfilter1=8, n_hidden=15, nfilter2=8,
                 dim_control_points=(2,2), warptype="gaussian"):

        self.dim_in = dim_in
        self.dim_control_points = dim_control_points
        self.dim_ST_control = (2, 3+np.prod(self.dim_control_points))

        # convolution
        ksize1 = (10,10)        # filter size
        stride1 = (5,5)         # "off-set"
        pad1 = (0,0)            # padding on the edges

        self.dim_conv1_out = ((dim_in[0] + 2*pad1[0] - ksize1[0])/stride1[0] + 1,
                        (dim_in[1] + 2*pad1[1] - ksize1[1])/stride1[1] + 1)
        self.nfilter1 = nfilter1

        # hidden
        self.n_hidden = n_hidden

        # deconvolution
        ksize2 = (10,10)        # filter size
        stride2 = (5,5)         # "off-set"
        pad2 = (0,0)            # padding on the edges

        self.dim_conv2_in = ((dim_in[0] + 2*pad2[1] - ksize2[0])//stride2[0] + 1,
                        (dim_in[1] + 2*pad2[1] - ksize2[1])//stride2[1] + 1)
        self.nfilter2 = nfilter2

        super(ConvLTSMSpatialTransformerDeconv, self).__init__()
        super(RNN, self).__init__(
            conv1 = L.Convolution2D(in_channels=1, out_channels=nfilter1,
                                    ksize=ksize1, stride=stride1, pad=pad1),

            l1_x = L.Linear(np.prod(self.dim_conv1_out)*nfilter1, 4 * n_hidden),
            l1_h = L.Linear(n_hidden, 4 * n_hidden),
            l2_h1 = L.Linear(n_hidden, 4 * n_hidden),
            l2_h = L.Linear(n_hidden, 4 * n_hidden),

            # hidden state to transformation
            h_to_A1 = L.Linear(n_hidden, int(n_hidden/2)),
            A1_to_A2 = L.Linear(int(n_hidden/2), np.prod(self.dim_ST_control)),

            # hidden state to output
            h_to_t1 = L.Linear(n_hidden, int(n_hidden/2)),
            t1_to_x = L.Linear(int(n_hidden/2), np.prod(self.dim_conv2_in)*nfilter2),
            conv2 = L.Deconvolution2D(in_channels=nfilter2, out_channels=1,
                                      ksize=ksize2, stride=stride2, pad=pad2),

            # final smooting
            conv_final = blur.Gaussian_blur(ksize=7)

        )
        self.G_target = st.thin_plates(dim_in, dim_control_points=self.dim_control_points,
                                       type=warptype) # constant target grid

    def make_initial_state(self):
        s = {'h1': Variable(xp.zeros((1, self.n_hidden), dtype=xp.float32), volatile=False),
             'c1': Variable(xp.zeros((1, self.n_hidden), dtype=xp.float32), volatile=False),
             'h2': Variable(xp.zeros((1, self.n_hidden), dtype=xp.float32), volatile=False),
             'c2': Variable(xp.zeros((1, self.n_hidden), dtype=xp.float32), volatile=False),
             'x_last': Variable(xp.zeros((1,1)+self.dim_in, dtype=xp.float32), volatile=False)}

        return s


    def forward_one_step(self, state, x_last, train=True):

        x = Variable(x_last, volatile=False)
        a = F.elu(self.conv1(x))

        l1 = F.dropout(F.elu(self.l1_x(a) + self.l1_h(state['h1'])), train=train)
        c1, h1 = F.lstm(state['c1'], l1)
        l2 = F.dropout(F.elu(self.l2_h1(h1) + self.l2_h(state['h2'])), train=train)
        c2, h2 = F.lstm(state['c2'], l2)

        state = {'c1': c1, 'h1': h1, 'c2': c2, 'h2': h2,
                 'x_last': x}
        return state

    def state_to_pred(self, state, train=True):
        Ainit = Variable(xp.hstack((xp.eye(2), xp.zeros((2,self.dim_ST_control[1]-2)))).astype("float32"))
        A1 = F.elu(self.h_to_A1(state['h2']))
        A2 = F.reshape(F.tanh(self.A1_to_A2(A1)), self.dim_ST_control) + Ainit
        G = st.transform_grid(A2, self.G_target)
        x_trans = st.interpolate(Variable(state['x_last'].data[0,0,:], volatile=False), G)
        x_trans = F.reshape(x_trans, (1,1)+self.dim_in)

        t1 = F.elu(self.h_to_t1(state['h1']))
        x_cor = F.reshape(F.elu(self.t1_to_x(t1)), (1, self.nfilter2) + self.dim_conv2_in)
        x_cor = self.conv2(x_cor)

        xout =  x_trans + x_cor

        # final filter for smoothing
        xout = self.conv_final(xout)

        return xout


    def state_to_pred_split(self, state, train=True):
        """Use this function only for diagnostic purposes!!!
        Returns tuple (xout, x_trans, x_cor)"""
        Ainit = Variable(xp.hstack((xp.eye(2), xp.zeros((2,self.dim_ST_control[1]-2)))).astype("float32"))
        A1 = F.elu(self.h_to_A1(state['h2']))
        A2 = F.reshape(F.tanh(self.A1_to_A2(A1)), self.dim_ST_control) + Ainit
        G = st.transform_grid(A2, self.G_target)
        x_trans = st.interpolate(Variable(state['x_last'].data[0,0,:], volatile=False), G)
        x_trans = F.reshape(x_trans, (1,1)+self.dim_in)

        t1 = F.elu(self.h_to_t1(state['h1']))
        x_cor = F.reshape(F.elu(self.t1_to_x(t1)), (1, self.nfilter2) + self.dim_conv2_in)
        x_cor = self.conv2(x_cor)

        xout =  x_trans + x_cor

        # final filter for smoothing
        xout = self.conv_final(xout)

        return (xout, x_trans, x_cor)



## ---------------------------------
## Helper functions

def move_state_to_gpu(state, gpuID=0):
    for key, value in state.items():
        value.data = cuda.to_gpu(value.data, gpuID)


def count_parameters(model):
    npara = 0
    for l in dir(model):
        typ = type(getattr(model, l))

        if typ in [L.connection.linear.Linear,
                   L.connection.convolution_2d.Convolution2D,
                   L.connection.deconvolution_2d.Deconvolution2D]:

            npara += np.size(getattr(model, l).b.data)
            npara += np.size(getattr(model, l).W.data)

    return npara
