#
# define training class to hold trainings information
#
# ---------------------------------

import sys
import os
import time
import copy
import numbers
import chainer
from chainer import Variable, Chain, cuda
import numpy as np


# -----------
# Class to hold training information

class training(list):

    def __init__(self):
        super(training, self).__init__()

    def add_run(self, optimizer, epochs, batchsize, train_files,
                    test_files, gpuID, **kwargs):
        """
        Store all information of all trainings as a list of dictionaries

        Keyword Arguments:
        optimizer_* -- a chainer.optimizer object
        epochs      -- number of epochs
        batchsize   --
        train_files -- list of file names
        test_files  -- list of file names
        gpuID       -- ID of GPU (or CPU)
        **kwargs    -- further optional arguments
        """

        # training related
        D = {}
        assert isinstance(optimizer, chainer.Optimizer)
        D["optimizer"] = str(type(optimizer))
        D["epochs"] = epochs
        D["batchsize"] = batchsize
        D["train_files"] = train_files
        D["test_files"] = test_files
        D["gpuID"] = gpuID
        D["loss"] = None
        D["date"] = None

        # optional entries
        for k,v in kwargs.items():
            D[str(k)] = v

        # environment
        D["chainer_version"] = chainer.__version__
        D["python_version"] = sys.version
        D["environ"] = os.environ

        # add a new directory if the last run was finalized, else replace
        if len(self) == 0:
            self.append(D)
        else:
            if self[-1]["loss"] is not None:
                self.append(D)
            else:
                self[-1] = D


    def finalize_run(self, loss):
        """
        To call *after* a traings run is finished.

        Keyword Arguments:
        loss      -- list with losses per epoch
        """
        assert isinstance(loss, list)
        assert len(loss) == self[-1]["epochs"]
        self[-1]["loss"] = loss
        self[-1]["date"] = time.strftime("%d/%m/%Y %H:%M:%S")



    def repeat_run(self, **kwargs):
        """
        Repeat the last trainigs run. Reuse the last optimizer.

        Keyword Arguments:
        **kwargs -- given keyword arguments to be changed
        """
        if len(self) == 0:
            raise ValueError('Contains no finalized run!')

        if self[-1]["loss"] is None and len(self)==1:
            raise ValueError('Contains no finalized run!')

        if self[-1]["loss"] is not None:  # add new run
            Dlast = copy.deepcopy(self[-1])
            self.append(Dlast)
        else:                   # replace last run
            self[-1] = copy.deepcopy(self[-2])

        self[-1]["loss"] = None
        self[-1]["date"] = None

        # modify run with optional entries
        for k,v in kwargs.items():
            self[-1][str(k)] = v



    def getlast(self, key, array=True):
        """
        Returns the last entry of "key"
        """
        assert isinstance(key, str)
        return self[-1][key]

    def getall(self, key, array=True):
        """
        Returns a list or np.array of all "key"
        """
        li = []
        assert isinstance(key, str)

        for i in range(len(self)):
            li.append(self[i][key])

        if array & isinstance(li[0], numbers.Number):
            li = np.array(li)
        return li


    def epochs(self):
        """
        Return total number of finalized trainings epochs
        """
        ep = np.sum(self.getall("epochs"))
        if self[-1]["loss"] is None:
            ep = ep - self.getlast("epochs")

        return ep


    def losses(self):
        """
        Return an array with all losses
        """
        lo = self.getall("loss")
        if lo[-1] == None:
            lo.pop()            # remove last item
        return np.hstack(lo)
