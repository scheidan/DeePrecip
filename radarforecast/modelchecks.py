#
# Check if the model produces valid predictions
#
# ---------------------------------

import numpy as np
from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers
import chainer.functions as F
from chainer import computational_graph as c

import dataimport as g              # data generators
from . import RNN


# performs basic test with a model
def check_model(model, test_files, gpuID):

    xp = cuda.cupy if gpuID >= 0 else np

    X_tests = xp.asarray(next(g.batch_sequence_multi_hdf5(test_files, batch_size=20)),
                         dtype=np.float32)

    s0 = model.make_initial_state()

    # move states to GPU
    if gpuID >= 0:
        RNN.move_state_to_gpu(s0, gpuID)

    # prediction
    pp = model.predict_n_steps(s0, X_tests[1,:,:],
                               nstep_ahead=5, train=False)

    assert pp.shape == (5, X_tests.shape[1], X_tests.shape[2])

    # state update
    s1 = model.update_state(s0, X_tests[1:10,:,:])
    assert np.sum(abs(s1['h2'].data)) > 0

    # loss
    l1, _ = model.loss_series(s0, X_tests[1:20,:,:], train=True)
    l2, _ = model.loss_series(s0, X_tests[1:20,:,:], burn_in=2, train=True)
    l3, _ = model.loss_series(s0, X_tests[1:20,:,:], burn_in=10)
    assert l1.data > 0
    assert l2.data > 0
    # assert l2.data > l3.data

    print("All tests passed :)")
