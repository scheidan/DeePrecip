#
# Run convolutional LSTM
#
# ---------------------------------

import os
import sys

import numpy as np
from chainer import cuda, gradient_check, Variable, optimizers, serializers
from chainer import computational_graph as c
import chainer

import subprocess
import dill
import copy
import time
import importlib


import radarforecast as forecast
import radarplot
import dataimport as g              # data generators

# importlib.reload(forecast)

# np.random.seed(3287097)



load_model = True
train = False
plot = True

gpuID = 0         # -1 = CPU, 0 = GPU

modelname = "CH_test"
dim_in = (220, 360)

snapshot = 10     # epochs between snapshots
verbose = 1       # every 'verbose' epoch output is printed

# -----------
# define/load model


if not load_model:
     # warptype="thinplate"
    model = forecast.ConvLTSMSpatialTransformerDeconv(dim_in=dim_in, nfilter1=4,
                                                 n_hidden=400, nfilter2=4,
                                                 dim_control_points=(3,3), warptype="gaussian")
    state = model.make_initial_state()
else:
    model, state = dill.load(open("../Models/{}.pic".format(modelname), "rb" ))
    print(type(model))
    print("Model loaded. Pretrained on {} epochs.".format(model.training.epochs()))

# move model to GPU
if gpuID >= 0:
    cuda.get_device(gpuID).use()
    xp = cuda.cupy # xp is either numpy or cuda.cupy depending on GPU id
    model.to_gpu()
    forecast.move_state_to_gpu(state, gpuID=gpuID)
else:
    xp = np

print("The model has {} parameters ({} mb)".format(forecast.count_parameters(model),
                                                   np.round(forecast.count_parameters(model)*4/1024**2, 2)))

# -----------
# define training


# --- define data


data_path = "/home/scheidan/Dropbox/Projects/Nowcasting/MeteoSwissRadar/MeteoSwissRadarHDF5/CH_360x220/"

train_files = ["ch_360x220_2014.02.hdf5",
               # "ch_360x220_2014.03.hdf5",
               # "ch_360x220_2014.04.hdf5",
               # "ch_360x220_2014.05.hdf5",
               # "ch_360x220_2014.06.hdf5",
               # "ch_360x220_2014.07.hdf5",
               # "ch_360x220_2014.08.hdf5",
               # "ch_360x220_2014.09.hdf5",
               # "ch_360x220_2014.10.hdf5",
               # "ch_360x220_2014.11.hdf5",
               # "ch_360x220_2014.12.hdf5",
               # "ch_360x220_2015.01.hdf5",
]

test_files = ["ch_360x220_2015.02.hdf5"]


train_files = [os.path.join(data_path, f) for f in train_files]
test_files = [os.path.join(data_path, f) for f in test_files]



# -- choose optimizer

#optimizer = optimizers.MomentumSGD(lr=0.00001, momentum=0.9)
#optimizer = optimizers.Adam()
#optimizer = optimizers.AdaDelta()
optimizer = optimizers.RMSprop(lr=0.00001, alpha=0.99, eps=1e-08)
#optimizer = optimizers.NesterovAG(lr=0.001, momentum=0.9)

clipGradHook = chainer.optimizer.GradientClipping(3.0)


# -- define run

if train:
    model.training.add_run(optimizer=optimizer, epochs=1, batchsize=64, train_files=train_files,
                           test_files=test_files, gpuID=gpuID, eps_min=1.0/48.0, eps_decay=0.82)

    # model.training.repeat_run(epochs = 1, eps_decay=0.82, eps_min=1.0/48.0,
    #                           train_files=train_files, test_files=test_files)

# -----------
# plot options

nstep_plot = 24*1  # length of validation plot


## -----------
## tests

forecast.check_model(model, test_files, gpuID=gpuID)


## -----------
## fit model

if train:

    batchsize = model.training.getlast("batchsize")
    train_files  = model.training.getlast("train_files")
    test_files = model.training.getlast("test_files")

    optimizer.setup(model)
    optimizer.add_hook(clipGradHook, name="GradClip")

    if load_model and model.training.getall("optimizer")[-2] == str(type(optimizer)):
        serializers.load_hdf5("../Models/{}_opt".format(modelname), optimizer)
        print("Optimizer state loaded!")


    loss_arr = []

    print("Increase forecast horizon by {}% per epoch (max {} steps).".format(
        round((1/model.training.getlast("eps_decay") - 1)*100), round(1/model.training.getlast("eps_min"))))

    for epoch in range(model.training.getlast("epochs")):

        optimizer.new_epoch()

        # adjust eps
        eps = max(model.training.getlast("eps_min"),
                  model.training.getlast("eps_decay")**(model.training.epochs()+epoch))

        t1 = time.time()

        ## --- predict validation loss

        # always same initial state for test data
        state_test = model.make_initial_state()
        if gpuID >= 0:
            forecast.move_state_to_gpu(state_test, gpuID)

        sum_loss = 0
        N = 0
        for x_bb in g.batch_sequence_multi_hdf5(test_files, batchsize):
            x_batch = xp.asarray(x_bb, dtype=np.float32)
            N += x_batch.shape[0]    # length of sequence
            # run model forward
            RMS, state_test = model.loss_series(state_test, x_batch,
                             burn_in = 0, train=False)
            RMS.unchain_backward() # delete computational 'history'
            sum_loss += RMS * x_batch.shape[0]

        mean_loss = sum_loss / N
        mean_loss.to_cpu()
        loss_arr.append(mean_loss.data)
        print('Mean loss over test data: {}\n'.format(mean_loss.data))


        ## --- training
        state = model.make_initial_state()
        if gpuID >= 0:
            forecast.move_state_to_gpu(state, gpuID)

        nprocess = 0
        for x_bb in g.batch_sequence_multi_hdf5(train_files, batchsize):
            x_batch = xp.asarray(x_bb, dtype=np.float32)
            nprocess += x_batch.shape[0]
            sys.stdout.write("\r{} images processed (ca. {} days)".format(nprocess, round(nprocess/576.0, 1)))
            sys.stdout.flush()

            # run model forward
            RMS, state = model.loss_series(state, x_batch, eps = eps, burn_in = 0)

            model.zerograds()
            RMS.backward()
            RMS.unchain_backward() # delete computational 'history'
            optimizer.update()


        t2 = time.time()
        epoch_tot = epoch+1+model.training.epochs()

        if (epoch_tot)%verbose == 0:
            print('\nepoch: ' + str(epoch_tot))
            print('training time: {} sec. ({} epochs/h)'.format(round(t2-t1,1), round(3600/(t2-t1),1)))
            print('  batch loss: ' + str(RMS.data))


        # --- save snapshot
        if (epoch_tot)%snapshot == 0:

            snap_name = "../Models/{:04d}_Snapshot_{}".format(epoch_tot, modelname)
            dill.dump((model.to_cpu(), state.copy()),
                      open(snap_name+".pic", "wb" ), protocol=2)
            serializers.save_hdf5(snap_name+"_opt", optimizer)
            print("Snapshot saved: " + snap_name)
            if gpuID >= 0:
                  model.to_gpu()


    # --- finalize training run
    model.training.finalize_run(loss_arr)

    print(("Model trained on {} epochs ({} new).\n").format(model.training.epochs(),
                                                           model.training.getlast("epochs")))


    # --- save output
    dill.dump((model.to_cpu(), state.copy()),
                open("../Models/{}.pic".format(modelname), "wb"), protocol=2)
    serializers.save_hdf5("../Models/{}_opt".format(modelname), optimizer)
    print("Final model saved as: ../Models/{}.pic".format(modelname))
    if gpuID >= 0:
        model.to_gpu()




## --- plots

if plot:

    # -- graph
    n_graph_steps = 3
    assert n_graph_steps >=3
    X_graph = next(g.batch_sequence_multi_hdf5(model.training.getlast("test_files"),
                                               batch_size=n_graph_steps))
    RMSplot, _ = model.loss_series(state, xp.asarray(X_graph, dtype=np.float32), train=True)

    with open("../Models/{}.dot".format(modelname), "w") as o:
        o.write(c.build_computational_graph((RMSplot, ), rankdir='LR').dump())
    cmdstr = "dot -Tpng ../Models/{}.dot > ../Plots/{}_graph.png".format(modelname, modelname)
    status = subprocess.call(cmdstr, shell=True)


    # -- error plot
    radarplot.learningplot("../Plots/{}_loss.pdf".format(modelname), model)


    # -- validation

    nstep_ahead = 48
    # X_plot = next(g.batch_sequence_multi_hdf5(model.training.getlast("test_files"),
    #                                           batch_size=nstep_plot+nstep_ahead))
    X_plot = xp.asarray(next(g.batch_sequence_multi_hdf5(test_files,
                                              batch_size=nstep_plot+nstep_ahead)),
                        dtype=np.float32)

    pp = model.predict_n_steps_series(state, X_plot, nstep_ahead=nstep_ahead)


    # --- correction
    radarplot.correction("../Plots/{}_LocalCorrection.pdf".format(modelname), model, state, X_plot)


    radarplot.prediction_series("../Plots/{}_series.pdf".format(modelname),
                                X_true=X_plot[nstep_ahead:,:,:], X_pred=pp[nstep_ahead:,:,:],
                                offset=0, zmax=50)

    radarplot.error("../Plots/{}_error.pdf".format(modelname),
                          X_true=X_plot[nstep_ahead:,:,:], X_pred=pp[nstep_ahead:,:])


    # radarplot.validation_map("../Plots/{}_validation.pdf".format(modelname), n_pred=[15, 30, 45, 60, 75],
    #                          X_true=x[nstep_ahead:,:,:], X_pred=pp[nstep_ahead:,:,:], zmax=50,
    #                          lon=(470000,830000), lat=(65000, 285000), CH1903=True)

    radarplot.validation("../Plots/{}_validation.pdf".format(modelname), n_pred=[15, 30, 45, 60, 75],
                         X_true=X_plot[nstep_ahead:,:,:], X_pred=pp[nstep_ahead:,:,:], zmax=50)


    # # --- data plot
    # x = next(g.batch_sequence_multi_hdf5(model.training.getlast("test_files"),
    #                                      batch_size=10000))
    # radarplot.data_summary("Plots/{}_data.pdf".format(modelname), x)



    # --- plot convolution filter
    radarplot.convolution_filter("../Plots/{}_filter.pdf".format(modelname), model)

    # --- plot RMSE
    online_optimizer = optimizers.MomentumSGD(lr=0.1, momentum=0.6)
    radarplot.RMSE("../Plots/{}_RMSE.pdf".format(modelname), model, state, test_files,
                   length=24*10, max_pred=72)
    radarplot.RMSE("../Plots/{}_RMSE_updating.pdf".format(modelname), model, state, test_files,
                   length=24*10, max_pred=72, optimizer=online_optimizer)

    print("plots finished")


# ============================================
