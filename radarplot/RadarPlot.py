#
# some plot functions for radar nowcasting
#
# ---------------------------------

import matplotlib.pyplot as plt
import matplotlib.colors as cl
from matplotlib.backends.backend_pdf import PdfPages

# from mpl_toolkits.basemap import Basemap

from chainer import cuda
import chainer.links as L
import numpy as np

import dill
import os

import dataimport as g              # data generators

# avoid warnings
plt.rcParams.update({'figure.max_open_warning': 0})


# define non-linear transformation of color scale
nlnorm = cl.PowerNorm(1.0/2.0)


# plot validation error and epsilon
def learningplot(pdfname, model, scale="linear"):
    with PdfPages(pdfname) as pdf:
        fig, axes = plt.subplots(nrows=2, ncols=1,  figsize=(6,7))

        # --- loss
        loss = model.training.losses()
        axes[0].plot(loss, marker='o')
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("validation loss")
        axes[0].set_yscale(scale)

        for e in np.cumsum(model.training.getall("epochs")):
            axes[0].axvline(e-0.5, color="gray")
        axes[0].set_xlim([0, model.training.epochs()])


        # --- learning schedule
        epochs = np.arange(model.training.epochs())
        eps = np.repeat(model.training.getall("eps_decay"), model.training.getall("epochs"))
        eps_min = np.repeat(model.training.getall("eps_min"), model.training.getall("epochs"))
        steps = 1.0/np.maximum(eps**epochs, eps_min)

        axes[1].plot(epochs, steps, marker='o')
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("expected number of steps\n between training images")

        for e in np.cumsum(model.training.getall("epochs")):
            axes[1].axvline(e-0.5, color="gray")

        tf = model.training.getall("train_files")
        xtext = 0
        for i in range(len(tf)):
            tt = tf[i]
            for j in range(len(tt)):
                tt[j] = tt[j].replace("/home/scheidan/RadarData/SequenceData/", "")
                axes[1].text(xtext-0.3+0.8*j, 1, tt[j], fontsize=7, color="green",
                             rotation=90, ha='left', va="bottom")
            xtext = np.cumsum(model.training.getall("epochs"))[i]

        tf = model.training.getall("test_files")
        for i in range(len(tf)):
            tt = tf[i]
            xtext = np.cumsum(model.training.getall("epochs"))[i]
            for j in range(len(tt)):
                tt[j] = tt[j].replace("/home/scheidan/RadarData/SequenceData/", "")
                axes[1].text(xtext-1.2-0.8*j, 1, tt[j], fontsize=7, color="red",
                             rotation=90, ha='left', va="bottom")

        axes[1].set_xlim([0, model.training.epochs()])
        pdf.savefig()
        plt.close()


# plot a series of predictions
# X_true    - array with dim (N, 50, 50)
# X_pred    - array with dim (N, n_pred, 50, 50)
def prediction_series(pdfname, X_true, X_pred, offset=0, zmin=0.0, zmax=5.0, clrnorm = nlnorm):

    X_true = cuda.to_cpu(X_true)
    X_pred = cuda.to_cpu(X_pred)

    with PdfPages(pdfname) as pdf:

        for t in range(min(X_pred.shape[1], X_true.shape[0])):
            fig, axes = plt.subplots(nrows=1, ncols=2)

            im = axes[0].imshow(X_true[offset+t,:,:], interpolation="none", cmap="cubehelix_r",
                                vmin=zmin, vmax=zmax, norm=clrnorm)

            # add colorbar
            cbaxes = fig.add_axes([0.1, 0.1, 0.8, 0.03])
            cb = fig.colorbar(im, orientation='horizontal', cax = cbaxes)
            cb.set_label("rain intensity [mm/h]")

            axes[0].set_xlabel("xcoor")
            axes[0].set_ylabel("ycoor")
            axes[0].set_title("Ref t={}".format(t+offset))

            if t==0:
                axes[1].imshow(X_true[offset+t,:,:], interpolation="none", cmap="cubehelix_r",
                                vmin=zmin, vmax=zmax, norm=clrnorm)
            else:
                axes[1].imshow(X_pred[offset,t-1,:,:], interpolation="none", cmap="cubehelix_r",
                               vmin=zmin, vmax=zmax, norm=clrnorm)
            axes[1].set_xlabel("xcoor")
            axes[1].get_yaxis().set_ticks([]) # no labels/numbers
            axes[1].set_title("Predicted at t={}".format(offset))



            pdf.savefig()
            plt.close()



# plot online predictions
# for different time steps
# X_true    - array with dim (N, 50, 50)
# X_pred    - array with dim (N, n_pred, 50, 50)
# n_pred    - array of length 5 with the forecast horizones [min]
# time_step - time between two images [min]
def validation(pdfname, X_true, X_pred, n_pred=[5, 10, 15, 30, 60], time_step=2.5,
               zmin=0.0, zmax=5.0, clrnorm = nlnorm):

    X_true = cuda.to_cpu(X_true)
    X_pred = cuda.to_cpu(X_pred)

    assert int(max(n_pred)/2.5) <= X_pred.shape[1], "Not enough prediction steps are calculated!"

    with PdfPages(pdfname) as pdf:

        for t in range(12, X_pred.shape[0]):
            fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(1.5*9,1.5*5))
            # fig.tight_layout()

            im = axes[0,0].imshow(X_true[t,:,:], interpolation="none", cmap="cubehelix_r",
                           vmin=zmin, vmax=zmax, norm=clrnorm)

            axes[0,0].set_xlabel("xcoor")
            axes[0,0].set_ylabel("ycoor")
            axes[0,0].set_title("Ref t={}".format(t))

            cx = 1
            cy = 0
            for ii in range(5):
                off = int(n_pred[ii]/time_step)
                axes[cy, cx].imshow(X_pred[t-off,off+1,:,:],
                                 interpolation="none", cmap="cubehelix_r",
                                 vmin=zmin, vmax=zmax, norm=clrnorm)
                axes[cy, cx].set_xlabel("xcoor")
                axes[cy, cx].get_yaxis().set_ticks([]) # no labels/numbers
                axes[cy, cx].set_title("Pred {} min".format(n_pred[ii]))
                cx += 1
                if cx > 2:
                    cx = 0
                    cy += 1

            # add colorbar
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            fig.colorbar(im, cax=cbar_ax)

            pdf.savefig()
            plt.close()


# similar to "validation()" but with a background map
#  (based on basemap)
#
# X_true    - array with dim (N, 50, 50)
# X_pred    - array with dim (N, n_pred, 50, 50)
# n_pred    - array of length 5 with the forecast horizones [min]
# time_step - time between two images [min]
# lon       - tuple with longitude of the west and east corners
#             in WG84. Swiss coordinates can be used with CH1903 = True.
# lat       - tuple with latitude of the south and north corners
# CH1903    - if true, lon and lat are can be given in CH1903 coordinates.
# frame_factor - factor of the width or hight of the image used as frame (east-west, north-south)

def validation_map(pdfname, X_true, X_pred, n_pred=[5, 10, 15, 30, 60],
                   time_step=2.5, zmin=0.0, zmax=5.0, clrnorm = nlnorm,
                   lon=(5, 10), lat=(45, 48), CH1903=False, frame_factor=(0.3, 0.3)):

    X_true = cuda.to_cpu(X_true)
    X_pred = cuda.to_cpu(X_pred)

    assert int(max(n_pred)/2.5) <= X_pred.shape[1], "Not enough prediction steps are calculated!"

    # convert swiss coordiantes
    if CH1903:
        c1 = CH1903_to_WGS84(lon[0], lat[0])
        c2 = CH1903_to_WGS84(lon[1], lat[1])
        lon = (c1[0], c2[0])         # west / east
        lat = (c1[1], c2[1])         # north / south

    # define map boundaries
    lon_offset = frame_factor[0]*(lon[1] - lon[0])/2.0
    lat_offset = frame_factor[1]*(lat[1] - lat[0])/2.0


    # -- prepare map objects and cache it
    _, axes = plt.subplots(nrows=2, ncols=3, figsize=(1.5*9,1.5*5))
    # fig.tight_layout()

    for cy in range(2):
        for cx in range(3):
            # llcrnrlat,llcrnrlon,urcrnrlat,urcrnrlon
            # are the lat/lon values of the lower left and upper right corners
            map = Basemap(projection='merc', ax=axes[cy,cx],
                          llcrnrlat=min(lat)-lat_offset, urcrnrlat=max(lat)+lat_offset,
                          llcrnrlon=min(lon)-lon_offset, urcrnrlon=max(lon)+lon_offset,
                          resolution='h',
                          area_thresh = 20) # the minimal area [km2] of an feature to be plotted

            # map.bluemarble()
            map.drawcountries(linewidth=1.3, color="gray")
            # map.drawcoastlines(color='lightblue', linewidth=0.8)
            # map.drawrivers(color='lightblue', linewidth=0.8)
            # map.shadedrelief()
            # map.drawparallels(np.arange(45, 50, 1), color="gray", linewidth=0.8,
            #                   labels=[True,False,False,False]) # labels = [left,right,top,bottom]
            # map.drawmeridians(np.arange(0, 20, 1),labels=[False,False,False,True],
            #                   color="gray", linewidth=0.8)

    dill.dump(axes, open("temp_map.pip", "wb" ), protocol=2)

    with PdfPages(pdfname) as pdf:

        for t in range(12, X_pred.shape[0]):
            fig, _ = plt.subplots(nrows=2, ncols=3, figsize=(1.5*9,1.5*5))
            axes = dill.load(open("temp_map.pip", "rb" ))

            # plot image
            x0, y0 = map(lon[0], lat[0])
            x1, y1 = map(lon[1], lat[1])
            im = axes[0,0].imshow(X_true[t,:,:], extent=(x0, x1, y0, y1),
                                  vmin=zmin, vmax=zmax, norm=clrnorm,
                                  interpolation="none", cmap="cubehelix_r")
            axes[0,0].set_title("Ref t={}".format(t))

            cx = 1
            cy = 0
            for ii in range(5):
                off = int(n_pred[ii]/time_step)

                im = axes[cy, cx].imshow(X_pred[t-off,off+1,:,:], extent=(x0, x1, y0, y1),
                                    vmin=zmin, vmax=zmax, norm=clrnorm,
                                    interpolation="none", cmap="cubehelix_r")
                axes[cy, cx].set_title("Pred {} min".format(n_pred[ii]))
                cx += 1
                if cx > 2:
                    cx = 0
                    cy += 1

            # add colorbar
            # fig.subplots_adjust(right=0.8)
            # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            # fig.colorbar(im, cax=cbar_ax)

            # add colorbar
            cbaxes = fig.add_axes([0.1, 0.1, 0.8, 0.03])
            cb = fig.colorbar(im, orientation='horizontal', cax = cbaxes)
            cb.set_label("rain intensity [mm/h]")

            pdf.savefig()
            plt.close()

    os.remove("temp_map.pip")


# plot error of online predictions
# for different time steps
# X_true    - array with dim (N, 50, 50)
# X_pred    - array with dim (N, n_pred, 50, 50)
def error(pdfname, X_true, X_pred):

    X_true = cuda.to_cpu(X_true)
    X_pred = cuda.to_cpu(X_pred)

    with PdfPages(pdfname) as pdf:

        fig, ax = plt.subplots(2,1)

        ax[0].plot(range(X_true.shape[0]), np.mean(X_true, (1,2)), label="avg. rain intensity")
        ax[0].set_ylabel("avg. rain intensity")

        ax[1].set_xlabel("time")
        ax[1].set_ylabel("RSME")

        colors = plt.cm.rainbow(np.linspace(0, 1, X_pred.shape[1]))
        for k, c in zip(range(X_pred.shape[1]), colors):
            error = np.std(X_true[k:,:,:] - X_pred[k:,k,:,:], (1,2))
            time = range(k, X_pred.shape[0])
            ax[1].plot(time, error, label="{}-step ahead".format(k), color=c)
        #ax.legend()

        pdf.savefig()
        plt.close()

# plot RMSE for different prediction horizont
def RMSE(pdfname, model, state, datafiles, length=24, max_pred=72):
    def MS(A):
        assert len(A.shape) == 3, "Input array A must have 3 dimensions!"
        ms = np.empty(A.shape[0]-1)
        for p in range(A.shape[0]-1):
            ms[p] = np.mean(np.power(A[0,:]-A[p+1,:], 2))
        return(ms)

    ms = model.block_prediction(state=state,  datafiles=datafiles,
                                length=length, pred_horizon=max_pred, fun=MS,
                                batchsize=8, return_state=False)
    ms = np.array(ms)

    RMS = np.sqrt(np.mean(ms, axis = 0))
    with PdfPages(pdfname) as pdf:

        plt.plot(RMS)
        plt.xlabel('prediction horizon [steps]')
        plt.ylabel('RMSE')
        plt.title('root mean squared error')
        for e in range(0, max_pred+24, 24):
            plt.axvline(e, color="gray")

        pdf.savefig()
        plt.close()


# plot RMSE for different prediction horizont
def correction(pdfname, model, state, x_data, zmin=0.0, zmax=5.0, clrnorm=nlnorm):
    # run model to update state
    state = model.update_state(state, x_data)

    x_comb, _, x_cor  = model.state_to_pred_split(state)
    x_comb = cuda.to_cpu(x_comb.data)
    x_cor = cuda.to_cpu(x_cor.data)

    with PdfPages(pdfname) as pdf:
        fig, ax = plt.subplots(2,1)

        im0 = ax[0].imshow(x_comb[0,0,:], vmin=zmin, vmax=zmax, norm=clrnorm,
                           interpolation="none", cmap="cubehelix_r")
        ax[0].set_title("Prediction")
        plt.colorbar(im0, ax=ax[0])

        im1 = ax[1].imshow(x_cor[0,0,:],
                           interpolation="none", cmap="BrBG")
        ax[1].set_title("local correction")
        plt.colorbar(im1, ax=ax[1])

        pdf.savefig()
        plt.close()


# plot averaged rain intensities
def data_summary(pdfname, X, clrnorm=nlnorm):

    with PdfPages(pdfname) as pdf:

        fig, axes = plt.subplots(nrows=1, ncols=2)
        fig.tight_layout()

        im = axes[0].imshow(np.flipud(np.mean(X ,0)), interpolation="none",
                            cmap="cubehelix_r", norm=clrnorm)

        axes[0].set_xlabel("xcoor")
        axes[0].set_ylabel("ycoor")
        axes[0].set_title("Mean")

        #fig.subplots_adjust(right=0.8)
        #cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        #fig.colorbar(axes[0], cax=cbar_ax)
        plt.colorbar(im, ax=axes[0])



        im = axes[1].imshow(np.std(X,0), interpolation="none", cmap="cubehelix_r")
        axes[1].set_xlabel("xcoor")
        axes[1].set_ylabel("ycoor")
        axes[1].set_title("Standard deviation ({})".format(np.round(np.std(X),4)))

        #fig.subplots_adjust(right=0.8)
        #cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        #plt.colorbar(axes[1])
        plt.colorbar(im, ax=axes[1])

        pdf.savefig()
        plt.close()


# plot all convolution filter of a model
# for different time steps
# model     - a RNN model
def convolution_filter(pdfname, model):

    with PdfPages(pdfname) as pdf:
        for l in dir(model):
            typ = type(getattr(model, l))

            if typ in [L.connection.convolution_2d.Convolution2D,
                       L.connection.deconvolution_2d.Deconvolution2D]:


                W = cuda.to_cpu(getattr(model, l).W.data)

                fig, ax = plt.subplots(ncols=5, nrows=int(np.ceil(W.shape[0]/5.0)))
                if len(ax.shape) == 1:
                    ax = np.expand_dims(ax, axis=0)
                x = 0
                y = 0
                for i in range(W.shape[0]):
                    ax[x,y].imshow(W[i,0,:], interpolation="none", cmap="gray")
                    ax[x,y].set_title(str(l)+"\nfilter "+str(i))
                    y += 1
                    if y == 5:
                        y = 0
                        x += 1

                pdf.savefig()
                plt.close()


## helper function
def CH1903_to_WGS84(east, north):
    """Convert CH1093 coordiantes into WGS84.
    Based on 'Approximate solution for the transformation
    CH1903 to WGS84', Swisstopo 2005. """
    y_aux = (east - 600000) / 1000000.0
    x_aux = (north - 200000) / 1000000.0

    lon = 2.6779094 \
          + (4.728982 * y_aux) \
          + (0.791484 * y_aux * x_aux) \
          + (0.1306 * y_aux * pow(x_aux, 2)) \
          - (0.0436 * pow(y_aux, 3))
    lon = lon*100/36.0

    lat = 16.9023892 \
          + (3.238272 * x_aux) \
          - (0.270978 * pow(y_aux, 2)) \
          - (0.002528 * pow(x_aux, 2)) \
          - (0.0447 * pow(y_aux, 2) * x_aux) \
          - (0.0140 * pow(x_aux, 3))
    lat = lat*100/36.0

    return (lon, lat)
