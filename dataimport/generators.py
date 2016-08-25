#
# define generators to provide data
#
# ---------------------------------


import h5py
import random
import numpy as np
# import netCDF4


# ---------------------------------
# hdf5 generator


def batch_sequence_hdf5(filename, batch_size = 128):
    """Returns an iterable object over a hdf5 file"""

    h5_file = h5py.File(filename, "r")

    d_rainrate = h5_file['rainrate']

    n = d_rainrate.shape[0]
    for ndx in range(0, n, batch_size):

        yield (np.transpose(d_rainrate[ndx:min(ndx + batch_size, n),:], (0,2,1)))

    h5_file.close()



# # get the data in batches

# # f5 = h5py.File("/home/scheidan/RadarData/MochaData/radar_lucern_serial_35x35_2014.02.hdf5", "r")
# f5 = "/home/scheidan/RadarData/SequenceData/lucern_50x50_2014.02.hdf5"

# for x in batch_sequence_hdf5(f5, batch_size = 912):
#     print("----")
#     print("x:", x.shape)




def batch_sequence_multi_hdf5(filenames, batch_size = 128, shuffle=False):
    """Returns an iterable object over multiple hdf5 file"""

    if shuffle:
        random.shuffle(filenames)

    for fname in filenames:

        h5_file = h5py.File(fname, "r")

        d_rainrate = h5_file['rainrate']

        n = d_rainrate.shape[0]
        for ndx in range(0, n, batch_size):
            yield (np.transpose(d_rainrate[ndx:min(ndx + batch_size, n),:], (0,2,1)))

        h5_file.close()


# data_path = "/home/scheidan/Dropbox/Projects/Nowcasting/MeteoSwissRadar/MeteoSwissRadarHDF5/CH_360x220/"
# files = ["ch_360x220_2014.02.hdf5",
#          "ch_360x220_2014.03.hdf5",
# ]
# nam = [os.path.join(data_path, f) for f in files]

# for x in batch_sequence_multi_hdf5(nam, batch_size = 2912, shuffle=True):
#     print("----")
#     print("x:", x.shape)


def nstep_sequence_multi_hdf5(filenames, steps = 1000, batch_size = 128, shuffle=False):
    """Returns an iterable object over multiple hdf5 file until 'step' steps."""

    if shuffle:
        random.shuffle(filenames)

    n=0
    while n < steps:
        for fname in filenames:
            h5_file = h5py.File(fname, "r")
            d_rainrate = h5_file['rainrate']

            m = min(d_rainrate.shape[0], steps-n)
            for ndx in range(0, m, batch_size):
                yield (np.transpose(d_rainrate[ndx:min(ndx + batch_size, m),:], (0,2,1)))
            n += m

            h5_file.close()


# for x in nstep_sequence_multi_hdf5(nam, steps=36, batch_size=11):
#     print("shape:{}, mean: {}".format(x.shape, np.mean(x)))


# ---------------------------------
# NetCDF generator

# def batch_sequence_netcdf(filename, batch_size = 128):
#     """Returns an iterable object over a netcdf file"""

#     nc_file = netCDF4.Dataset(f5, "r")
#     d_rainrate = nc_file.variables['rain']

#     n = d_rainrate.shape[0]
#     for ndx in range(0, n, batch_size):
#         yield (d_rainrate[ndx:min(ndx + batch_size, n),:])

#     nc_file.close()



# # get the data in batches

# f5 = "/home/scheidan/RadarData/NetCDF/raincell.2014.04.nc"

# for x in batch_sequence_netcdf(f5, batch_size = 912):
#     print("----")
#     print("x:", x.shape)




# def batch_sequence_multi_netcdf(filenames, batch_size = 128, shuffle=False):
#     """Returns an iterable object over multiple netcdf files"""

#     if shuffle:
#         random.shuffle(filenames)

#     for fname in filenames:
#         nc_file = netCDF4.Dataset(fname, "r")
#         d_rainrate = nc_file.variables['rain']

#         n = d_rainrate.shape[0]
#         for ndx in range(0, n, batch_size):
#             yield (d_rainrate[ndx:min(ndx + batch_size, n),:])

#         nc_file.close()
