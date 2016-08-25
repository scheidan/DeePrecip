## -------------------------------------------------------
##
## Elements to implement a spatial transformer network
##
## See: Jaderberg, M., Simonyan, K., Zisserman, A., and Kavukcuoglu,
## K. (2015) Spatial Transformer Networks. arXiv:1506.02025
##
## February  9, 2016 -- Andreas Scheidegger
## andreas.scheidegger@eawag.ch
## -------------------------------------------------------


import numpy as np
from chainer import cuda, Function, Variable, Link
import chainer.functions as F

# ---------------------------------
# Grid transformation


def expand_grid(dim, polynomial=None):
    """
    Returns a Grid representation of the coordiantes normalized to -1 ... 1.

    If the argument `polynomial` is given, higher order terms are added.
    (Set `polynomial=1` for bilinear transformation.)
    The output dimensions are (`polynomial`+1)^2 x prod(`dim`)
    or 3 x prod(`dim`).
    """

    x = np.linspace(-1, 1, dim[1])
    y = np.linspace(-1, 1, dim[0])
    xG, yG = np.meshgrid(x, y)
    xG = xG.flatten()
    yG = yG.flatten()
    if polynomial==None:        # linear
        G = np.vstack((xG, yG, np.ones(np.prod(dim))))
    else:
        G = np.asarray([xG**i * yG**j for i in range(polynomial+1) for j in range(polynomial+1)])
        G[[0,polynomial+1],:] = G[[polynomial+1,0],:]
    return G.astype("float32")

def thin_plates(dim, dim_control_points, type="gaussian"):
    """
    Returns a Grid representation of the coordiantes normalized to -1 ... 1,
    augmented with non linear terms for thin plates splines.

    The output dimensions are prod(`dim_control_points`)+3 x prod(`dim`).
    """
    gr = expand_grid(dim, polynomial=None)
    g_cp = expand_grid(dim_control_points, polynomial=None)
    g_cp[0,:] *= 2.0/(dim_control_points[0]+1)
    g_cp[1,:] *= 2.0/(dim_control_points[1]+1)

    gr = np.vstack((gr, np.zeros((g_cp.shape[1], gr.shape[1]))))
    sx = 2.0/(dim_control_points[0]+1)
    sy = 2.0/(dim_control_points[1]+1)

    for i in range(gr.shape[1]):
        for j in range(g_cp.shape[1]):
            dist = np.abs( ((gr[0,i]-g_cp[0,j])/sx)**2 + ((gr[1,i]-g_cp[1,j])/sy)**2 )
            if type=="gaussian":
                gr[3+j,i] = np.exp(-np.power(dist, 2)) # gaussian
            elif type=="thinplate":
                gr[3+j,i] = dist**2 * np.log(dist+0.00001) # thin_plates

            ## gr[3+j,i] = np.max(1-dist/0.2, 0) # linear

    return gr.astype("float32")


class LearnableTargetGrid(Link):
    """
    Holds coordinates of the transformed image, augmented
    with non-linear terms. `dimout` defines the output shape of
    the interpolation.

    The coordinates of the transformed image are changed during training!
    """
    def __init__(self, dimout, polynomial=None, thin_plate=False, dim_control_points=(3,3)):
        if not thin_plate:
            d = 3 if polynomial==None else (polynomial+1)**2
        else:
            d = 3 + np.prod(dim_control_points)

        super(TargetGrid, self).__init__(
            g_target=(d, np.prod(dimout)),
        )
        if not thin_plate:
            self.g_target.data[...] = expand_grid(dimout, polynomial)
        else:
            self.g_target.data[...] = thin_plates(dimout, dim_control_points)

    def __call__(self):
        return self.g_target


def transform_grid(A, target_grid):
    """
    Generate an array of coordinates of the sampling points.
    The transformation matrix 'A' is applied to the coordinates
    in 'target_grid'.

    If 'target_grid must be of shape (X, prod(target_dims))'
    'A' must be of shape (2,X).

    The identity matrix below result in no transformation:
     [[1, 0, 0, 0, ...],
      [0, 1, 0, 0, ...]]
    """
    xp = cuda.get_array_module(A.data)
    target_grid_var = Variable(xp.array(target_grid))
    G = F.matmul(A, target_grid_var)
    return G




# ---------------------------------
# Interpolate at sampling grid
# input U at grid G_sample

# see Lasagne code here:
# https://github.com/Lasagne/Lasagne/blob/master/lasagne/layers/special.py#L232-L309

class Interpolate(Function):

    def forward_cpu(self, inputs):

        U, grid = inputs

        height= U.shape[0]
        width = U.shape[1]

        x = grid[0,:]
        y = grid[1,:]

        # clip coordinates to [-1, 1], i.e. edge pixels are repeated.
        x = x.clip(-1, 1)
        y = y.clip(-1, 1)

        # rescale coordiantes from [-1, 1] to [0, width/height - 1]
        # (The factor 0.9999 ensures that the end points are always
        # floored to the same side.)
        x = (x*0.9999+1)/2 * (width-1)
        y = (y*0.9999+1)/2 * (height-1)

        # indices of the 2x2 pixel neighborhood surrounding the coordinates
        x0 = np.floor(x)
        x1 = x0+1
        y0 = np.floor(y)
        y1 = y0+1

        # get weights
        w1 = (x1-x) * (y1-y)
        w2 = (x-x0) * (y1-y)
        w3 = (x1-x) * (y-y0)
        w4 = (x-x0) * (y-y0)

        V = np.zeros(grid.shape[1]).astype("float32")
        for i in range(grid.shape[1]):
            V[i] = w1[i]*U[y0[i],x0[i]] + w2[i]*U[y0[i],x1[i]] + \
                   w3[i]*U[y1[i],x0[i]] + w4[i]*U[y1[i],x1[i]]

        return V,


    def backward_cpu(self, inputs, grad_outputs):

        U, grid = inputs
        gV, = grad_outputs      # same dimension as V

        height= U.shape[0]
        width = U.shape[1]

        x = grid[0,:]
        y = grid[1,:]

        # clip coordinates to [-1, 1], i.e. edge pixels are repeated.
        x = x.clip(-1, 1)
        y = y.clip(-1, 1)

        # rescale coordiantes from [-1, 1] to [0, width/height - 1]
        # (The factor 0.9999 ensures that the end points are always
        # floored to the same side.)
        x = (x*0.9999+1)/2 * (width-1)
        y = (y*0.9999+1)/2 * (height-1)

        # indices of the 2x2 pixel neighborhood surrounding the coordinates
        x0 = np.floor(x)
        x1 = x0+1
        y0 = np.floor(y)
        y1 = y0+1

        # weights
        wx0 = (x1-x)
        wx1 = (x-x0)
        wy0 = (y1-y)
        wy1 = (y-y0)

        # --- gx, gy
        gx = np.zeros(grid.shape[1]).astype("float32")
        gy = np.zeros(grid.shape[1]).astype("float32")
        for i in range(grid.shape[1]):
            gx[i] =  - wy0[i] * U[y0[i],x0[i]] \
                     + wy0[i] * U[y0[i],x1[i]] \
                     - wy1[i] * U[y1[i],x0[i]] \
                     + wy1[i] * U[y1[i],x1[i]]

            gy[i] =  - wx0[i] * U[y0[i],x0[i]] \
                     - wx1[i] * U[y0[i],x1[i]] \
                     + wx0[i] * U[y1[i],x0[i]] \
                     + wx1[i] * U[y1[i],x1[i]]

        gx = gx * gV
        gy = gy * gV
        ggrid = np.vstack((gx, gy))

        # --- gU

        gU = np.zeros((height, width)).astype("float32")

        for cx in range(width):
            for cy in range(height):
                select_q1 = (x >= cx) & (x < cx+1) & (y <= cy) & (y > cy-1)
                select_q2 = (x <= cx) & (x > cx-1) & (y <= cy) & (y > cy-1)
                select_q3 = (x <= cx) & (x > cx-1) & (y >= cy) & (y < cy+1)
                select_q4 = (x >= cx) & (x < cx+1) & (y >= cy) & (y < cy+1)
                gU[cy,cx] = np.sum(wx0[select_q1]*wy1[select_q1]*gV[select_q1]) + \
                            np.sum(wx1[select_q2]*wy1[select_q2]*gV[select_q2]) + \
                            np.sum(wx1[select_q3]*wy0[select_q3]*gV[select_q3]) + \
                            np.sum(wx0[select_q4]*wy0[select_q4]*gV[select_q4])

        return gU, ggrid



    def forward_gpu(self, inputs):

        U, grid = inputs

        height= U.shape[0]
        width = U.shape[1]

        x = grid[0,:]
        y = grid[1,:]

        # clip coordinates to [-1, 1], i.e. edge pixels are repeated.
        x = x.clip(-1, 1)
        y = y.clip(-1, 1)

        # rescale coordiantes from [-1, 1] to [0, width/height - 1]
        # (The factor 0.9999 ensures that the end points are always
        # floored to the same side.)
        x = (x*0.9999+1)/2 * (width-1)
        y = (y*0.9999+1)/2 * (height-1)

        # indices of the 2x2 pixel neighborhood surrounding the coordinates
        x0 = cuda.cupy.floor(x).astype("int32")
        x1 = x0+1
        y0 = cuda.cupy.floor(y).astype("int32")
        y1 = y0+1

        # get weights
        w1 = (x1-x) * (y1-y)
        w2 = (x-x0) * (y1-y)
        w3 = (x1-x) * (y-y0)
        w4 = (x-x0) * (y-y0)

        kern = cuda.cupy.ElementwiseKernel(
            'raw T U, T w1, T w2, T w3, T w4, int32 x0, int32 x1, int32 y0, int32 y1, int32 N',
            'T V',
            'V = w1*U[y0*N+x0] + w2*U[y0*N+x1] + w3*U[y1*N+x0] + w4*U[y1*N+x1]',
            'compute_V'
        )
        V = kern(U,
                 w1.astype("float32"), w2.astype("float32"),
                 w3.astype("float32"), w4.astype("float32"),
                 x0, x1, y0, y1, U.shape[1])

        return V,



    def backward_gpu(self, inputs, grad_outputs):

        U, grid = inputs
        gV, = grad_outputs      # same dimension as V

        if len(U.shape)!=2:
            print("backward")
            print(type(U))
            print(U.shape)

        height= U.shape[0]
        width = U.shape[1]

        x = grid[0,:]
        y = grid[1,:]

        # clip coordinates to [-1, 1], i.e. edge pixels are repeated.
        x = x.clip(-1, 1)
        y = y.clip(-1, 1)

        # rescale coordiantes from [-1, 1] to [0, width/height - 1]
        # (The factor 0.9999 ensures that the end points are always
        # floored to the same side.)
        x = (x*0.9999+1)/2 * (width-1)
        y = (y*0.9999+1)/2 * (height-1)

        # indices of the 2x2 pixel neighborhood surrounding the coordinates
        x0 = cuda.cupy.floor(x).astype("int32")
        x1 = x0+1
        y0 = cuda.cupy.floor(y).astype("int32")
        y1 = y0+1

        # weights
        wx0 = (x1-x)
        wx1 = (x-x0)
        wy0 = (y1-y)
        wy1 = (y-y0)

        # --- gx, gy

        gx_kern = cuda.cupy.ElementwiseKernel(
            'raw T U, T wy0, T wy1, int32 x0, int32 x1, int32 y0, int32 y1, T gV, int32 N',
            'T gx',
            'gx = gV * (-wy0*U[y0*N+x0] + wy0*U[y0*N+x1] - wy1*U[y1*N+x0] + wy1*U[y1*N+x1])',
            'compute_gx'
        )
        gx = gx_kern(U, wy0.astype("float32"), wy1.astype("float32"),
                     x0, x1, y0, y1, gV, U.shape[1])

        gy_kern = cuda.cupy.ElementwiseKernel(
            'raw T U, T wx0, T wx1, int32 x0, int32 x1, int32 y0, int32 y1, T gV, int32 N',
            'T gy',
            'gy = gV * (-wx0*U[y0*N+x0] - wx1*U[y0*N+x1] + wx0*U[y1*N+x0] + wx1*U[y1*N+x1])',
            'compute_gx'
        )
        gy = gy_kern(U, wx0.astype("float32"), wx1.astype("float32"),
                     x0, x1, y0, y1, gV, U.shape[1])

        ggrid = cuda.cupy.vstack((gx, gy))


        # --- gU

        gU = cuda.cupy.zeros((height, width)).astype("float32")
        # z = cuda.cupy.zeros_like(wx1)
        # for cx in range(width):
        #     for cy in range(height):
        #         select_q1 = (x >= cx) & (x < cx+1) & (y <= cy) & (y > cy-1)
        #         select_q2 = (x <= cx) & (x > cx-1) & (y <= cy) & (y > cy-1)
        #         select_q3 = (x <= cx) & (x > cx-1) & (y >= cy) & (y < cy+1)
        #         select_q4 = (x >= cx) & (x < cx+1) & (y >= cy) & (y < cy+1)

        #         gU[cy,cx] = cuda.cupy.sum(cuda.cupy.where(select_q1, wx0*wy1*gV, z)) + \
        #                     cuda.cupy.sum(cuda.cupy.where(select_q2, wx1*wy1*gV, z)) + \
        #                     cuda.cupy.sum(cuda.cupy.where(select_q3, wx1*wy0*gV, z)) + \
        #                     cuda.cupy.sum(cuda.cupy.where(select_q4, wx0*wy0*gV, z))

        # gU_kern = cuda.cupy.ElementwiseKernel(
        #     'T wx0, T wx1, T wy0, T wy1, int32 x0, int32 x1, int32 y0, int32 y1, T gV, int32 N',
        #     'raw T gU',
        #     '''
        #     for (int i = 0; i < N; ++i) {
        #        gU[i] = 0.0;
        #     }
        #     /* for (int i = 0; i < N; ++i) {
        #        U[y0[i]*n+x0[i]] = U[y0[i]*n+x0[i]] + wx0[i]*wy0[i]*gV[i];
        #        U[y0[i]*n+x1[i]] = U[y0[i]*n+x1[i]] + wx1[i]*wy0[i]*gV[i];
        #        U[y1[i]*n+x0[i]] = U[y1[i]*n+x0[i]] + wx0[i]*wy1[i]*gV[i];
        #        U[y1[i]*n+x1[i]] = U[y1[i]*n+x1[i]] + wx1[i]*wy1[i]*gV[i];
        #     } */
        #     ''',
        #     'compute_gU'
        # )
        # gU = gU_kern(wx0.astype("float32"), wx1.astype("float32"),
        #              wy0.astype("float32"), wy1.astype("float32"),
        #              x0, x1, y0, y1, gV, x0.shape[0])

        # gU = cuda.cupy.reshape(gU, (height, width))


        return gU, ggrid


# Wrapper
def interpolate(U, grid):
    """Sample from input feature map 'U' at coordinates of 'grid'
       applying bilinear interpolation, see:
       Jaderberg, M., Simonyan, K., Zisserman, A., and Kavukcuoglu,
       K. (2015) Spatial Transformer Networks. arXiv:1506.02025

       'U' has shape HxW, 'grid' is a two column matrix containing
       (normalized) sampling coordinates. """
    return Interpolate()(U, grid)
