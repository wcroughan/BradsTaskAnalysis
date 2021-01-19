import numpy as np
import struct


def parseMDAFile(filename):
    dtypes = ["unknown", "b", "f", "h", "i", "H", "d", "I"]
    np_dtypes = [np.dtype(np.int32), np.dtype(np.int8), np.dtype(np.float), np.dtype(
        np.int16), np.dtype(np.int32), np.dtype(np.uint16), np.dtype(np.double), np.dtype(np.uint32)]

    with open(filename, mode='rb') as f:
        headerbytes = f.read(3*4)
        headervals = [i[0] for i in struct.iter_unpack("i", headerbytes)]

        dti = -(headervals[0]) - 1
        fmt = dtypes[dti]

        if not (struct.calcsize(fmt) == headervals[1]):
            raise Exception("Sizes don't match in mda header")

        num_dims = headervals[2]

        if not (num_dims >= 1 and num_dims <= 50):
            raise Exception("Invalid num dims in mda header")

        dim_sz_bytes = f.read(num_dims * 4)
        dim_sz = [i[0] for i in struct.iter_unpack("i", dim_sz_bytes)]

        res = np.fromfile(f, dtype=np_dtypes[dti])
        for i, ds in enumerate(dim_sz):
            if ds == 0:
                dim_sz[i] = -1
                break

        res = np.reshape(res, dim_sz)

        return res
