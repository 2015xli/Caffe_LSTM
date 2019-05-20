import gzip
import struct
import numpy as np
import os

def data_shuffle(X, Y):
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    return X[idx], Y[idx]

# pre-requirement: MNIST data files stored in local directory under $folder/mnist/
# after downloaded from http://yann.lecun.com/exdb/mnist/
class MnistInput:
    def __init__(self, data, folder='.'):
        if data == "train":
            zX = os.path.join(folder, 'mnist', 'train-images-idx3-ubyte.gz')
            zY = os.path.join(folder, 'mnist', 'train-labels-idx1-ubyte.gz')
        elif data == "test":
            zX = os.path.join(folder, 'mnist', 't10k-images-idx3-ubyte.gz')
            zY = os.path.join(folder, 'mnist', 't10k-labels-idx1-ubyte.gz')
        else:
            raise ValueError("Incorrect data input")

        self.zX = zX
        self.zY = zY
        self.has_read = False
        return

    def read(self, num=-1):  # -1 to read all

        if self.has_read == True: return

        zX = self.zX
        zY = self.zY
        with gzip.open(zX) as fX, gzip.open(zY) as fY:
            magic, nX, rows, cols = struct.unpack(">IIII", fX.read(16))
            magic, nY = struct.unpack(">II", fY.read(8))
            if nX != nY: raise ValueError("Inconsistent data and label files")

            img_size = cols * rows
            if num <= 0 or num > nX: num = nX
            X = struct.unpack("B" * img_size * num, fX.read(img_size * num))
            X = np.array(X).reshape(num, rows, cols)
            Y = struct.unpack("B" * num, fY.read(num))
            Y = np.array(Y).reshape(num, 1)

        self.X = X
        self.Y = Y
        self.has_read = True
        return

    def prepare_lstm(self, recur_steps):
        X = self.X
        num_batches, batch_size, depth, rows, cols = X.shape
        assert depth == 1 and rows%recur_steps == 0  # rows = recur_steps * new_rows to make it simple
        new_rows = rows/recur_steps
        frame_shape = (depth, new_rows, cols)

        # Caffe LSTM requires X shape to be T*N*(Frame), i.e., (recur_steps, batch_size, frame_shape)
        X = X.reshape((num_batches,) + (batch_size, recur_steps) + frame_shape)
        X = X.swapaxes(1,2)  # use transpose to keep the sub-frames of different batches aligned at same time step
        X = X.reshape((num_batches,) + (recur_steps * batch_size,) + frame_shape) # flatten for CNN
        # create continuation indicator for each sequence (here an image)
        C = np.ones(shape=(num_batches, recur_steps, batch_size))
        C[:, 0, :] = 0
        self.X = X
        self.C = C

        return X, self.Y, C

    def prepare(self, batch_size=300, recur_steps=7):

        self.read()

        X = self.X
        Y = self.Y
        # keep only multiples of batch_size
        num_batches = X.shape[0] / batch_size
        num_total = num_batches * batch_size
        X = X[:num_total, :, :]
        Y = Y[:num_total, :]

        # X, Y = data_shuffle(X, Y)  # shuffle so that every batch includes multiple classes
        X = np.expand_dims(X, 1)  # depth=1 for caffe to work
        X = X.reshape((num_batches, batch_size) + X.shape[1:])
        Y = Y.reshape((num_batches, batch_size) + Y.shape[1:])

        self.X = X
        self.Y = Y
        if recur_steps == 1:
            return X, Y

        # reshape for caffe LSTM
        return self.prepare_lstm(recur_steps)

if __name__ == "__main__":
    traininput = MnistInput("train")
    trainX, trainY, trainC = traininput.prepare()
    testinput = MnistInput("test")
    testX, testY, testC = traininput.prepare()
