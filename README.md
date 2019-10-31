## Simple LSTM network tutorial code with Caffe for MNIST 

Lots of people have confusions when using Caffe to implement LSTM network. The python code here uses MNIST as an example to show how to use Caffe LSTM operator. The simple network can achieve 96% validation accuracy in 4 epochs. To run it, invoke "train.py" in command line.

The key confusion usually comes from the data structure requirement in Caffe LSTM, where the input shape for a batch is (time_step, batch_size, data_shape), instead of (batch_size, time_step, data_shape). In this tutorial code, the original data shape of an image for Caffe is (1, 28, 28). When it is split into a sequence of steps, say 7, the new data_shape becomes (1, 7, 4, 28) or (7, 1, 4, 28), indicating 7 steps of sub-image data (1, 4, 28). If it has a batch_size 300, the vanila input shape is (300, 7, 1, 4, 28). Now for Caffe LSTM, it should be (7, 300, 1, 4, 8). That is, at every time step, 300 sub-images (1, 4, 8) are fed into the network. Since the sub-images of an image should be fed into the network in order, the data (300, 7, 1, 4, 8) cannot be np.reshaped into (7, 300, 1, 4, 8); instead, np.transpose(1,0,2,3,4) or np.swapaxes(0,1) should be used.

Other questions in using Caffe LSTM like how to deal with its output, how to connect it with other layers etc., can also get answers from the tutorial code.

Note, the network model prototxt file is generated at runtime and saved to current directory. One can use it directly, while losing a little bit flexibility, since the model is associated with specific input data shape. The code generates the model and the input data using same set of a few variables, so that they always match with each other.
