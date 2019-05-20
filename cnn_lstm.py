import caffe
from caffe import layers as L
from caffe import params as P


def cnn_lstm(**args):
    n_rows = args.get('n_rows', 28)
    n_cols = args.get('n_cols', 28)
    kernel_size = args.get('kernel_size', 5)
    n_output = args.get('n_output', n_cols)
    batch_size = args.get('batch_size', 300)
    recur_steps = args.get('recur_steps', 4)
    n_classes = args.get('n_classes', 10)
    frame_shape = args.get('frame_shape', (1, n_rows/recur_steps, n_cols))

    n = caffe.NetSpec()

    weight_param = dict(lr_mult=1, decay_mult=1)
    bias_param = dict(lr_mult=1, decay_mult=0)
    param = [weight_param, bias_param]

    N = batch_size;
    T = recur_steps;
    input_dim = list(frame_shape)
    input_dim.insert(0, N*T)

    n.data, n.clip, n.label = L.Input(
        shape=[dict(dim=input_dim),
               dict(dim=[T, N]),
               dict(dim=[N, 1])], ntop=3)

    n.conv1 = L.Convolution(n.data, kernel_size=[1, kernel_size],
                            stride_h=1, stride_w=1, num_output=n_output,
                            pad=0, param=param,
                            weight_filler=dict(type='msra'),
                            bias_filler=dict(type='constant'))
    n.relu1 = L.ReLU(n.conv1, in_place=True)

    n.conv2 = L.Convolution(n.relu1, kernel_size=[1, kernel_size],
                            stride_h=1, stride_w=1, num_output=n_output,
                            pad=0, param=param,
                            weight_filler=dict(type='msra'),
                            bias_filler=dict(type='constant'))
    n.relu2 = L.ReLU(n.conv2, in_place=True)

    n.pool1 = L.Pooling(n.relu2, kernel_h=1, kernel_w=2, stride_h=1, stride_w=2, pool=P.Pooling.MAX)

    n.fc1 = L.InnerProduct(n.pool1, num_output=n_output, param=param,
                           weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))

    n.fc1_relu = L.ReLU(n.fc1, in_place=True)

    n.fc1_reshape = L.Reshape(n.fc1_relu, shape=dict(dim=[T, N, n_output]))

    n.lstm = L.LSTM(n.fc1_reshape, n.clip, recurrent_param=dict(num_output=n_output))

    n.lstm_last_step = L.Slice(n.lstm, slice_param=dict(axis=0, slice_point=T-1), ntop=2)[-1]

    n.lstm_reshape = L.Reshape(n.lstm_last_step, shape=dict(dim=[N, n_output]))

    n.attrs = L.InnerProduct(n.lstm_reshape, num_output=n_classes, param=param,
                             weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))

    n.loss = L.SoftmaxWithLoss(n.attrs, n.label)

    n.class_prob = L.Softmax(n.attrs, in_place=False)

    return n.to_proto()

if __name__ == "__main__":

    proto = cnn_lstm()
    print proto