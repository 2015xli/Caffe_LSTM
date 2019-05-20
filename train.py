import os, sys, importlib
import caffe
import numpy as np

from mnist import MnistInput

def callfunc(path, model_name, **kwparams):
    sys.path.append(os.path.abspath(path))
    mymod = importlib.import_module(model_name)
    result = getattr(mymod, model_name)(**kwparams)
    return result

def generate_model(py_model, **kwparams):
    path, pyfile = os.path.split(py_model)
    model_name = os.path.splitext(pyfile)[0]
    netproto = callfunc(path, model_name, **kwparams)
    netproto = str(netproto)
    netfile = model_name + '.prototxt'

    with open(netfile, 'w') as f:
        f.write('# Generated network model: {} \n'.format(model_name))
        f.write(netproto)

    return


def compute_result(conf_matrix):
    m = conf_matrix
    truth = m.sum(axis=0)
    recall = [m[i, i]/truth[i] for i in range(truth.shape[0])]
    preds = m.sum(axis=1)
    FDR = [1-m[i, i]/preds[i] for i in range(preds.shape[0])]
    accuracy = m.trace()/m.sum()
    return recall, FDR, accuracy


def compute_confmatrix(n_classes, truths, preds):
    conf_matrix = np.zeros(shape=(n_classes, n_classes))

    for i in range(n_classes):
        pred_to_i = truths.flatten()[preds == i]
        bincount = np.bincount(pred_to_i)
        row_i = np.zeros(n_classes)
        row_i[np.arange(bincount.shape[0])] = bincount   # pres_to_i may have less elements than n_classes
        conf_matrix[i] = row_i

    return conf_matrix


def run_one_batch(solver, params, data, is_training):

    if is_training:
        solver.net.blobs['data'].data[...] = data[0]                #X
        solver.net.blobs['label'].data[...] = data[1].astype(int)   #Y
        if params['recur_steps'] != 1:
            solver.net.blobs['clip'].data[...] = data[2].astype(int)   #C

        solver.step(1)
        prob = solver.net.blobs['class_prob'].data

    else:
        solver.test_nets[0].blobs['data'].data[...] = data[0]                #X
        solver.test_nets[0].blobs['label'].data[...] = data[1].astype(int)   #Y
        if params['recur_steps'] != 1:
            solver.test_nets[0].blobs['clip'].data[...] = data[2].astype(int)   #C

        # to work around Caffe's bug when restoring training
        # solver.test_nets[0].share_with(solver.net)
        solver.test_nets[0].forward()
        prob = solver.test_nets[0].blobs['class_prob'].data

    preds = np.argmax(prob, axis=1)
    truths = data[1]
    conf_matrix = compute_confmatrix(params['n_classes'], truths, preds)

    return conf_matrix


def run_one_epoch(solver, params, data, is_training):
    # (X, Y, [C]) = data
    X = data[0]; Y = data[1]
    num_batches = X.shape[0]
    conf_matrix = np.zeros(shape=(params['n_classes'], params['n_classes']))

    #num_batches = 3
    for i in range(num_batches):
        if params['recur_steps'] != 1:
            one_batch = (X[i], Y[i], data[2][i])
        else:
            one_batch = (X[i], Y[i])

        conf_matrix += run_one_batch(solver, params, one_batch, is_training)

    return conf_matrix


def train_and_valid(solver, data_train, data_valid, params):
    n_epochs = params['n_epochs']
    for i in range(n_epochs):
        print "Epochs {:d}:".format(i)

        train_confmat = run_one_epoch(solver, params, data_train, is_training=True)
        valid_confmat = run_one_epoch(solver, params, data_valid, is_training=False)
        train_accuracy = compute_result(train_confmat)[-1]
        valid_accuracy = compute_result(valid_confmat)[-1]

        print "     Accuracy of training: {:4.2f}, validation {:4.2f}".format(train_accuracy, valid_accuracy)

    return


def main():
    py_model = "./cnn_lstm.py"
    solverfile = "./solver.prototxt"
    params = {
        'kernel_size': 5,
        'n_output': 28,
        'batch_size': 300,
        'recur_steps': 4,
        'n_classes': 10,
        'n_cols': 28,
        'n_rows': 28,
        'n_epochs': 5
    }

    generate_model(py_model, **params)
    solver = caffe.get_solver(solverfile)

    data_train = MnistInput("train").prepare(params['batch_size'], params['recur_steps'])
    data_valid = MnistInput("test").prepare(params['batch_size'], params['recur_steps'])

    train_and_valid(solver, data_train, data_valid, params)
    return


if __name__ == "__main__":
    main()