
import pickle
import gzip

import numpy as np
import numpy  

def load_data():

    f = gzip.open('mnist.pkl.gz', 'rb')
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    training_data, validation_data, test_data = u.load()
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)

def vectorized_result(j):
        e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def reshape_matrix(data):
    x, y = zip(*data)

    x = numpy.array(x)
    x = x.reshape(-1, 784)

    y = numpy.array(y).squeeze()
    
    return x,y
def get_mnist_data():

    training_data, validation_data, test_data = load_data_wrapper()
    train=reshape_matrix(training_data)
    validation=reshape_matrix(validation_data)
    test=reshape_matrix(test_data)
    return train[0],np.argmax(train[1],axis=1),validation[0],np.int32(validation[1].squeeze()),test[0],np.int32(test[1].squeeze())


def show_mean_image(dataset):

    import matplotlib.pyplot as plt
    
    m=np.mean(dataset)
    plt.imshow(m.reshape(28,-1),cmap='gray')
    plt.axis('off')
