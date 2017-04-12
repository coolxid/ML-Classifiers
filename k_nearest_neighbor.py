import numpy as np
from scipy import stats
from collections import Counter

class KNearestNeighbor:
  """ a kNN classifier with L2 distance """

  def __init__(self):
    pass

  def train(self, X, y):

    self.X_train = X
    self.y_train = y
    
  def predict(self, X, k, num_loops=0):

    if num_loops == 0:
      dists = self.compute_distances_no_loops(X)
    elif num_loops == 1:
      dists = self.compute_distances_one_loop(X)
    elif num_loops == 2:
      dists = self.compute_distances_two_loops(X)
    else:
      raise ValueError('Invalid value %d for num_loops' % num_loops)

    return self.predict_labels(dists, k=k)

  def compute_distances_two_loops(self, X):
 
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in xrange(num_test):
      for j in xrange(num_train):

        dists[i,j]=np.sqrt(np.sum(np.square(self.X_train[j]-X[i])))

    return dists

  def compute_distances_one_loop(self, X):

    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in xrange(num_test):

      dists[i,:]=np.sqrt(np.sum(np.square(self.X_train-X[i]),axis=1))

    return dists

  def compute_distances_no_loops(self, X):

    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train)) 

    print "X SHAPE",X.shape
    ab = X.dot(self.X_train.T)
    a2 = np.sum(self.X_train ** 2, axis=1)
    b2 = np.sum(X ** 2, axis=1)
    dists = np.sqrt(a2 + b2.reshape(-1, 1)-2*ab)

    return dists

  def predict_labels(self, dists, k):

    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in xrange(num_test):

      closest_y = []


      closest_y = self.y_train[np.argsort(dists[i])[:k]]
  
      y_pred[i] = np.argmax(np.bincount(closest_y))


    return y_pred

