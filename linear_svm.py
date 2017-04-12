import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):

  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[0]
  num_train = X.shape[1]
  loss = 0.0
  for i in xrange(num_train):
    scores = W.dot(X[:, i])
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[y[i], :] += -X[:, i]
        dW[j, :] += X[:, i]
  loss /= num_train

  loss += 0.5 * reg * np.sum(W * W)
  dW /= num_train
  dW +=reg * W
  return loss, dW


def svm_loss_vectorized(W, X, y, reg):

  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  num_classes = W.shape[0]
  num_train = X.shape[1]
  loss = 0.0
  scores_function=W.dot(X)
  col=np.arange(0,num_train)
  margins=np.maximum(0,(scores_function-scores_function[y,col])+1)
  margins[y,col]=0
  loss=np.sum(margins)
  loss/=num_train
  loss+=0.5*reg*np.sum(W*W)

  num_pos = np.sum(margins > 0, axis=0)
  dscores = np.zeros(scores_function.shape)
  dscores[margins > 0] = 1
  dscores[y, range(num_train)] = -num_pos
  dW = dscores.dot(X.T) / num_train + reg * W

  return loss, dW
