import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):

  loss = 0.0
  dW = np.zeros_like(W)
  scores_function = W.dot(X)
  max_scores = np.max(scores_function, axis=0)
  scores_function -= max_scores
  exp_scores = np.exp(scores_function)
  sums = np.sum(exp_scores, axis=0)
  log_sums = np.log(sums)
  scores_y = np.array([scores_function[y[i],i] for i in xrange(X.shape[1])])
  loss = np.sum(-scores_y + log_sums)
  loss /= X.shape[1]
  loss += .5 * reg* np.sum(W * W)

  classes=np.unique(y)
  nclasses=len(classes)

  one=np.zeros((y.shape[0],nclasses))
  for i, l in enumerate(y):        
      one[i,classes==l]=1
    
  dW=(-1)*(one-(exp_scores/sums).T).T.dot(X.T)
  dW /= X.shape[1]
  dW += reg * np.sum(W)   

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):

  loss = 0.0
  dW = np.zeros_like(W)

  scores_function = W.dot(X)
  max_scores = np.max(scores_function, axis=0)
  scores_function -= max_scores
  exp_scores = np.exp(scores_function)
  sums = np.sum(exp_scores, axis=0)
  log_sums = np.log(sums)
  #scores_y = np.array([scores[y[i],i] for i in xrange(X.shape[1])])
  scores_y=np.array(scores_function[y,xrange(X.shape[1])])
  loss = np.sum(-scores_y + log_sums)
  loss /= X.shape[1]
  loss += .5 * reg* np.sum(W * W)

  classes=np.unique(y)
  nclasses=len(classes)
  col=np.arange(0,X.shape[1])
  one=np.zeros((y.shape[0],nclasses))
  '''for i, l in enumerate(y):        
      one[i,classes==l]=1'''
  one[col,y]=1

  dW=(-1)*(one-(exp_scores/sums).T).T.dot(X.T)
  dW /= X.shape[1]
  dW += reg * np.sum(W)   

  return loss, dW