import numpy as np

def affine_forward(x, w, b):

  out = None

  x_shape = x.shape
  x = x.reshape( (x.shape[0], -1) )
  print (x.shape,x_shape,w.shape)
  out = np.dot(x, w) + b
  x = x.reshape(x_shape)

  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  
  x, w, b = cache
  dx, dw, db = None, None, None
  
  N = x.shape[0]
  D = w.shape[0]
  M = w.shape[1]

  x_shape =  x.shape
  x = x.reshape(x.shape[0], -1)
  dw = np.dot(x.T, dout)
  dx = np.dot(dout, w.T)
  db = dout.sum(axis=0)

  dx = dx.reshape(x_shape)
  #print dx.shape

  return dx, dw, db


def relu_forward(x):

  out = None
  relu=lambda x:np.maximum(x,0)
  out=relu(x)
  cache = x
  return out, cache


def relu_backward(dout, cache):

  dx, x = None, cache
  
  dx = dout * (x > 0)
    return dx


def svm_loss(x, y):
  
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx
