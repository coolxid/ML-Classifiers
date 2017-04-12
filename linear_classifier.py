import numpy as np

class LinearClassifier:

  def __init__(self):
    self.W = None

  def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):

    dim, num_train = X.shape
    num_classes = np.max(y) + 1 # assume y takes values 0...K-1 where K is number of classes
    if self.W is None:
      # lazily initialize W
      self.W = np.random.randn(num_classes, dim) * 0.001

    # Run stochastic gradient descent to optimize W
    loss_history = []
    for it in xrange(num_iters):
      X_batch = None
      y_batch = None

      rand_idx = np.random.choice(num_train, batch_size)
      X_batch = X[:, rand_idx]
      y_batch = y[rand_idx]


      # evaluate loss and gradient
      loss, grad = self.loss(X_batch, y_batch, reg)
      loss_history.append(loss)

      self.W += -1 * learning_rate * grad
    

      if verbose and it % 100 == 0:
        print 'iteration %d / %d: loss %f' % (it, num_iters, loss)

    return loss_history

  def predict(self, X):

    y_pred = np.zeros(X.shape[1])
    y_pred = np.zeros(X.shape[1])
    scores_function = self.W.dot(X)
    y_pred = np.argmax(scores_function, axis=0) # top scoring class
    return y_pred

    return y_pred
  
  def loss(self, X_batch, y_batch, reg):

    pass


class LinearSVM(LinearClassifier):
  """ A subclass that uses the Multiclass SVM loss function """

  def loss(self, X_batch, y_batch, reg):
    return svm_loss_vectorized(self.W, X_batch, y_batch, reg)

  def svm_loss_vectorized(W, X, y, reg):
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero
    num_classes = W.shape[0]
    num_train = X.shape[1]
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


class Softmax(LinearClassifier):
  """ A subclass that uses the Softmax + Cross-entropy loss function """

  def loss(self, X_batch, y_batch, reg):
    return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)
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
