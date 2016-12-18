import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  for i in range(y.size):
        scores = X[i].dot(W)
        ex_scores = np.exp(scores)
        probs = ex_scores / np.sum(ex_scores)
        loss -= np.log(probs[y[i]])
        for j in range(W.shape[1]):
            dW[:, j] += probs[j] * X[i]
            if (j == y[i]):
                dW[:, j] -= X[i]
                
  loss /= y.size
  dW /= y.size
  loss += .5 * reg * np.sum(W * W)
  dW += reg * W
        

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  N = y.size
  C = W.shape[1]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  scores = X.dot(W)
  ex_scores = np.exp(scores)
  probs = (ex_scores.T / np.sum(ex_scores, 1)).T
  loss -= np.sum(np.log(probs[np.arange(N), y])) / N
  
  dW = X.T.dot(probs)
  bool = np.zeros(probs.shape)
  bool[np.arange(N), y] = 1
  dW -= X.T.dot(bool)
  dW /= N
  
  loss += .5 * reg * np.sum(W * W)
  dW += reg * W
  
  

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

