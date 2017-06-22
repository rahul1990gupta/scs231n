import numpy as np
from random import shuffle
from past.builtins import xrange

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
  num_classes = W.shape[1]
  num_train = X.shape[0]
  for i in xrange(num_train): 
        scores = X[i].dot(W)
        stabilityConstant = - np.max(scores)
        correct_class_score = scores[y[i]]
       # loss += -correct_class_score + np.log(np.sum(np.exp(scores)))
        loss +=  - np.log(np.exp(correct_class_score + stabilityConstant)/ np.sum(np.exp(scores + stabilityConstant )) )

        for j in xrange(num_classes):
            if j == y[i]:
                dW[:,j] += X[i]*(-1 + np.exp(correct_class_score + stabilityConstant)/np.sum(np.exp(scores + stabilityConstant)) )
            else:
                dW[:,j] += X[i]* np.exp(scores[j] + stabilityConstant)/np.sum(np.exp(scores + stabilityConstant)) 
                
  loss /= num_train
  loss += reg * np.sum(W*W)
  dW /= num_train   
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  dW += reg * 2 * W

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
    
  #loss += - np.sum( np.log(  np.choose(y, np.exp(X.W - np.max(X.W , axis = 1) ) )/ np.sum(np.exp(X.W - np.max(X.W, axis = 1)) , axis = 1) ) )
  loss = 0
    
  mat = np.exp(X.dot(W).T - np.max(X.dot(W) , axis = 1).T)   # CXN
  loss  += np.sum( - np.log( mat[y, np.arange(X.shape[0])] / np.sum(mat , axis = 0) ) )
  dW += np.dot(  mat / np.sum(mat , axis = 0).T  , X).T
  temp = np.zeros((X.shape[0], dW.shape[1])) 
  temp[np.arange(X.shape[0]),y] = 1
  dW += - np.dot(X.T, temp ) 
    
  loss /= X.shape[0]  
  dW /= X.shape[0]
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss += reg * np.sum(W*W)
  dW += reg * 2 * W

  return loss, dW

