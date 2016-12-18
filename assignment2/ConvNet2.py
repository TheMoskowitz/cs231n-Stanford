import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *

class ConvNet2(object):
  """
  the architecture will be
  
  [[conv-relu]xK-conv-relu-pool]xN - [affine - relu]xM - affine - [softmax]
  
  where batch normalization and dropout are optional
  """

  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=3,
               num_conv=1, num_pool=1, num_affine=0,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dropout=0, use_batchnorm=False, seed=None, dtype=np.float32):
    """
    Initialize a new ConvNet.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.

    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_conv = num_conv
    self.num_pool = num_pool
    self.num_affine = num_affine
    self.dtype = dtype
    self.params = {}
    self.filter_size = filter_size
    self.num_layers = (num_conv + 1) * num_pool + num_affine + 1
    self.total_conv = (num_conv + 1) * num_pool
    
    ############################################################################
    # TODO: Initialize the parameters of the network, storing all values in    #
    # the self.params dictionary. Store weights and biases for the first layer #
    # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
    # initialized from a normal distribution with standard deviation equal to  #
    # weight_scale and biases should be initialized to zero.                   #
    #                                                                          #
    # When using batch normalization, store scale and shift parameters for the #
    # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
    # beta2, etc. Scale parameters should be initialized to one and shift      #
    # parameters should be initialized to zero.                                #
    ############################################################################
    
    C, H, W = input_dim
    
    # intialize conv matrices
    for i in range((self.num_conv + 1) * self.num_pool):
        self.params['W' + str(i + 1)] = np.random.normal(0, weight_scale, (num_filters, num_filters, filter_size, filter_size))
        self.params['b' + str(i + 1)] = np.zeros(num_filters)
    
    # fix the size of the very first matrix
    self.params['W1'] = np.random.normal(0, weight_scale, (num_filters, C, filter_size, filter_size))
        
    if self.use_batchnorm:
        for i in range((self.num_conv + 1) * self.num_pool):
            self.params['gamma' + str(i + 1)] = np.ones(num_filters)
            self.params['beta' + str(i + 1)] = np.zeros(num_filters)
    
    HH = H / (2**self.num_pool)
    WW = W / (2**self.num_pool)
    
    # initialize affine matrices
    for i in range(self.total_conv, self.num_layers):
        row_dim, col_dim = hidden_dim, hidden_dim
        if i == (self.total_conv):
            row_dim = num_filters * HH * WW
        if i == (self.num_layers - 1):
            col_dim = num_classes
        self.params['W' + str(i + 1)] = np.random.normal(0, weight_scale, (row_dim, col_dim))
        self.params['b' + str(i + 1)] = np.zeros(col_dim)
        
    for k, v in self.params.iteritems():
        print k, v.shape
    
    print self.use_batchnorm, self.use_dropout
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
    
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    if self.use_batchnorm:
      self.bn_params = [[{'mode': 'train'} for i in range(self.num_conv + 1)] for j in range(self.num_pool)]
    
    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'
    
    # pass conv_param to the forward pass for the convolutional layer
    conv_param = {'stride': 1, 'pad': (self.filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode   
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        for bn in bn_param:
            bn[mode] = mode

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the fully-connected net, computing  #
    # the class scores for X and storing them in the scores variable.          #
    #                                                                          #
    # When using dropout, you'll need to pass self.dropout_param to each       #
    # dropout forward pass.                                                    #
    #                                                                          #
    # When using batch normalization, you'll need to pass self.bn_params[0] to #
    # the forward pass for the first batch normalization layer, pass           #
    # self.bn_params[1] to the forward pass for the second batch normalization #
    # layer, etc.                                                              #
    ############################################################################
    
    # create caches
    conv_caches = [[None for i in range(self.num_conv + 1)] for j in range(self.num_pool)]
    affine_caches = [None] * (self.num_affine)
    dropout_caches = [None] * (self.num_affine)
    if self.use_batchnorm:
        bn_caches = [[None for i in range(self.num_conv + 1)] for j in range(self.num_pool)]
    
    # calculate forward pass
    input_x = X
    
    # conv layers
    for i in range(self.num_pool):
        for j in range(self.num_conv + 1):
            s = str(i * (self.num_conv + 1) + j + 1)
            if j != self.num_conv:
                input_x, conv_caches[i][j] = conv_relu_forward(input_x, self.params['W' + s], self.params['b' + s], conv_param)
            else:
                input_x, conv_caches[i][j] = conv_relu_pool_forward(input_x, self.params['W' + s], self.params['b' + s], 
                                                                    conv_param, pool_param)
                
            if self.use_batchnorm:
                input_x, bn_caches[i][j] = spatial_batchnorm_forward(input_x, self.params['gamma' + s], self.params['beta' + s],
                                                                     self.bn_params[i][j])
    # affine layers
    for i in range(self.num_affine):
        s = str(i + 1 + self.total_conv)
        input_x, affine_caches[i] = affine_relu_forward(input_x, self.params['W' + s], self.params['b' + s])
        if self.use_dropout:
            input_x, dropout_caches[i] = dropout_forward(input_x, self.dropout_param)
        
    # last FC layer
    s = str(self.num_layers)
    scores, last_cache = affine_forward(input_x, self.params['W'+s], self.params['b'+s])

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If test mode return early
    if mode == 'test':
      return scores

    loss, grads = 0.0, {}
    ############################################################################
    # TODO: Implement the backward pass for the fully-connected net. Store the #
    # loss in the loss variable and gradients in the grads dictionary. Compute #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    #                                                                          #
    # When using batch normalization, you don't need to regularize the scale   #
    # and shift parameters.                                                    #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    
    # data loss
    loss, dx = softmax_loss(scores, y)

    # last layer
    s = str(self.num_layers)
    dx, grads['W'+s], grads['b'+s] = affine_backward(dx, last_cache)  
    grads['W'+s] += self.reg * self.params['W'+s]
    loss += 0.5 * self.reg * np.sum(self.params['W' + s] * self.params['W' + s])
    
    
    # affine layers
    for i in reversed(range(self.num_affine)):
        s = str(i + 1 + self.total_conv)
        if self.use_dropout:
            dx = dropout_backward(dx, dropout_caches[i])
        
        dx, grads['W' + s], grads['b' + s] = affine_relu_backward(dx, affine_caches[i])
        grads['W' + s] += self.reg * self.params['W' + s]
        loss += 0.5 * self.reg * np.sum(self.params['W' + s] * self.params['W' + s])

    # conv layers
    for i in reversed(range(self.num_pool)):
        for j in reversed(range(self.num_conv + 1)):
            s = str(i * (self.num_conv + 1) + j + 1)
            if self.use_batchnorm:
                dx, grads['gamma' + s], grads['beta' + s] = spatial_batchnorm_backward(dx, bn_caches[i][j])
            if j != self.num_conv:
                dx, grads['W' + s], grads['b' + s] = conv_relu_backward(dx, conv_caches[i][j])
            else:
                dx, grads['W' + s], grads['b' + s] = conv_relu_pool_backward(dx, conv_caches[i][j])
            grads['W' + s] += self.reg * self.params['W' + s]
            loss += 0.5 * self.reg * np.sum(self.params['W' + s] * self.params['W' + s])
        
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads
