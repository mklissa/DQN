import pdb
import numpy as np
import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d

class ConvLayer(object):
    
    def __init__(self, input, filter_shape, image_shape, stride, layer_num, W=None,b=None):
        """
            input: a 4D tensor explicited by image_shape
            filter_shape: tuple representing (current number of feature maps, previous number of feature maps, height, width)
            image_shape: tuple representing (batch_size, number of feature maps, height, width), use None for a non-constant shape
            """
        
        #assert image_shape[1] == filter_shape[1]
        
        self.input = input
        
        fan_in = np.prod(filter_shape[1:])

        #fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) // np.prod(poolsize))

        bound = np.sqrt(1. /  fan_in) 
        
        if W is None:
#            W_values = np.ones((filter_shape))/10.

            W_values = np.asarray(np.random.uniform(low=-bound,
                                                    high=bound,
                                                    size=filter_shape)
                                  ,dtype='float32')

        self.W = theano.shared(value=W_values, name='W_conv%i'%layer_num, borrow=True)
                                  
        if b is None:
#            b_values = np.zeros((filter_shape[0],))/10.
            b_values = np.asarray(np.full((filter_shape[0],), 0.1),
                                  dtype ='float32')


        self.b = theano.shared(value=b_values, name='b_conv%i'%layer_num, borrow=True)


        conv_out = conv2d(input=input,
                          filters=self.W,
                          filter_shape=filter_shape,
                          input_shape=None,
                          subsample=stride)
                   
 
        self.output = T.nnet.relu(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        
    
        #self.params = OrderedDict()
        #self.params[self.W] = self.W.get_value()
        #self.params[self.b] = self.b.get_value()
        
        self.params = [self.W, self.b]
        





class HiddenLayer(object):
    def __init__(self, input, n_in, n_out, layer_name, W=None, b=None, activation=T.nnet.relu):
        """
            input: a 2D matrix of shape (batch_size, n_in)
            """

        self.input = input
        
        bound = np.sqrt(1. /  n_in)
        
        if W is None:
#            W_values = np.ones((n_in, n_out))/10.
            W_values = np.asarray(np.random.uniform(low=-bound,
                                          high=bound,
                                          size=(n_in, n_out)) ,dtype='float32')

        self.W = theano.shared(value=W_values, name='W_' + layer_name, borrow=True)
        
        if b is None:
            b_values = np.asarray(np.full((n_out,),0.1), 
                                  dtype='float32')



        self.b = theano.shared(value=b_values, name='b_'+ layer_name, borrow=True)
                                  

        
        lin_output = T.dot(input, self.W) + self.b
        
        if activation == None:
            #pdb.set_trace()
            self.output = lin_output

                        
        else:
            self.output = activation(lin_output)
            
        #self.output = activation(lin_output) if activation != None else lin_output
        
        self.params = [self.W, self.b]

