'''
Created on 1Mar.,2017

@author: ruiwang
'''
import theano
from theano import pp, tensor as T
import numpy as np

FLOAT32  = "float32"

class HiddenLayer(object):
    """
    Class for HiddenLayer
    """
    def __init__(self, input, n_in, n_out, activation, is_train):
        
        rng = np.random.RandomState(8568)
        
        self.input = input
        self.input = theano.printing.Print("input flatten x")(input)
        if (activation =="tanh"):
            W_values = np.asarray(rng.uniform(low=-np.sqrt(6. / (n_in + n_out)), 
                                             high=np.sqrt(6. / (n_in + n_out)),
                                             size=(n_in, n_out)), dtype=FLOAT32)
        else:
            W_values = np.asarray(0.01 * rng.standard_normal(size=(n_in, n_out)), 
                                            dtype=FLOAT32)
        
        self.W = theano.shared(value=W_values, name='hidden_W')        
        
        #b_values = np.zeros((n_out,), dtype=FLOAT32)
        #self.b = theano.shared(value=b_values, name='hidden_b')

        lin_output = T.dot(input, self.W) 
        if (activation == "tanh"):
            self.output = T.tanh(lin_output)
        elif (activation == "sigmoid"):
            self.output = T.nnet.sigmoid(lin_output)
        else: self.output = lin_output

        def dropout(inp):
	        # Dropout (as in https://github.com/mdenil/dropout)
	        p = 0.5 # 50% dropout
	        srng = theano.tensor.shared_randomstreams.RandomStreams(
	                rng.randint(999999))
	        # p=1-p because 1's indicate keep and p is prob of dropping
	        mask = srng.binomial(n=1, p=1-p, size=inp.shape)
	        # The cast is important because
	        # int * float32 = float64 which pulls things off the gpu
	        output = inp * T.cast(mask, theano.config.floatX)
	        return output


        train_output = dropout(self.output)

        # Only use dropout in the training phase
        self.output = T.switch(T.neq(is_train, 0), train_output, self.output)


        #self.params = [self.W, self.b]
        self.params = self.W