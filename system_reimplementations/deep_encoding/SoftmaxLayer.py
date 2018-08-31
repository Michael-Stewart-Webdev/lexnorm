'''
Created on 1Mar.,2017

@author: ruiwang
'''
import theano
import numpy
from theano import pp, tensor as T
import numpy as np


FLOAT32  = "float32"

class SoftMaxRegression(object):


    def __init__(self, input, n_in, n_out):

        self.weight_for_vector = theano.shared(value=numpy.zeros((n_in, n_out),
                                dtype=FLOAT32),
                                name='weight_for_vector', borrow=True)

        #self.b = theano.shared(value=numpy.zeros((n_out,),
        #                       dtype=FLOAT32),
        #                       name='b', borrow=True)


        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.weight_for_vector) )
        

        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.output = self.p_y_given_x
        #self.params = [self.weight_for_vector, self.b]
        self.params = self.weight_for_vector

    def negative_log_likelihood(self, y):

        return -T.mean(T.log(self.p_y_given_x)* y)
    
    def cross_entropy(self, y):

        return -T.sum(y * T.log(self.p_y_given_x))

    def errors(self, y):

        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', target.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            kl_divergence = T.sum(y*T.log(y /self.p_y_given_x ))
            return kl_divergence #self.cross_entropy(y)