'''
Created on 1Mar.,2017

@author: ruiwang, heavily modified by michael stewart
'''

import theano
import numpy
import os

from theano import pp, tensor as T
from collections import OrderedDict
import numpy as np
import time
from theano.ifelse import ifelse
import HiddenLayer as hidden_layer
import SoftmaxLayer as softmax_layer
import sys, random, pickle

from colorama import Fore, Back, Style, init 
init()




FLOAT32  = "float32"

RANDOM_DATA  = 0
GOOGLE_WORDS = 1

MODE = GOOGLE_WORDS

# this one does use mean cost and conditional update
class deep_encoding_model(object):
        
    def __init__(self, letter_dim, window_size, hidden_size, vocab_size, is_train):

        
        #self.names  = ['embeddings', 'W', 'W_u', 'vec_output', 'b_w', 'b_c', 'b_z']
        self.letter_emb = theano.shared(name='emb', value=0.2 * numpy.random.RandomState(7846).uniform(-1.0, 1.0,\
                       (vocab_size, letter_dim)).astype(FLOAT32), borrow=True) 


        y1 = T.fvector()
        y2 = T.fvector()
        y3 = T.fvector()
        y4 = T.fvector()
        y5 = T.fvector()
        y6 = T.fvector()
        y7 = T.fvector()
        y8 = T.fvector()
        y9 = T.fvector()
        y10 = T.fvector()
        y11 = T.fvector()
        y12 = T.fvector()
        y13 = T.fvector()
        y14 = T.fvector()
        y15 = T.fvector()
        y16 = T.fvector()
        y17 = T.fvector()
        y18 = T.fvector()
        y19 = T.fvector()
        y20 = T.fvector()



        #idxs = theano.shared(name="idxs", value=numpy.zeros(window_size, dtype=np.int))
        idxs = T.ivector()
        #idxs = theano.printing.Print("idxs")(idxs)
        x = self.letter_emb[idxs]


        all_x = self.letter_emb;
        T.set_subtensor(all_x[idxs], self.letter_emb[idxs]); # Need to combine x with the rest of the embeddings

        #print "eval:"
        #print self.letter_emb.eval()
        #print "done"

        #x = theano.printing.Print("input x")(x)

        # Two layers of size 2000, as in the paper

        self.hidden_layer_1 = hidden_layer.HiddenLayer(x.flatten(), letter_dim * window_size, hidden_size, "tanh", is_train)
        hidden_out_1 = self.hidden_layer_1.output

        self.hidden_layer = hidden_layer.HiddenLayer(hidden_out_1, hidden_size, hidden_size, "tanh", is_train)
        hidden_out = self.hidden_layer.output

        
        
        self.softmax_1 = softmax_layer.SoftMaxRegression(hidden_out, n_in = hidden_size, n_out = vocab_size)
        self.softmax_2 = softmax_layer.SoftMaxRegression(hidden_out, n_in = hidden_size, n_out = vocab_size)
        self.softmax_3 = softmax_layer.SoftMaxRegression(hidden_out, n_in = hidden_size, n_out = vocab_size)
        self.softmax_4 = softmax_layer.SoftMaxRegression(hidden_out, n_in = hidden_size, n_out = vocab_size)
        self.softmax_5 = softmax_layer.SoftMaxRegression(hidden_out, n_in = hidden_size, n_out = vocab_size)
        self.softmax_6 = softmax_layer.SoftMaxRegression(hidden_out, n_in = hidden_size, n_out = vocab_size)
        self.softmax_7 = softmax_layer.SoftMaxRegression(hidden_out, n_in = hidden_size, n_out = vocab_size)
        self.softmax_8 = softmax_layer.SoftMaxRegression(hidden_out, n_in = hidden_size, n_out = vocab_size)
        self.softmax_9 = softmax_layer.SoftMaxRegression(hidden_out, n_in = hidden_size, n_out = vocab_size)
        self.softmax_10 = softmax_layer.SoftMaxRegression(hidden_out, n_in = hidden_size, n_out = vocab_size)
        self.softmax_11 = softmax_layer.SoftMaxRegression(hidden_out, n_in = hidden_size, n_out = vocab_size)
        self.softmax_12 = softmax_layer.SoftMaxRegression(hidden_out, n_in = hidden_size, n_out = vocab_size)
        self.softmax_13 = softmax_layer.SoftMaxRegression(hidden_out, n_in = hidden_size, n_out = vocab_size)
        self.softmax_14 = softmax_layer.SoftMaxRegression(hidden_out, n_in = hidden_size, n_out = vocab_size)
        self.softmax_15 = softmax_layer.SoftMaxRegression(hidden_out, n_in = hidden_size, n_out = vocab_size)
        self.softmax_16 = softmax_layer.SoftMaxRegression(hidden_out, n_in = hidden_size, n_out = vocab_size)
        self.softmax_17 = softmax_layer.SoftMaxRegression(hidden_out, n_in = hidden_size, n_out = vocab_size)
        self.softmax_18 = softmax_layer.SoftMaxRegression(hidden_out, n_in = hidden_size, n_out = vocab_size)
        self.softmax_19 = softmax_layer.SoftMaxRegression(hidden_out, n_in = hidden_size, n_out = vocab_size)
        self.softmax_20 = softmax_layer.SoftMaxRegression(hidden_out, n_in = hidden_size, n_out = vocab_size)        

        #self.params = self.hidden_layer.params + self.softmax_1.params + self.softmax_2.params +\
                      #self.softmax_3.params + self.softmax_4.params + self.softmax_5.params 

        
        
        self.output1 = self.softmax_1.output
        self.output2 = self.softmax_2.output
        self.output3 = self.softmax_3.output
        self.output4 = self.softmax_4.output
        self.output5 = self.softmax_5.output
        self.output6 = self.softmax_6.output
        self.output7 = self.softmax_7.output
        self.output8 = self.softmax_8.output
        self.output9 = self.softmax_9.output
        self.output10 = self.softmax_10.output
        self.output11 = self.softmax_11.output
        self.output12 = self.softmax_12.output
        self.output13 = self.softmax_13.output
        self.output14 = self.softmax_14.output
        self.output15 = self.softmax_15.output
        self.output16 = self.softmax_16.output
        self.output17 = self.softmax_17.output
        self.output18 = self.softmax_18.output
        self.output19 = self.softmax_19.output
        self.output20 = self.softmax_20.output

        # optionally we can try a shared softmax layer, then in such case 
        # one only needs one softmax layer, and the following code would be the sum of the cost from each y without
        # other softmax layers
        self.cost = (self.softmax_1.negative_log_likelihood(y1) + \
                     self.softmax_2.negative_log_likelihood(y2) + \
                     self.softmax_3.negative_log_likelihood(y3) + \
                     self.softmax_4.negative_log_likelihood(y4) + \
                     self.softmax_5.negative_log_likelihood(y5) + \
                     self.softmax_6.negative_log_likelihood(y6) + \
                     self.softmax_7.negative_log_likelihood(y7) + \
                     self.softmax_8.negative_log_likelihood(y8) + \
                     self.softmax_9.negative_log_likelihood(y9) + \
                     self.softmax_10.negative_log_likelihood(y10) + \
                     self.softmax_11.negative_log_likelihood(y11) + \
                     self.softmax_12.negative_log_likelihood(y12) + \
                     self.softmax_13.negative_log_likelihood(y13) + \
                     self.softmax_14.negative_log_likelihood(y14) + \
                     self.softmax_15.negative_log_likelihood(y15) + \
                     self.softmax_16.negative_log_likelihood(y16) + \
                     self.softmax_17.negative_log_likelihood(y17) + \
                     self.softmax_18.negative_log_likelihood(y18) + \
                     self.softmax_19.negative_log_likelihood(y19) + \
                     self.softmax_20.negative_log_likelihood(y20) 
                     
                     )

        
        self.y_pred1 = self.softmax_1.y_pred
        self.y_pred2 = self.softmax_2.y_pred
        self.y_pred3 = self.softmax_3.y_pred
        self.y_pred4 = self.softmax_4.y_pred
        self.y_pred5 = self.softmax_5.y_pred
        self.y_pred6 = self.softmax_6.y_pred
        self.y_pred7 = self.softmax_7.y_pred
        self.y_pred8 = self.softmax_8.y_pred
        self.y_pred9 = self.softmax_9.y_pred
        self.y_pred10 = self.softmax_10.y_pred
        self.y_pred11 = self.softmax_11.y_pred
        self.y_pred12 = self.softmax_12.y_pred
        self.y_pred13 = self.softmax_13.y_pred
        self.y_pred14 = self.softmax_14.y_pred
        self.y_pred15 = self.softmax_15.y_pred
        self.y_pred16 = self.softmax_16.y_pred
        self.y_pred17 = self.softmax_17.y_pred
        self.y_pred18 = self.softmax_18.y_pred
        self.y_pred19 = self.softmax_19.y_pred
        self.y_pred20 = self.softmax_20.y_pred        

        lr = T.scalar('lr')
        

        d_all_x =  T.grad(self.cost, wrt = all_x)
        d_hidden = T.grad(self.cost, wrt = self.hidden_layer.params)
        d_soft_1 = T.grad(self.cost, wrt = self.softmax_1.params)
        d_soft_2 = T.grad(self.cost, wrt = self.softmax_2.params)
        d_soft_3 = T.grad(self.cost, wrt = self.softmax_3.params)
        d_soft_4 = T.grad(self.cost, wrt = self.softmax_4.params)
        d_soft_5 = T.grad(self.cost, wrt = self.softmax_5.params)
        d_soft_6 = T.grad(self.cost, wrt = self.softmax_6.params)
        d_soft_7 = T.grad(self.cost, wrt = self.softmax_7.params)
        d_soft_8 = T.grad(self.cost, wrt = self.softmax_8.params)
        d_soft_9 = T.grad(self.cost, wrt = self.softmax_9.params)
        d_soft_10 = T.grad(self.cost, wrt = self.softmax_10.params)
        d_soft_11 = T.grad(self.cost, wrt = self.softmax_11.params)
        d_soft_12 = T.grad(self.cost, wrt = self.softmax_12.params)
        d_soft_13 = T.grad(self.cost, wrt = self.softmax_13.params)
        d_soft_14 = T.grad(self.cost, wrt = self.softmax_14.params)
        d_soft_15 = T.grad(self.cost, wrt = self.softmax_15.params)
        d_soft_16 = T.grad(self.cost, wrt = self.softmax_16.params)
        d_soft_17 = T.grad(self.cost, wrt = self.softmax_17.params)
        d_soft_18 = T.grad(self.cost, wrt = self.softmax_18.params)
        d_soft_19 = T.grad(self.cost, wrt = self.softmax_19.params)
        d_soft_20 = T.grad(self.cost, wrt = self.softmax_20.params)

        #letter_emb_new = new x at the right positions
        #letter_emb[idxs] = x 
        
        updates_param = [
                   (self.letter_emb, all_x -  lr * d_all_x ),
                   (self.hidden_layer.params, self.hidden_layer.params -  lr * d_hidden ),
                   (self.softmax_1.params, self.softmax_1.params -  lr * d_soft_1 ),
                   (self.softmax_2.params, self.softmax_2.params -  lr * d_soft_2 ),
                   (self.softmax_3.params, self.softmax_3.params -  lr * d_soft_3 ),
                   (self.softmax_4.params, self.softmax_4.params -  lr * d_soft_4 ),
                   (self.softmax_5.params, self.softmax_5.params -  lr * d_soft_5 ),
                   (self.softmax_6.params, self.softmax_6.params -  lr * d_soft_6 ),
                   (self.softmax_7.params, self.softmax_7.params -  lr * d_soft_7 ),
                   (self.softmax_8.params, self.softmax_8.params -  lr * d_soft_8 ),
                   (self.softmax_9.params, self.softmax_9.params -  lr * d_soft_9 ),
                   (self.softmax_10.params, self.softmax_10.params -  lr * d_soft_10 ),
                   (self.softmax_11.params, self.softmax_11.params -  lr * d_soft_11 ),
                   (self.softmax_12.params, self.softmax_12.params -  lr * d_soft_12 ),
                   (self.softmax_13.params, self.softmax_13.params -  lr * d_soft_13 ),
                   (self.softmax_14.params, self.softmax_14.params -  lr * d_soft_14 ),
                   (self.softmax_15.params, self.softmax_15.params -  lr * d_soft_15 ),
                   (self.softmax_16.params, self.softmax_16.params -  lr * d_soft_16 ),
                   (self.softmax_17.params, self.softmax_17.params -  lr * d_soft_17 ),
                   (self.softmax_18.params, self.softmax_18.params -  lr * d_soft_18 ),
                   (self.softmax_19.params, self.softmax_19.params -  lr * d_soft_19 ),
                   (self.softmax_20.params, self.softmax_20.params -  lr * d_soft_20 ),
                   ]
        
        self.train = theano.function( inputs  = [idxs, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16, y17, y18, y19, y20, lr],
                                      outputs = [self.cost, self.output1, self.output2, 
                                                 self.output3, self.output4, self.output5, self.output6,
                                                 self.output7, self.output8, self.output9, self.output10,
                                                 self.output11, self.output12, self.output13, self.output14,
                                                 self.output15, self.output16, self.output17, self.output18,
                                                 self.output19, self.output20],
                                      updates = updates_param,
                                      givens = { is_train: np.cast['int32'](1) },
                                      on_unused_input='ignore',
                                      allow_input_downcast=True)
        
        self.evaluate = theano.function( inputs  = [idxs],
                                      outputs = [self.output1, self.output2, 
                                                 self.output3, self.output4, self.output5, self.output6,
                                                 self.output7, self.output8, self.output9, self.output10,
                                                 self.output11, self.output12, self.output13, self.output14,
                                                 self.output15, self.output16, self.output17, self.output18,
                                                 self.output19, self.output20],
                                      on_unused_input='ignore',
                                      givens = { is_train: np.cast['int32'](0) },
                                      allow_input_downcast=True)


    def save(self, folder):   
        for param in self.params:
            numpy.save(os.path.join(folder, param.name + '.npy'), param.get_value())
            
            
    def load(self, folder):  
        for param in self.params:
            param.set_value(numpy.load(os.path.join(folder, param.name+ '.npy')))
                  
                  
            
    def get_emb(self, idx):
        return self.emb[idx]


                
            
def run_deep_encoder(training_data, testing_data, dsts, acronyms, word_set, fold):

    ALPHABET = "abcdefghijklmnopqrstuvwxyz _"
    WINDOW_SIZE = 20

    print "Encoding the data..."

    token_indexes = []
    deep_encoding_predictions = {} # Maps integers to predictions (where integer is the tok_id)

    # Removes any non-ALPHABETical characters from words.
    def clean_word(word):
        return ''.join([i if i in ALPHABET else '' for i in word])

    # Encode the data (for the NN)
    def encode_y(word):
        encoding = [[0 for a in range(len(ALPHABET))] for b in range(WINDOW_SIZE)]
        c = 0
        for char in word:
            index = ALPHABET.index(char)
            encoding[c][index] = 1
            c += 1
        return encoding
    def encode_x(word):
        return [ALPHABET.index(c) for c in word]

    def encode_dataset(dataset, mode):
        xs = []
        ys = []
        tok_id = 0
        for document in dataset:
            for i in range(len(document["input"])):
                inp = clean_word(document["input"][i].lower())
                outp = clean_word(document["output"][i].lower())
                if inp.isalpha() and len(inp) <= WINDOW_SIZE and len(outp) <= WINDOW_SIZE:             
                    
                    
                    encode_this_word = False
                    # Figure out the error type of the word if there is one
                    if inp != outp:
                        encode_this_word = True                 # Spelling error if input doesn't match output      
                    else:
                        if inp in dsts:
                            encode_this_word = True            # Domain-specific if output is not in lexicon, and is the same in the
                                                                    # original and annotated data
                                                                    # "bogger" -> "bogger", for example.
                    if inp in acronyms:
                        encode_this_word = True                # Acronym if input is part of pre-defined list of acronyms
                    
                    if(mode == "training"):
                        encode_this_word = True # It should encode every word in the training data, but only the error words in the test data.
                    if encode_this_word:
                        # Encode the word
                        #print inp.ljust(WINDOW_SIZE, "_")
                        x = encode_x(inp.ljust(WINDOW_SIZE, "_"))
                        y = encode_y(outp.ljust(WINDOW_SIZE, "_"))
                        xs.append(x)
                        ys.append(y)
                        if mode == "testing":
                            token_indexes.append(tok_id)

                tok_id += 1
                        #print x, "\n", y
        return xs, ys

    xtraining, ytraining = encode_dataset(training_data, "training")
    xtesting,  ytesting  = encode_dataset(testing_data, "testing")

    # Create the validation sets by using 5% of the training data
    xvalid = []
    yvalid = []
    for x in range(len(xtraining) / 20):
      xvalid.append(xtraining.pop())
      yvalid.append(ytraining.pop())


    print len(xtraining), "training pairs"
    print len(xtesting), "testing pairs"
    print len(xvalid), "validation pairs"

    def deencode_y(word):
      chars = []
      # De-encode the word
      for c in word:
          chars.append(ALPHABET[(c.index(1))])
      charstring = "".join(c for c in chars)
      return charstring

    def deencode_x(word):
        chars = []
        for c in word:
            chars.append(ALPHABET[c])
        return "".join(c for c in chars)





    def run_validation():
      total = len(xvalid)
      correct = 0
      for i in range(total):
        ev = model.evaluate(xvalid[i])
        prediction = ""
        for e in ev:
            prediction += ALPHABET[e.argmax()]

        correct_prediction = deencode_y(yvalid[i])
        if prediction == correct_prediction:
          correct += 1
      return 1.0 * correct / total, correct, total


    # Initialise the model and train it
    is_train = T.iscalar('is_train')

    model = deep_encoding_model(letter_dim = 25, window_size = WINDOW_SIZE, hidden_size = 2000, vocab_size = len(ALPHABET), is_train = is_train)
    accuracy = 0.0
    prev_accuracy = 0.0
    prev_accuracy_10 = 0.0
    no_improvement_times = 0
    patience = 20   # Will break for loop if no improvement for this many times

    print "\nTraining...\n"
    start_time = time.time()
    lxover50 = len(xtraining) / 50

    try:
        for i in xrange(1, 501):
            #print "---- training ----"
            print "" + Fore.GREEN + "Epoch " + str(i) + Fore.WHITE + ": ",
            sys.stdout.flush()
            for j in range(len(xtraining)):
              
              model.train(xtraining[j], ytraining[j][0], ytraining[j][1], ytraining[j][2], ytraining[j][3], ytraining[j][4], ytraining[j][5],
                        ytraining[j][6], ytraining[j][7], ytraining[j][8], ytraining[j][9],
                        ytraining[j][10], ytraining[j][11], ytraining[j][12], ytraining[j][13], ytraining[j][14],
                        ytraining[j][15], ytraining[j][16], ytraining[j][17], ytraining[j][18], ytraining[j][19], 0.01)
              if (j+1) % lxover50 == 0:
                sys.stdout.write(".")
                sys.stdout.flush()
            #print ""
            #if i % 1 == 0:
            #print "Epoch", str(i + 1) + " complete."



            # Evaluate the model against the validation set
            #print "Running validation...",
            accuracy, correct, total = run_validation()
            #print " done."
            print " " + Style.DIM + str(total - correct) + " / " + str(total) + Style.RESET_ALL,
            sys.stdout.flush()
            print ""

            #if i == 1:
            #    prev_accuracy_10 = accuracy
            #if i % 10 == 0:
            #    if accuracy <= prev_accuracy_10:
            #        print "No improvement found since epoch", i - 10, "- training complete."
            #        break;
            #    prev_accuracy_10 = accuracy

            if accuracy <= prev_accuracy:
              no_improvement_times += 1
            else:
              no_improvement_times = 0

            prev_accuracy = accuracy
            if no_improvement_times == patience:
              print "No improvement found for last", patience, "epochs - training complete."
              break;

            if i > 0 and i % 5 == 0:
              time_taken = time.time() - start_time
              m, s = divmod(time_taken, 60)
              h, m = divmod(m, 60)
              time_taken_str = "%d:%02d:%02d" % (h, m, s)

              time_per_epoch = (time_taken/i)
              time_eta = time.time() + (time_per_epoch * (100 - i))
              time_eta_str = time.strftime("%I:%M %p", time.localtime(time_eta))   
              print "----------------------------------------------------------------------------"
              print "" + Fore.YELLOW + "Error:            " + Fore.WHITE + "%.3f%%".ljust(7) % (100 - 100*accuracy) + " (%d / %d)" % (total-correct, total)
              print      Fore.YELLOW + "Elapsed time:     " + Fore.WHITE + time_taken_str.ljust(7) + " (Fold %d)" % fold
              print      Fore.YELLOW + "ETA:              " + Fore.WHITE + time_eta_str.ljust(7)   + " (100 epochs)"
              print "----------------------------------------------------------------------------"
            #print model.evaluate(x, y1, y2, y3, y4, y5,)
    except KeyboardInterrupt:
        print "Terminating training..."

    # For each word ...



    print "\n===================================================================="        
    print " ACTUAL".ljust(WINDOW_SIZE) + "| " + "Corrpt".ljust(WINDOW_SIZE) + "| " + "Pred.".ljust(WINDOW_SIZE) + "| Probabilities"
    print "===================================================================="        


    n_correct = 0
    n_fail    = 0
    tok_id = 0

    for i in range(len(xtesting)):
      ev = model.evaluate(xtesting[i])





      charstring  = deencode_y(ytesting[i])
      charstring2 = deencode_x(xtesting[i])

      sys.stdout.write(charstring + " | ")
      sys.stdout.write(charstring2 + " | ")

      # Print evaluation info (most likely characters and probabilities of most likely character)
      ch_ind = 0
      prediction = []
      for e in ev:
          ch_ind += 1
          m = ALPHABET[e.argmax()]
          sys.stdout.write(m)
          prediction.append(m)
      sys.stdout.write(" | ")

      prediction_str = "".join(prediction)

      for e in ev:
          for ch in e:
              m = ch.argmax()
              print "%.3f" % ch[m], 


      if prediction_str == charstring:
          print " | " + Fore.GREEN + u'\u2713' + Fore.WHITE,
          n_correct += 1
      else:
          print " | " + Fore.RED + u'\u274C' + Fore.WHITE,
          n_fail += 1
      print ""      
      deep_encoding_predictions[token_indexes[tok_id]] = prediction_str
      tok_id += 1
    print "===================================================================="
    print "\nCorrect:   " + Fore.GREEN + str(n_correct) + Fore.WHITE
    print "Incorrect: " + Fore.RED   + str(n_fail)    + Fore.WHITE
    print "Accuracy:  " + "%.2f" % (100.0 * n_correct / (n_correct + n_fail)) + "%"
    print "\n===================================================================="

    return deep_encoding_predictions