#!/usr/bin/env python
"""
Evaluation scripts for English Lexical Normalisation shared task in W-NUT 2015.
"""


import sys
import argparse
try:
    import ujson as json
except ImportError:
    import json

from config import Config


def evaluate(conf=None):

    config = conf if conf else Config()

    pred_file = config.predictions_output_filename
    oracle_file = config.truth_output_filename

    pred_list = json.load(open(pred_file))
    oracle_list = json.load(open(oracle_file))

    correct_norm = 0.0
    total_norm = 0.0
    total_nsw = 0.0
    c = 0
    for pred, oracle in zip(pred_list, oracle_list):
        try:
            assert(pred["index"] == oracle["index"])
            input_tokens = pred["input"]
            pred_tokens = pred["output"]
            oracle_tokens = oracle["output"]
            sent_length = len(input_tokens)


            c += 1

            #print pred_tokens
            #print oracle_tokens

            for i in range(sent_length):              


                if pred_tokens[i].lower() != input_tokens[i].lower() and oracle_tokens[i].lower() == pred_tokens[i].lower():
                    correct_norm += 1
                  
                if oracle_tokens[i].lower() != input_tokens[i].lower():
                    total_nsw += 1
                    #f.write(input_tokens[i].ljust(30), pred_tokens[i].ljust(30), oracle_tokens[i].ljust(30))
                if pred_tokens[i].lower() != input_tokens[i].lower():
                    total_norm += 1



                #print input_tokens[i].lower().ljust(30), pred_tokens[i].lower().ljust(30), oracle_tokens[i].lower().ljust(30), correct_norm, total_nsw, total_norm
        except AssertionError:
            print "Invalid data format"
            sys.exit(1)
    #print total_nsw
    #print correct_norm
    # calc p, r, f
    f1 = 0.0
    try:
        p = correct_norm / total_norm
        r = correct_norm / total_nsw

    except ZeroDivisionError:
        p = 0.0
        r = 0.0
        f1 = 0.0

    print "Evaluating", pred_file
    if p != 0 and r != 0:
        f1 =  (2 * p * r) / (p + r)

        print "%s %s %s"  % (p, r, f1)
        print "precision:", round(p, 4)
        print "recall:   ", round(r, 4)
        print "F1:       ", round(f1, 4)
    else:
        print "precision:", round(p, 4)
        print "recall:   ", round(r, 4)
    with open(config.eval_output_filename, 'w') as f:
        json.dump({ "Precision": p, "Recall": r, "F1": f1}, f)
    return p, r, f1 

def main():
    evaluate()


if __name__ == "__main__":
    main()