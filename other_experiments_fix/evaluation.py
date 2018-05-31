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


def evaluate(pred_file, oracle_file):
    pred_list = json.load(open(pred_file))
    oracle_list = json.load(open(oracle_file))

    correct_norm = 0.0
    total_norm = 0.0
    total_nsw = 0.0
    for pred, oracle in zip(pred_list, oracle_list):
        try:
            assert(pred["index"] == oracle["index"])
            input_tokens = pred["input"]
            pred_tokens = pred["output"]
            oracle_tokens = oracle["output"]
            sent_length = len(input_tokens)


           # print input_tokens, "\n", pred_tokens, "\n", oracle_tokens

            #print "---"

            


            for i in range(sent_length):


               
                


                if pred_tokens[i].lower() != input_tokens[i].lower() and oracle_tokens[i].lower() == pred_tokens[i].lower() and oracle_tokens[i].strip():
                    correct_norm += 1
                    # for x in range(10000000):
                    #     x = 0
                    # print pred["index"], pred_tokens[i].lower()

                if oracle_tokens[i].lower() != input_tokens[i].lower() and oracle_tokens[i].strip():
                    total_nsw += 1
                if pred_tokens[i].lower() != input_tokens[i].lower() and pred_tokens[i].strip():
                    total_norm += 1

            #print correct_norm, total_nsw, total_norm


                #print input_tokens[i].lower().ljust(30), pred_tokens[i].lower().ljust(30), oracle_tokens[i].lower().ljust(30), correct_norm, total_nsw, total_norm
        except AssertionError:
            print "Invalid data format"
            sys.exit(1)
    #print total_nsw
    #print correct_norm
    # calc p, r, f
    p = correct_norm / total_norm
    r = correct_norm / total_nsw

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
    return p, r, f1 


def main():
    scores = []    
    for i in range(10):
        p, r, f1 = evaluate("fixed/wookhee_lstm_dictionary/fold %d/predictions.json" % (i + 1), "../data/fold_%d/test_truth.json" % (i + 1))
        scores.append((p, r, f1))
        print ""

    print "\nAverage scores:"   
    print "=================================="
    print "precision:", round(sum(s[0] for s in scores) / len(scores), 4)
    print "recall:   ", round(sum(s[1] for s in scores) / len(scores), 4)
    print "F1:       ", round(sum(s[2] for s in scores) / len(scores), 4)      


if __name__ == "__main__":
    main()
