# lexnorm

This is the source code for our paper on Lexical Normalisation.

In order to run the LSTM-CRF in the pipeline, please download the Sequence Tagging code from https://github.com/guillaumegenthial/sequence_tagging and place it under "libraries". The source code for the Quasi-Recurrent neural network is available at https://github.com/Kyubyong/quasi-rnn.

The input data to the system must be in JSON format, and placed under "data/fold_x" where x is a number between 1 and 10. There should be three files:

- test_data.json
- test_truth.json
- train_data.json

An example document from `train_data.json` might look as follows:

```
{"tid": 1, "index": 1, "input": ["person", "triped", "over"], "output": ["person", "tripped", "over"]}
```
`test_data` needs no "input", and `test_truth` needs no "output".

To run the experiments, adjust the 'experiments' list in `experiments.py` and run it using `python run_experiments.py`.

If you experience any issues running the code please let me know.
