# -*- coding: utf-8 -*-
from global_vars import *
import os

class Config():

    def __init__(self, e="default", d=True, c=BI_LSTM_CRF, s=ASPELL_ONLY, i=0):
        if e == "shared":
            raise IOError("Experiment name cannot be 'shared'.")
        experiment_name = e
        self.experiment_name = e
        self.use_dictionary_normalisation = d
        self.classification_strategy = c
        self.spellcheck_strategy = s
        self.experiment_name_full = "%s_%s_%s_%s" % (str(i).zfill(2), d, c, s)

        self.data_path = "data/"
        self.asset_path = "asset/"
        self.output_path = "output/"
        self.output_path_exp = self.output_path + self.experiment_name
        self.output_path_exp_full = self.output_path_exp + "/" + self.experiment_name_full
        # The folder to contain the visualisation.
        self.visualisation_folder = "visualisation/"

        # Sentence too long for QRNN to use char
        self.qrnn_sent_too_long_char = "*"
        # The text file containing a list of all English words.
        if os.path.isdir("data/shared/word_sets"):
            self.words_en = "data/shared/word_sets/words_en.txt"
            print "using shared words.en"
        else:
            self.words_en = self.data_path + experiment_name + "/word_sets/words_en.txt"

        # The text file to learn the word embeddings from.
        if os.path.isdir("data/shared/embedding_data"):
            self.embeddings_data = "data/shared/embedding_data"
            print "using shared embeddings data"
        else:
            self.embeddings_data = self.data_path + experiment_name + "/embedding_data"        

        self.embeddings_data_file = "data/shared/embedding_data/text_docs_all.txt"

        # The training file, in JSON format.
        self.train_data = self.data_path + experiment_name + "/train_data.json"

        # The testing file.
        self.test_data  = self.data_path + experiment_name + "/test_data.json"

        # The truth file, with the correct tokens from the testing file.
        self.truth_data = self.data_path + self.experiment_name + "/test_truth.json"

        # The filename of the embeddings model.
        self.emb_model_folder = self.asset_path + self.experiment_name + "/embeddings_model"
        self.emb_model_filename = self.asset_path + self.experiment_name + "/embeddings_model/model"

        # The supplemental dataset to train the QRNN on.
        # This could be a large corpus of clean data. It will be 'corrupted' so that it may be used
        # to learn spelling errors.
        if os.path.isdir("data/shared/qrnn_supplemental_dataset"):
            self.qrnn_supplemental_dataset = "data/shared/qrnn_supplemental_dataset/qrnn_supplemental_dataset.txt"
            self.qrnn_supplemental_common_errors_dict = "data/shared/qrnn_supplemental_dataset/common_errors.txt"
            print "using shared qrnn supplemental dataset"
        else:
            self.qrnn_supplemental_dataset = self.data_path + experiment_name + "/qrnn_supplemental_dataset/qrnn_supplemental_dataset.txt"
            self.qrnn_supplemental_common_errors_dict = self.data_path + experiment_name + "/qrnn_supplemental_dataset/common_errors.txt"

        self.qrnn_supplemental_corruption_coefficient_1   = 0.333 # The percentage of words that are 'corrupted' in the supplemental dataset for the QRNN.
        self.qrnn_supplemental_corruption_coefficient_2 = 0.333 # The percentage of words that are 'corrupted' more than once in the supplemental dataset.
        self.qrnn_supplemental_corruption_coefficient_3 = 0.15 # The percentage of words that are joined together in the supplemental dataset, such as 'theman' -> 'the_man'


        self.embedding_data_train = self.asset_path + experiment_name + "/embedding_data_train/emb_data_train.txt"
        self.embedding_data_train_all = self.asset_path + experiment_name + "/embedding_data_train/emb_data_train_all.txt"
        self.embedding_data_train_folder = self.asset_path + experiment_name + "/embedding_data_train"

        self.embedding_model_pretrained = None#"data/shared/embedding_model/dmp_model"

        # A file containing stopwords (common words). Useful for the EMB model.
        self.stopwords_file = self.data_path + "/shared/stopwords/common.txt"


        """ ======================================= """

        if self.embedding_model_pretrained:
            self.emb_model_filename = self.embedding_model_pretrained

        # Please do not modify the following values.

        # The token to represent the start of a sentence in the embedding data.
        self.SOS_TOKEN = "_SOS_"

        # The token to represent the end of a sentence in the embedding data.
        self.EOS_TOKEN = "_EOS_"

        # The folder to hold the qrnn training dataset.
        self.qrnn_training_dataset_folder = self.asset_path + experiment_name + "/qrnn_training_dataset"
        self.qrnn_training_dataset_filename_co = self.qrnn_training_dataset_folder + "/train.co"
        self.qrnn_training_dataset_filename_en = self.qrnn_training_dataset_folder + "/train.en"





        # The filename to contain the padded data for embedding training.
        #self.padded_emb_filename = self.asset_path + self.experiment_name + "/embeddings_padded_data/padded_embedding_data.txt"

        # The folder to learn the word embeddings from.
        #self.padded_emb_folder = self.asset_path + self.experiment_name + "/embeddings_padded_data"

        # The name of the folder to contain the class data.
        self.class_data_folder  = self.asset_path + self.experiment_name + "/class_data"

        # The name of the unique replacements filename.
        self.unique_replacements_folder = self.asset_path + self.experiment_name + "/unique_replacements"
        self.unique_replacements_filename = self.unique_replacements_folder + "/unique_replacements.json"

        # The name of the file to contain the most commonly occuring acronym expansions for tokens in the test data.
        self.acronyms_folder = self.asset_path + self.experiment_name + "/acronym_possibilities"
        self.acronyms_filename = self.acronyms_folder + "/acronym_possibilities.json"

        self.predictions_output_filename = self.output_path_exp_full + "/predictions.json"


        self.truth_output_filename = self.output_path_exp_full + "/test_truth.json"

        # The name of the file where the classifier's predictions will be sent.
        #self.classifier_predictions_output_filename = self.asset_path + self.experiment_name + "/classifier_predictions/predictions.txt"

        self.classifier_predictions_folder = self.asset_path + self.experiment_name + "/classifier_predictions"

        self.aspell_pwl_folder = self.asset_path + self.experiment_name + "/pwl"
        self.aspell_pwl_file = self.aspell_pwl_folder + "/pwl.txt"


        self.qrnn_predictions_folder = self.asset_path + self.experiment_name + "/qrnn_data"

        # The file containing the QRNN predictions.
        self.qrnn_predictions_file = self.qrnn_predictions_folder +"/qrnn_predictions.txt"

        # The file containing the input for the QRNN predictions.
        self.qrnn_test_input_file = self.asset_path + self.experiment_name + "/qrnn_data/test_input.txt"

        # The file containing the input for the QRNN predictions, including the overly long sentences.
        self.qrnn_test_input_file_all = self.asset_path + self.experiment_name + "/qrnn_data/test_input_all.txt"

        # The file containing data about how the test set for the QRNN was segmented, so it can be put together later.
        self.qrnn_segmentation_metadata_file = self.asset_path + self.experiment_name + "/qrnn_data/segmentation_metadata.json"

   
        self.lstm_classifier_predictions_folder = "libraries/sequence_tagging/results/" + self.experiment_name 
        self.lstm_classifier_predictions_unmoved_filename = "libraries/sequence_tagging/results/predictions.txt" 
        self.lstm_classifier_predictions_filename = self.classifier_predictions_folder + "/lstm_predictions.txt"
        self.none_classifier_predictions_filename = self.classifier_predictions_folder + "/none_predictions.txt"

        if self.classification_strategy == BI_LSTM_CRF:
            self.classifier_predictions_filename = self.lstm_classifier_predictions_filename
            self.classifier_predictions_filename_no_folder = "lstm_predictions.txt"
        else:
            self.classifier_predictions_filename = self.none_classifier_predictions_filename
            self.classifier_predictions_filename_no_folder = "none_predictions.txt"



        # The name of the Javascript file containing information about how each token was normalised.
        self.normalised_tokens_output_filename = self.output_path_exp_full + "/normalised_tokens.json"
        self.normalised_documents_output_filename = self.output_path_exp_full + "/normalised_documents.json"

        # The name of the Javascript file containing the results of each experiment.
        self.eval_output_filename = self.output_path_exp_full + "/results.json"





    def print_config(self):
        print "-" * 60
        print "Experiment name        : %s" % self.experiment_name
        print "Experiment name (full) : %s" % self.experiment_name_full
        print "Use Dictionary norm    : %s" % self.use_dictionary_normalisation
        print "Classification strategy: %s" % self.classification_strategy
        print "Spellcheck strategy    : %s" % self.spellcheck_strategy
        print "-" * 60

    #fold = 9

    """ ======================================= """

 

