QRNN_SPE    = "QRNN"
EMB_ONLY    = "EMB"
EMB_ASPELL  = "EMBAspell"
ASPELL_ONLY = "Aspell"
NONE        = "None"
BI_LSTM_CRF = "BiLSTM"

DICT_NORMALISATION_STRATEGIES = [
	False,
	True
]

SPELLCHECKING_STRATEGIES = [
	#
	NONE,
	QRNN_SPE,
	EMB_ONLY,
	ASPELL_ONLY,
	EMB_ASPELL,
]

CLASSIFICATION_STRATEGIES = [
	NONE,
	BI_LSTM_CRF	
]


M_DICTIONARY_NORMALISATION 	= "Dict normalisation"
M_ACRONYM 					= "Acr expansion"
M_ACRONYM_ERROR				= "Acr (not found)"
M_QRNN 						= "QRNN"
M_QRNN_ALIGNMENT_ERROR 		= "QRNN (Align err)"
M_ASPELL 					= "Aspell"
M_EMB 						= "Emb model"
M_EMB_FAILED 				= "Emb (Not found)"
M_NO_SPELLCHECKING 			= "Not spellchecked"
M_DST 						= "DST"