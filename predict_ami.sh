########################################
########### ORACLE PREDICTION ##########
########################################

#cluster2sent + t5base
allennlp predict ./ami_models/t5_models/ser_entrywise_conditioned_t5_base/model.tar.gz \
--include-package t5 --cuda-device 0 --predictor beamsearch --silent \
dataset_ami/entrywise_summarization/test.jsonl \
--output-file ./ami_models/t5_models/ser_entrywise_conditioned_t5_base/test_outputs.jsonl

#cluster2sent + t5small
allennlp predict ./ami_models/t5_models/ser_entrywise_conditioned_t5_small/model.tar.gz \
--include-package t5 --cuda-device 0 --predictor beamsearch --silent \
dataset_ami/entrywise_summarization/test.jsonl \
--output-file ./ami_models/t5_models/ser_entrywise_conditioned_t5_small/test_outputs.jsonl

#ext2sec + t5small
allennlp predict ./ami_models/t5_models/ser_sectionwise_conditioned_t5_small/model.tar.gz \
--include-package t5 --cuda-device 0 --predictor beamsearch --silent \
dataset_ami/sectionwise_allxmin_summarization/test.jsonl \
--output-file ./ami_models/t5_models/ser_sectionwise_conditioned_t5_small/test_outputs.jsonl

#ext2note + t5small
allennlp predict ./ami_models/t5_models/ser_allxmins_fullsummary_t5_small/model.tar.gz \
--include-package t5 --cuda-device 0 --predictor beamsearch --silent \
dataset_ami/allxmin_summarization/test.jsonl \
--output-file ./ami_models/t5_models/ser_allxmins_fullsummary_t5_small/test_outputs.jsonl


#cov2note + pg
allennlp predict ./ami_models/pg_models/ser_fullconversation_fullsummary/model.tar.gz \
--include-package pointergen --cuda-device 0 --predictor beamsearch_constrained --silent \
dataset_ami/full_summarization/test.jsonl \
--output-file ./ami_models/pg_models/ser_fullconversation_fullsummary/test_outputs.jsonl

#ex2note + pg
allennlp predict ./ami_models/pg_models/ser_allxmins_fullsummary/model.tar.gz \
--include-package pointergen --cuda-device 0 --predictor beamsearch_constrained --silent \
dataset_ami/allxmin_summarization/test.jsonl \
--output-file ./ami_models/pg_models/ser_allxmins_fullsummary/test_outputs.jsonl

#ext2sec + pg
allennlp predict ./ami_models/pg_models/ser_sectionwise_allxmins_sectionsummary/model.tar.gz \
--include-package pointergen --cuda-device 0 --predictor beamsearch --silent \
dataset_ami/sectionwise_allxmin_summarization/test.jsonl \
--output-file ./ami_models/pg_models/ser_sectionwise_allxmins_sectionsummary/test_outputs.jsonl

#cluster2sent + pg
allennlp predict ./ami_models/pg_models/ser_entrywise_summarization/model.tar.gz \
--include-package pointergen --cuda-device 0 --predictor beamsearch --silent \
dataset_ami/entrywise_summarization/test.jsonl \
--output-file ./ami_models/pg_models/ser_entrywise_summarization/test_outputs.jsonl


##############################################
##### EXTRACTING NOTEWORTHY UTTERANCES #######
##############################################

#hlstm multilabel
allennlp predict ./ami_models/xmin_prediction/ami_hlstm_multilabel/model.tar.gz \
--include-package sequential_sentence_tagger --cuda-device 0 --predictor simple_multilabel_classifier --silent \
dataset_ami/sectionwise_xmin_multilabel_classification/val.jsonl \
--output-file ./ami_models/xmin_prediction/ami_hlstm_multilabel/val_outputs.jsonl

allennlp predict ./ami_models/xmin_prediction/ami_hlstm_multilabel/model.tar.gz \
--include-package sequential_sentence_tagger --cuda-device 0 --predictor simple_multilabel_classifier --silent \
dataset_ami/sectionwise_xmin_multilabel_classification/test.jsonl \
--output-file ./ami_models/xmin_prediction/ami_hlstm_multilabel/test_outputs.jsonl

#hlstm binary
allennlp predict ./ami_models/xmin_prediction/ami_hlstm_binary/model.tar.gz \
--include-package sequential_sentence_tagger --cuda-device 0 --predictor simple_unilabel_classifier --silent \
dataset_ami/allxmin_binary_classification/val.jsonl \
--output-file ./ami_models/xmin_prediction/ami_hlstm_binary/val_outputs.jsonl

allennlp predict ./ami_models/xmin_prediction/ami_hlstm_binary/model.tar.gz \
--include-package sequential_sentence_tagger --cuda-device 0 --predictor simple_unilabel_classifier --silent \
dataset_ami/allxmin_binary_classification/test.jsonl \
--output-file ./ami_models/xmin_prediction/ami_hlstm_binary/test_outputs.jsonl



###############################
######### CLUSTERING  #########
###############################

python make_predicted_xmin_datasets.py -dataset ami -ser_dir ami_models/xmin_prediction/ami_hlstm_multilabel/ -mode multilabel
python make_predicted_xmin_datasets.py -dataset ami -ser_dir ami_models/xmin_prediction/ami_hlstm_binary/ -mode unilabel


####################################################################################################
################# GENERATION WITH PREDICTED NOTEWORTHY UTTERANCES ##################################
####################################################################################################

#cluster2sent + t5base
allennlp predict ./ami_models/t5_models/ser_entrywise_conditioned_t5_base/model.tar.gz \
--include-package t5 --cuda-device 0 --predictor beamsearch --silent \
./ami_models/xmin_prediction/ami_hlstm_multilabel/predicted_entrywise_gapped.jsonl \
--output-file ./ami_models/t5_models/ser_entrywise_conditioned_t5_base/test_outputs_on_hlstm.jsonl


#cluster2sent + t5small
allennlp predict ./ami_models/t5_models/ser_entrywise_conditioned_t5_small/model.tar.gz \
--include-package t5 --cuda-device 0 --predictor beamsearch --silent \
./ami_models/xmin_prediction/ami_hlstm_multilabel/predicted_entrywise_gapped.jsonl \
--output-file ./ami_models/t5_models/ser_entrywise_conditioned_t5_small/test_outputs_on_hlstm.jsonl

#ext2sec + t5small
allennlp predict ./ami_models/t5_models/ser_sectionwise_conditioned_t5_small/model.tar.gz \
--include-package t5 --cuda-device 0 --predictor beamsearch --silent \
./ami_models/xmin_prediction/ami_hlstm_multilabel/predicted_sectionwise_allxmin.jsonl \
--output-file ./ami_models/t5_models/ser_sectionwise_conditioned_t5_small/test_outputs_on_hlstm.jsonl

#ext2note + t5small
allennlp predict ./ami_models/t5_models/ser_allxmins_fullsummary_t5_small/model.tar.gz \
--include-package t5 --cuda-device 0 --predictor beamsearch --silent \
./ami_models/xmin_prediction/ami_hlstm_binary/predicted_allxmin_test.jsonl \
--output-file ./ami_models/t5_models/ser_allxmins_fullsummary_t5_small/test_outputs_on_hlstm.jsonl



#ex2note + pg
allennlp predict ./ami_models/pg_models/ser_allxmins_fullsummary/model.tar.gz \
--include-package pointergen --cuda-device 0 --predictor beamsearch_constrained --silent \
./ami_models/xmin_prediction/ami_hlstm_binary/predicted_allxmin_test.jsonl \
--output-file ./ami_models/pg_models/ser_allxmins_fullsummary/test_outputs_on_hlstm.jsonl

#ext2sec + pg
allennlp predict ./ami_models/pg_models/ser_sectionwise_allxmins_sectionsummary/model.tar.gz \
--include-package pointergen --cuda-device 0 --predictor beamsearch --silent \
./ami_models/xmin_prediction/ami_hlstm_multilabel/predicted_sectionwise_allxmin.jsonl \
--output-file ./ami_models/pg_models/ser_sectionwise_allxmins_sectionsummary/test_outputs_on_hlstm.jsonl

#cluster2sent + pg
allennlp predict ./ami_models/pg_models/ser_entrywise_summarization/model.tar.gz \
--include-package pointergen --cuda-device 0 --predictor beamsearch --silent \
./ami_models/xmin_prediction/ami_hlstm_multilabel/predicted_entrywise_gapped.jsonl \
--output-file ./ami_models/pg_models/ser_entrywise_summarization/test_outputs_on_hlstm.jsonl


#####################################################################################
############################   ROUGE CALCULATION   ##################################
#####################################################################################


python calculate_rouge.py -dataset ami -split test -algo cluster2sent -save_results -rouge_impl pyrouge -eval_file ./ami_models/t5_models/ser_entrywise_conditioned_t5_base/test_outputs.jsonl
python calculate_rouge.py -dataset ami -split test -algo cluster2sent -save_results -rouge_impl pyrouge -eval_file ./ami_models/t5_models/ser_entrywise_conditioned_t5_small/test_outputs.jsonl
python calculate_rouge.py -dataset ami -split test -algo ext2sec -save_results -rouge_impl pyrouge -eval_file ./ami_models/t5_models/ser_sectionwise_conditioned_t5_small/test_outputs.jsonl
python calculate_rouge.py -dataset ami -split test -algo ext2note -save_results -rouge_impl pyrouge -eval_file ./ami_models/t5_models/ser_allxmins_fullsummary_t5_small/test_outputs.jsonl
python calculate_rouge.py -dataset ami -split test -algo conv2note -save_results -rouge_impl pyrouge -eval_file ./ami_models/pg_models/ser_fullconversation_fullsummary/test_outputs.jsonl
python calculate_rouge.py -dataset ami -split test -algo ext2note -save_results -rouge_impl pyrouge -eval_file ./ami_models/pg_models/ser_allxmins_fullsummary/test_outputs.jsonl
python calculate_rouge.py -dataset ami -split test -algo ext2sec -save_results -rouge_impl pyrouge -eval_file ./ami_models/pg_models/ser_sectionwise_allxmins_sectionsummary/test_outputs.jsonl
python calculate_rouge.py -dataset ami -split test -algo cluster2sent -save_results -rouge_impl pyrouge -eval_file ./ami_models/pg_models/ser_entrywise_summarization/test_outputs.jsonl


python calculate_rouge.py -dataset ami -split test -algo cluster2sent -save_results -rouge_impl pyrouge -eval_file ./ami_models/t5_models/ser_entrywise_conditioned_t5_base/test_outputs_on_hlstm.jsonl
python calculate_rouge.py -dataset ami -split test -algo cluster2sent -save_results -rouge_impl pyrouge -eval_file ./ami_models/t5_models/ser_entrywise_conditioned_t5_small/test_outputs_on_hlstm.jsonl
python calculate_rouge.py -dataset ami -split test -algo ext2sec -save_results -rouge_impl pyrouge -eval_file ./ami_models/t5_models/ser_sectionwise_conditioned_t5_small/test_outputs_on_hlstm.jsonl
python calculate_rouge.py -dataset ami -split test -algo ext2note -save_results -rouge_impl pyrouge -eval_file ./ami_models/t5_models/ser_allxmins_fullsummary_t5_small/test_outputs_on_hlstm.jsonl
python calculate_rouge.py -dataset ami -split test -algo ext2note -save_results -rouge_impl pyrouge -eval_file ./ami_models/pg_models/ser_allxmins_fullsummary/test_outputs_on_hlstm.jsonl
python calculate_rouge.py -dataset ami -split test -algo ext2sec -save_results -rouge_impl pyrouge -eval_file ./ami_models/pg_models/ser_sectionwise_allxmins_sectionsummary/test_outputs_on_hlstm.jsonl
python calculate_rouge.py -dataset ami -split test -algo cluster2sent -save_results -rouge_impl pyrouge -eval_file ./ami_models/pg_models/ser_entrywise_summarization/test_outputs_on_hlstm.jsonl

