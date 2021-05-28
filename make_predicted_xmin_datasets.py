

import jsonlines
import os

from tqdm import tqdm
import numpy as np
from collections import defaultdict
from copy import deepcopy
import argparse


parser = argparse.ArgumentParser(description='calculate rouge scores')

parser.add_argument(
    '-dataset',
    dest='dataset',
    help='Dataset name (either medical or ami)',
    type=str,
    required=True,
)

parser.add_argument(
    '-ser_dir',
    dest='ser_dir',
    help='serialization directory containing validation and test outputs',
    type=str,
    required=True,
)


parser.add_argument(
    '-mode',
    dest='mode',
    help='unilabel/multilabel prediction',
    default='multilabel',
    type=str
)


args=parser.parse_args()
dataset = args.dataset
ser_dir = args.ser_dir
dataset_dir = f"dataset_{dataset}"
mode = args.mode


if mode=="unilabel":
    val_fpath = os.path.join(ser_dir, "val_outputs.jsonl")
    val_predictions = list(jsonlines.open(val_fpath))
    test_fpath = os.path.join(ser_dir, "test_outputs.jsonl")
    test_predictions = list(jsonlines.open(test_fpath))

    all_ground_truths=[]
    all_predictions=[]
    for elem in val_predictions:
        all_predictions.append(elem["prediction"][0])
        all_ground_truths.append(elem["ground_truth"])

    all_ground_truths = np.concatenate(all_ground_truths, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)


    base_rates = all_ground_truths.sum(axis=0)/len(all_ground_truths)

    thresholds = []

    for j in range(all_predictions.shape[1]):
        br = base_rates[j]
        sec_pred_probs = all_predictions[:,j]
        cutoff = np.quantile(sec_pred_probs,1-br)
        thresholds.append(cutoff)


    def get_allxmin_dp(base_dp):
        # pdb.set_trace()
        allxmin_utterances = []

        test_texts = base_dp["input"]
        y_pred = np.array(base_dp["prediction"][0])


        threshold = thresholds[0]
        sec_pred_probs = y_pred[:,0]
        is_xmin = sec_pred_probs>=threshold

        for line, pred in zip(test_texts, is_xmin):
            if pred:
                allxmin_utterances.append(line)

        dp_to_return = {"case_id":base_dp['case_id'],
                        "article_lines":allxmin_utterances,
                        "summary_lines":["dummy"]}

        # pdb.set_trace()

        return dp_to_return

    allxmin_dps = []

    for dp in tqdm(test_predictions):
        new_dp = get_allxmin_dp(dp)
        allxmin_dps.append(new_dp)


    existing_caseids = set()

    for dp in allxmin_dps:
        existing_caseids.add(dp["case_id"])

    print(f'{len(existing_caseids)} cases found in the outgoing file')

    output_path = os.path.join(ser_dir, "predicted_allxmin_test.jsonl")
    with jsonlines.open(output_path, "w") as w:
        for dp in allxmin_dps:
            w.write(dp)

    exit(0)




# GETTING LABEL DICT
temp_dataset_path = os.path.join(dataset_dir, "sectionwise_xmin_multilabel_classification", "test.jsonl")
temp_dataset=list(jsonlines.open(temp_dataset_path, "r"))
label_dict = temp_dataset[0]["label_dict"]
label_arr = ["_" for _ in range(len(label_dict.keys()))]
for label, idx in label_dict.items():
    label_arr[idx]=label


val_fpath = os.path.join(ser_dir, "val_outputs.jsonl")
val_predictions = list(jsonlines.open(val_fpath))
test_fpath = os.path.join(ser_dir, "test_outputs.jsonl")
test_predictions = list(jsonlines.open(test_fpath))

all_ground_truths=[]
all_predictions=[]
for elem in val_predictions:
    all_predictions.append(elem["prediction"][0])
    all_ground_truths.append(elem["ground_truth"])

all_ground_truths = np.concatenate(all_ground_truths, axis=0)
all_predictions = np.concatenate(all_predictions, axis=0)


base_rates = all_ground_truths.sum(axis=0)/len(all_ground_truths)

thresholds = []

for j in range(all_predictions.shape[1]):
    br = base_rates[j]
    sec_pred_probs = all_predictions[:,j]
    cutoff = np.quantile(sec_pred_probs,1-br)
    thresholds.append(cutoff)

thresholds = np.array(thresholds)
print("Thresholds=", thresholds)

def get_sectionwise_dps(base_dp):
    sectionwise_xmins = defaultdict(list)

    test_texts = base_dp["input"]
    y_pred = np.array(base_dp["prediction"][0])

    for j, section_name in enumerate(label_arr):
        threshold = thresholds[j]
        sec_pred_probs = y_pred[:,j]
        is_xmin = sec_pred_probs>=threshold

        for line, pred in zip(test_texts, is_xmin):
            if pred:
                sectionwise_xmins[section_name].append(line)

    dps_to_return=[]

    for section, lines in sectionwise_xmins.items():
        dps_to_return.append({"case_id":base_dp['case_id'], "article_lines":lines, "summary_lines":["dummy"], "section":section})

    return dps_to_return


sectionwise_allxmin_dps = []

for dp in tqdm(test_predictions):
    new_dps = get_sectionwise_dps(dp)
    sectionwise_allxmin_dps.extend(new_dps)


existing_caseids = set()

for dp in sectionwise_allxmin_dps:
    existing_caseids.add(dp["case_id"])

print(f'{len(existing_caseids)} cases found in the outgoing file')

output_path = os.path.join(ser_dir, "predicted_sectionwise_allxmin.jsonl")
with jsonlines.open(output_path, "w") as w:
    for dp in sectionwise_allxmin_dps:
        w.write(dp)



###############################
######## MAKING CLUSTERS
###############################


def get_intervals(arr, cohesion=0):
    arr=list(arr)
    arr2=deepcopy(arr)

    # making the
    for i, elem in enumerate(arr):
        if elem!=1:
            continue
        lookahead = arr[i+1:i+1+cohesion+1]
        if 1 in lookahead:
            first_occ = lookahead.index(1)
            for j in range(i+1,i+1+first_occ):
                if j<len(arr2):
                    arr2[j]=1

    finished_intervals=[]
    inside_interval=False
    ci_begin=None
    arr2=list(arr2)  # the next step wont work if arr is not a list
    new_arr=[0]+arr2+[0]  # for starting in the beginning and closing at the end
    for pos in range(1,len(new_arr)):
        prev_val = new_arr[pos-1]
        next_val = new_arr[pos]
        if prev_val==0 and next_val==1:
            assert inside_interval==False
            ci_begin=pos
            inside_interval=True
        elif prev_val==1 and next_val==0:
            assert inside_interval==True
            finished_intervals.append((ci_begin, pos-1))
            inside_interval=False
        else:
            continue

    finished_intervals = [(i-1,j-1) for (i,j) in finished_intervals]     # shift indices by 1 since we prepended 0 before this

    return finished_intervals


def get_entrywise_dps(base_dp, cohesion):
    test_texts = base_dp["input"]
    y_pred = np.array(base_dp["prediction"][0])


    dps_to_return=[]

    for j, section_name in enumerate(label_arr):
        threshold = thresholds[j]
        sec_pred_probs = y_pred[:,j]
        is_xmin = sec_pred_probs>=threshold

        labels=is_xmin.astype(int)
        snippet_intervals = get_intervals(labels, cohesion)

        for _i, interval in enumerate(snippet_intervals):
            # CHOICE1: add the sentences in between in the input cluster
#             relevant_input_lines = test_texts[interval[0]: interval[1]+1]

            # CHOICE2: do not add the sentences in between in the input cluster
            relevant_input_lines = []
            for idx in range(interval[0], interval[1]+1):
                if labels[idx]==1:
                    relevant_input_lines.append(test_texts[idx])

            dps_to_return.append({
              'article_lines': relevant_input_lines,
              'summary_lines': ['dummy'],
              'case_id': base_dp['case_id'],
              'index_in_note': _i,
              'section': section_name
            })


    return dps_to_return



# FIGURING OUT THE OPTIMAL VALUE OF COHESION PARAMETER FROM VALIDATION DATA
temp_dataset_path = os.path.join(dataset_dir, "entrywise_summarization", "val.jsonl")
temp_dataset=list(jsonlines.open(temp_dataset_path, "r"))
gt_num_clusters=len(temp_dataset)

chosen_cohesion_param=[]

for cohesion_param in range(0,100):
    # print(f"trying out cohesion parameter = {cohesion_param}")
    entrywise_xmin_dps = []
    for dp in val_predictions:
        new_dps = get_entrywise_dps(dp, cohesion=cohesion_param)
        entrywise_xmin_dps.extend(new_dps)
    print(f"ground truth has {gt_num_clusters} clusters, cohesion={cohesion_param} created {len(entrywise_xmin_dps)}")
    if len(entrywise_xmin_dps)<gt_num_clusters:
        chosen_cohesion_param.append(cohesion_param)
        break

chosen_cohesion_param=chosen_cohesion_param[0]

print(f"chosen_cohesion_param if {chosen_cohesion_param}")

######################

entrywise_xmin_dps = []
for dp in tqdm(test_predictions):
    new_dps = get_entrywise_dps(dp, cohesion=chosen_cohesion_param)
    entrywise_xmin_dps.extend(new_dps)

existing_caseids = set()

for dp in entrywise_xmin_dps:
    existing_caseids.add(dp["case_id"])

print(f'{len(existing_caseids)} cases found in the outgoing file with {len(entrywise_xmin_dps)} clusters in total')

output_path = os.path.join(ser_dir, "predicted_entrywise_gapped.jsonl")
with jsonlines.open(output_path, "w") as w:
    for dp in entrywise_xmin_dps:
        w.write(dp)


