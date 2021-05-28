
import numpy as np

from collections import OrderedDict, Counter
from tqdm import tqdm

from copy import deepcopy
import jsonlines
import os

train_dataset = list(jsonlines.open("../../dataset_ami/root/train.jsonl", "r"))
val_dataset = list(jsonlines.open("../../dataset_ami/root/val.jsonl", "r"))
test_dataset = list(jsonlines.open("../../dataset_ami/root/test.jsonl", "r"))

output_root_path = "../../dataset_ami/"

seq_of_sections = ["abstract",
                    "actions",
                    "decisions",
                    "problems"]

def makeTimeOrderedDict(unordered_dict):
    return OrderedDict({k:unordered_dict[k] for k in sorted(unordered_dict.keys(), key=lambda i:float(i))})


task_name = "allxmin_binary_classification"

def get_allxmin_binary_classification_dp(base_dp):
    base_dp=deepcopy(base_dp)

    all_xmins=set()
    for section_name in seq_of_sections:
        sorted_entries = sorted(base_dp[section_name],
                            key=lambda i:( float(i["xmin_interval"][0]) , float(i["xmin_interval"][1]) )  )
        for entry in sorted_entries:
            all_xmins.update(entry["sorted_xmins"])

    sorted_turns = makeTimeOrderedDict(base_dp["transcript"])
    input_lines = []
    for tstamp, turn in sorted_turns.items():
        input_lines.append(turn["speaker"]+" "+turn["txt"])

    if len(input_lines)==0:
        return None

    label_dict={"is_important":0}
    labels = np.zeros((len(input_lines),1), dtype=int).tolist()

    for idx, tstamp in enumerate(sorted_turns.keys()):
        if tstamp in all_xmins:
            labels[idx][0]=1

    return {"case_id":base_dp["id"] ,"article_lines":input_lines, "labels":labels, "label_dict":label_dict}

train_section_dataset = [get_allxmin_binary_classification_dp(x) for x in tqdm(train_dataset)]
val_section_dataset = [get_allxmin_binary_classification_dp(x) for x in val_dataset]
test_section_dataset = [get_allxmin_binary_classification_dp(x) for x in test_dataset]

task_output_path = os.path.join(output_root_path, task_name)
os.mkdir(task_output_path)

with jsonlines.open(os.path.join(task_output_path, "train.jsonl"), "w") as w:
    for obj in train_section_dataset:
        if obj!=None:
            w.write(obj)
with jsonlines.open(os.path.join(task_output_path, "val.jsonl"), "w") as w:
    for obj in val_section_dataset:
        if obj!=None:
            w.write(obj)
with jsonlines.open(os.path.join(task_output_path, "test.jsonl"), "w") as w:
    for obj in test_section_dataset:
        if obj!=None:
            w.write(obj)


task_name = "sectionwise_xmin_multilabel_classification"

def get_sectionwise_xmin_multilabel_classification_dp(base_dp):
    base_dp=deepcopy(base_dp)

    sorted_turns = makeTimeOrderedDict(base_dp["transcript"])
    input_lines = []
    for tstamp, turn in sorted_turns.items():
        input_lines.append(turn["speaker"]+" "+turn["txt"])

    if len(input_lines)==0:
        return None

    label_dict={sec:_i for (_i, sec) in enumerate(seq_of_sections)}
    labels = np.zeros((len(input_lines),len(seq_of_sections)), dtype=int).tolist()

    for section_name in seq_of_sections:
        all_xmins=set()
        sorted_entries = sorted(base_dp[section_name],
                            key=lambda i:( float(i["xmin_interval"][0]) , float(i["xmin_interval"][1]) )  )
        for entry in sorted_entries:
            all_xmins.update(entry["sorted_xmins"])

        short_section_name = section_name
        for idx, tstamp in enumerate(sorted_turns.keys()):
            if tstamp in all_xmins:
                labels[idx][label_dict[short_section_name]]=1

    return {"case_id":base_dp["id"] ,"article_lines":input_lines, "labels":labels, "label_dict":label_dict}


train_section_dataset = [get_sectionwise_xmin_multilabel_classification_dp(x) for x in tqdm(train_dataset)]
val_section_dataset = [get_sectionwise_xmin_multilabel_classification_dp(x) for x in val_dataset]
test_section_dataset = [get_sectionwise_xmin_multilabel_classification_dp(x) for x in test_dataset]

task_output_path = os.path.join(output_root_path, task_name)
os.mkdir(task_output_path)

with jsonlines.open(os.path.join(task_output_path, "train.jsonl"), "w") as w:
    for obj in train_section_dataset:
        if obj!=None:
            w.write(obj)
with jsonlines.open(os.path.join(task_output_path, "val.jsonl"), "w") as w:
    for obj in val_section_dataset:
        if obj!=None:
            w.write(obj)
with jsonlines.open(os.path.join(task_output_path, "test.jsonl"), "w") as w:
    for obj in test_section_dataset:
        if obj!=None:
            w.write(obj)

