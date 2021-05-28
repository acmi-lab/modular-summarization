

from collections import OrderedDict, Counter
from tqdm import tqdm
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

print(train_dataset[5].keys())


task_name = "full_summarization"

def get_full_summarization_dp(base_dp):
    sorted_turns = makeTimeOrderedDict(base_dp["transcript"])
    input_lines = []
    for turn in sorted_turns.values():
        input_lines.append(turn["speaker"]+" "+turn["txt"])

    if len(input_lines)==0:
        return None

    allsection_output_lines = []

    for section_name in seq_of_sections:
        output_lines = ["@@"+section_name+"@@"]
        sorted_entries = sorted(base_dp[section_name],
                            key=lambda i:( float(i["xmin_interval"][0]) , float(i["xmin_interval"][1]) )  )
        for entry in sorted_entries:
            output_lines.append(entry["summary"])
        allsection_output_lines.extend(output_lines)


    return {"case_id":base_dp["id"] ,"article_lines":input_lines, "summary_lines":allsection_output_lines}



task_output_path = os.path.join(output_root_path, task_name)
os.mkdir(task_output_path)

train_section_dataset = [get_full_summarization_dp(x) for x in tqdm(train_dataset)]
val_section_dataset = [get_full_summarization_dp(x) for x in val_dataset]
test_section_dataset = [get_full_summarization_dp(x) for x in test_dataset]

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



task_name = "allxmin_summarization"

def get_allxmin_summarization_dp(base_dp):
    all_xmins=set()
    allsection_output_lines = []

    for section_name in seq_of_sections:
        output_lines = ["@@"+section_name+"@@"]
        sorted_entries = sorted(base_dp[section_name],
                            key=lambda i:( float(i["xmin_interval"][0]) , float(i["xmin_interval"][1]) )  )
        for entry in sorted_entries:
            output_lines.append(entry["summary"])
            all_xmins.update(entry["sorted_xmins"])
        allsection_output_lines.extend(output_lines)

    all_xmins = sorted(list(all_xmins), key=float)

    sorted_turns = [base_dp["transcript"][k] for k in all_xmins]
    input_lines = []
    for turn in sorted_turns:
        input_lines.append(turn["speaker"]+" "+turn["txt"])

    if len(input_lines)==0 or len(allsection_output_lines)==0:
        return None

    return {"case_id":base_dp["id"] ,"article_lines":input_lines, "summary_lines":allsection_output_lines}



task_output_path = os.path.join(output_root_path, task_name)
os.mkdir(task_output_path)

train_section_dataset = [get_allxmin_summarization_dp(x) for x in tqdm(train_dataset)]
val_section_dataset = [get_allxmin_summarization_dp(x) for x in val_dataset]
test_section_dataset = [get_allxmin_summarization_dp(x) for x in test_dataset]

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



task_name = "sectionwise_allxmin_summarization"

def get_sectionwise_allxmin_summarization_dps(base_dp):
    new_datapoints=[]
    for section_name in seq_of_sections:
        all_xmins=set()
        output_lines = []
        sorted_entries = sorted(base_dp[section_name],
                            key=lambda i:( float(i["xmin_interval"][0]) , float(i["xmin_interval"][1]) )  )
        for entry in sorted_entries:
            output_lines.append(entry["summary"])
            all_xmins.update(entry["sorted_xmins"])

        all_xmins = sorted(list(all_xmins), key=float)

        sorted_turns = [base_dp["transcript"][k] for k in all_xmins]
        input_lines = []
        for turn in sorted_turns:
            input_lines.append(turn["speaker"]+" "+turn["txt"])

        if len(input_lines)==0 or len(output_lines)==0:
            continue

        new_datapoints.append({"case_id":base_dp["id"] ,"article_lines":input_lines, "summary_lines":output_lines, "section":section_name})

    return new_datapoints



task_output_path = os.path.join(output_root_path, task_name)
os.mkdir(task_output_path)

train_section_dataset = [get_sectionwise_allxmin_summarization_dps(x) for x in tqdm(train_dataset)]
val_section_dataset = [get_sectionwise_allxmin_summarization_dps(x) for x in val_dataset]
test_section_dataset = [get_sectionwise_allxmin_summarization_dps(x) for x in test_dataset]

with jsonlines.open(os.path.join(task_output_path, "train.jsonl"), "w") as w:
    for lst in train_section_dataset:
        for obj in lst:
            w.write(obj)
with jsonlines.open(os.path.join(task_output_path, "val.jsonl"), "w") as w:
    for lst in val_section_dataset:
        for obj in lst:
            w.write(obj)
with jsonlines.open(os.path.join(task_output_path, "test.jsonl"), "w") as w:
    for lst in test_section_dataset:
        for obj in lst:
            w.write(obj)




task_name = "entrywise_summarization"

def get_entrywise_summarization_dps(base_dp):
    individual_entry_dps=[]
    for section_name in seq_of_sections:
        sorted_entries = sorted(base_dp[section_name],
                            key=lambda i:( float(i["xmin_interval"][0]) , float(i["xmin_interval"][1]) )  )

        for idx, entry in enumerate(sorted_entries):
            sorted_turns = [base_dp["transcript"][k] for k in entry["sorted_xmins"]]
            input_lines = []
            for turn in sorted_turns:
                input_lines.append(turn["speaker"]+" "+turn["txt"])
            output_lines = [entry["summary"]]

            if len(input_lines)==0:
                continue

            individual_entry_dps.append({"case_id":base_dp["id"] , "index_in_note":idx , "article_lines":input_lines, "summary_lines":output_lines, "section":section_name})

    return individual_entry_dps



task_output_path = os.path.join(output_root_path, task_name)
os.mkdir(task_output_path)

train_section_dataset = [get_entrywise_summarization_dps(x) for x in tqdm(train_dataset)]
val_section_dataset = [get_entrywise_summarization_dps(x) for x in val_dataset]
test_section_dataset = [get_entrywise_summarization_dps(x) for x in test_dataset]

with jsonlines.open(os.path.join(task_output_path, "train.jsonl"), "w") as w:
    for lst in train_section_dataset:
        for obj in lst:
            w.write(obj)
with jsonlines.open(os.path.join(task_output_path, "val.jsonl"), "w") as w:
    for lst in val_section_dataset:
        for obj in lst:
            w.write(obj)
with jsonlines.open(os.path.join(task_output_path, "test.jsonl"), "w") as w:
    for lst in test_section_dataset:
        for obj in lst:
            w.write(obj)

