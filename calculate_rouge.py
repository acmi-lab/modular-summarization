import argparse

import json
import os
import re
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from utils.section_names import ami_section_names

from tempfile import mkdtemp
import shutil
from pyrouge import Rouge155

# taken from
# https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



def get_full_comparisions_dict(dataset, split, algo, dataset_dir, eval_objs):
    if dataset=="ami":
        canonical_seq_of_sections = ami_section_names
    else:
        raise NotImplementedError

    ground_truth_fpath = os.path.join(dataset_dir, "full_summarization", f"{split}.jsonl")

    fullsummary_comparisions={}
    with open(ground_truth_fpath, "r", encoding="utf-8") as src:
        for line in src:
            line = line.strip()
            if line=="":
                continue
            else:
                obj = json.loads(line)
                fullsummary_comparisions[obj["case_id"]]={"ground_truth":" ".join(obj["summary_lines"]).lower()}


    if algo=="conv2note" or algo=="ext2note":
        for obj in eval_objs:
            case_id = obj["case_id"]
            fullsummary_comparisions[case_id]["generated"]=obj["prediction"]

    elif algo=="cluster2sent":
        test_output_dict = defaultdict(lambda: defaultdict(list))
        for obj in eval_objs:
            case_id = obj["case_id"]
            section = obj["section"]
            test_output_dict[case_id][section].append(obj)

        for case_id in test_output_dict.keys():
            for section in canonical_seq_of_sections:
                lst = test_output_dict[case_id][section]
                lst.append({"ground_truth":"@@"+section+"@@",
                            "prediction":"@@"+section+"@@",
                            "section":section,
                            "index_in_note":-1})
                test_output_dict[case_id][section] = sorted(test_output_dict[case_id][section], key=lambda q:q["index_in_note"])

        to_keys=set(test_output_dict.keys())
        fc_keys=set(fullsummary_comparisions.keys())
        for k in to_keys.difference(fc_keys):
            print("warning: deleting ", k, "from predictions because it was not present in the ground truth")
            del test_output_dict[k]

        assert test_output_dict.keys() == fullsummary_comparisions.keys()

        for case_id in test_output_dict.keys():
            full_soapnote_str=""
            generated_summary_dict = test_output_dict[case_id]
            for section in canonical_seq_of_sections:
                section_summary = " ".join([ line["prediction"] for line in generated_summary_dict[section] ])
                full_soapnote_str+=section_summary+" "

            fullsummary_comparisions[case_id]["generated"]=full_soapnote_str.strip()

    elif algo=="ext2sec":
        test_output_dict = defaultdict(lambda: defaultdict(list))
        for obj in eval_objs:
            case_id = obj["case_id"]
            section = obj["section"]
            test_output_dict[case_id][section].append(obj)

        for case_id in test_output_dict.keys():
            for section in canonical_seq_of_sections:
                lst = test_output_dict[case_id][section]
                test_output_dict[case_id][section] = [{"ground_truth":"@@"+section+"@@",
                            "prediction":"@@"+section+"@@",
                            "section":section}] + lst       # no need to sort as there are only 2 entries

        to_keys=set(test_output_dict.keys())
        fc_keys=set(fullsummary_comparisions.keys())
        for k in to_keys.difference(fc_keys):
            print("warning: deleting ", k, "from predictions because it was not present in the ground truth")
            del test_output_dict[k]

        assert test_output_dict.keys() == fullsummary_comparisions.keys()


        for case_id in test_output_dict.keys():
            full_soapnote_str=""
            generated_summary_dict = test_output_dict[case_id]
            for section in canonical_seq_of_sections:
                section_summary = " ".join([ line["prediction"] for line in generated_summary_dict[section] ])
                full_soapnote_str+=section_summary+" "

            fullsummary_comparisions[case_id]["generated"]=full_soapnote_str.strip()

    else:
        raise NotImplementedError

    return fullsummary_comparisions


def evaluate_rouge_using_pyrouge(dict_of_comparisions):
    '''dict_of_comparisions is a dictionary with case_id as key and the value is a dict such that
    dict_of_comparisions[case_id][generated] is generated output
    dict_of_comparisions[case_id][ground_truth] is reference summary
    '''
    rouge_base_path = os.path.join(os.environ["HOME"], "ROUGE-1.5.5")

    sys_dir = mkdtemp()
    ref_dir = mkdtemp()

    r = Rouge155(rouge_dir=rouge_base_path)
    r.system_dir = sys_dir
    r.model_dir = ref_dir
    r.system_filename_pattern = 'generated.(\d+).txt'
    r.model_filename_pattern = 'reference.[A-Z].#ID#.txt'

    number_of_cases=len(dict_of_comparisions)
    number_of_digits_required = len(str(len(dict_of_comparisions)-1))

    for _i,k in enumerate(sorted(dict_of_comparisions.keys())):
        v = dict_of_comparisions[k]
        generated = v["generated"]
        reference = v["ground_truth"]
        index = str(_i).zfill(number_of_digits_required)

        with open(os.path.join(sys_dir, f"generated.{index}.txt"), "w") as w:
            w.write(generated)

        with open(os.path.join(ref_dir, f"reference.A.{index}.txt"), "w") as w:
            w.write(reference.lower())

    output = r.convert_and_evaluate()
    output_dict = r.output_to_dict(output)

    shutil.rmtree(sys_dir)
    shutil.rmtree(ref_dir)

    def cleanup(rouge_obj):
        pat = re.compile("(.*)system$")
        match_groups = list(pat.match(rouge_obj._system_dir).groups())
        assert(len(match_groups)==1)
        dir_to_remove = match_groups[0]
        print(dir_to_remove)

        shutil.rmtree(rouge_obj._config_dir)
        shutil.rmtree(dir_to_remove)

    cleanup(r)

    scores_df = pd.DataFrame({"Score":output_dict}).transpose()
    print(scores_df[["rouge_1_f_score","rouge_2_f_score","rouge_l_f_score"]])
    mini_df = pd.DataFrame({"rouge1":scores_df["rouge_1_f_score"],
                            "rouge2":scores_df["rouge_2_f_score"],
                            "rougel":scores_df["rouge_l_f_score"]})
    return mini_df



def evaluate_rouge_using_huggingface(dict_of_comparisions):
    import nlp
    rouge = nlp.load_metric('rouge')
    all_scores = {"rouge1":[],"rouge2":[],"rougel":[]}
    for _i,k in enumerate(tqdm(sorted(dict_of_comparisions.keys()))):
        v = dict_of_comparisions[k]
        generated = v["generated"]
        reference = v["ground_truth"]
    #     second one is ground truth
        rouge.add(generated, reference)
    score = rouge.compute(rouge_types=["rouge1", "rouge2", "rougeL"])
    all_scores["rouge1"].append( score['rouge1'].mid.fmeasure )
    all_scores["rouge2"].append( score['rouge2'].mid.fmeasure )
    all_scores["rougel"].append( score['rougeL'].mid.fmeasure )

    for (k,v) in all_scores.items():
        all_scores[k]=np.mean(v)

    scores_df = pd.DataFrame({"Score":all_scores}).transpose()
    return scores_df


def evaluate_rouge(rouge_impl, dataset, split, algo, dataset_dir, eval_objs):
    '''"eval_objs should either be a path to file or correctly formatted output dict"'''
    if type(eval_objs)==str:
        # assume that it is filename
        eval_file = eval_objs
        eval_objs = []
        with open(eval_file,"r") as reader:
            for line in reader:
                line = line.strip()
                if line=="":
                    continue
                else:
                    eval_objs.append(json.loads(line))


    fullsummary_comparisions = get_full_comparisions_dict(dataset, split, algo, dataset_dir, eval_objs)
    if rouge_impl=="pyrouge":
        return evaluate_rouge_using_pyrouge(fullsummary_comparisions)
    elif rouge_impl=="huggingface":
        return evaluate_rouge_using_huggingface(fullsummary_comparisions)
    else:
        raise NotImplementedError

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='calculate rouge scores')

    parser.add_argument(
        '-dataset',
        dest='dataset',
        help='Dataset name (either medical or ami)',
        type=str,
        required=True,
    )
    parser.add_argument(
        '-split',
        dest='split',
        help='train/test/val',
        type=str,
        required=True,
    )
    parser.add_argument(
        '-algo',
        dest='algo',
        help='algorithm used to generate the summary output file - conv2note, ext2note, ext2sec, cluster2sent',
        type=str,
        required=True,
    )
    parser.add_argument("-save_results", type=str2bool, nargs='?',
                const=True, default=False,
                help="save results")
    parser.add_argument(
        '-eval_file',
        dest='eval_file',
        help='file containing generated summaries',
        type=str,
        required=True,
    )
    parser.add_argument(
        '-rouge_impl',
        dest='rouge_impl',
        help='implementation of rouge to use - one of  pyrouge or sumeval',
        default='pyrouge',
        type=str
    )

    args = parser.parse_args()
    dataset = args.dataset
    algo = args.algo
    dataset_dir=f"dataset_{dataset}"
    eval_file=args.eval_file
    rouge_impl=args.rouge_impl
    save_results=args.save_results
    split=args.split


    result_df = evaluate_rouge(rouge_impl, dataset, split, algo, dataset_dir, eval_file)
    print(result_df)

    if save_results:
        result_df.to_csv(eval_file+".rougescores.csv", sep="\t", header=True)


