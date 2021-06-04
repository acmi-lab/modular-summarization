import jsonlines
from hashlib import md5
from collections import defaultdict
import json
from copy import deepcopy


ami_section_names = ["abstract",
            "actions",
            "decisions",
            "problems"]


def get_utt(obj):
    return obj["speaker"]+" "+obj["txt"]

clash_resolve_dict = json.load(open("./clash_resolutions.json"))

def patch_file(fpath):
    x=list(jsonlines.open(fpath))

    for x1 in x:
        id1=x1["id"]
        if id1 not in clash_resolve_dict:
            continue
        this_resolve_dict = clash_resolve_dict[id1]
        trans1=x1["transcript"]
        tstamps = list(x1["transcript"].keys())
        tstamps = sorted(tstamps, key=float)
        for (i,t) in enumerate(tstamps):
            if t in this_resolve_dict:
                first_tstamp = tstamps[i]
                second_tstamp = tstamps[i+1]
                first_utt = get_utt(trans1[first_tstamp])
                second_utt = get_utt(trans1[second_tstamp])
                first_hash=md5(first_utt.encode("utf-8")).digest()
                second_hash=md5(second_utt.encode("utf-8")).digest()
                current_order = first_hash<second_hash

                if current_order!=this_resolve_dict[t]:
                    #swap the order of utterances
                    first_obj = deepcopy(trans1[first_tstamp])
                    second_obj = deepcopy(trans1[second_tstamp])
                    trans1[first_tstamp]=second_obj
                    trans1[second_tstamp]=first_obj
                    print(id1, first_tstamp, " was SWAPPED")

                    # adjusting the xmins accordingly
                    for sec_name in ami_section_names:
                        section = x1[sec_name]
                        for entry in section:
                            sorted_xmins = entry["sorted_xmins"]
                            for i in range(len(sorted_xmins)):
                                if sorted_xmins[i]==first_tstamp:
                                    sorted_xmins[i]=second_tstamp
                                elif sorted_xmins[i]==second_tstamp:
                                    sorted_xmins[i]=first_tstamp
                            sorted_xmins = sorted(sorted_xmins, key=lambda x:float(x))
                            entry["sorted_xmins"] = sorted_xmins
                            entry["xmin_interval"] = [sorted_xmins[0], sorted_xmins[-1]]

                else:
                    print(id1, first_tstamp, " was OKAY")




    with jsonlines.open(fpath, "w") as w:
        for obj in x:
            w.write(obj)


patch_file("../../dataset_ami/root/test.jsonl")
patch_file("../../dataset_ami/root/train.jsonl")
patch_file("../../dataset_ami/root/val.jsonl")


