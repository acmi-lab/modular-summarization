
import os
import xml.etree.ElementTree as ET
import json
from collections import defaultdict
import glob
from pprint import pprint


train_meetings=["ES2002", "ES2005", "ES2006", "ES2007", "ES2008", "ES2009", "ES2010", "ES2012", "ES2013", "ES2015", "ES2016", "IS1000", "IS1001", "IS1002", "IS1003", "IS1004", "IS1005", "IS1006", "IS1007", "TS3005", "TS3008", "TS3009", "TS3010", "TS3011", "TS3012"]
val_meetings=["ES2003", "ES2011", "IS1008", "TS3004", "TS3006"]
test_meetings=["ES2004", "ES2014", "IS1009", "TS3003", "TS3007"]

scenario_ids = set(sum([train_meetings, val_meetings, test_meetings],[]))

transcript_dir = "../../rawAMI/words/"
summary_dir = "../../rawAMI/abstractive/"
dialogue_act_dir = "../../rawAMI/dialogueActs/"
extractive_dir = "../../rawAMI/extractive/"
corpus_resources_dir = "../../rawAMI/corpusResources/"


def get_roles():
    fpath=os.path.join(corpus_resources_dir, "meetings.xml")
    xmlroot = ET.parse(fpath).getroot()

    meetings=xmlroot.getchildren()

    all_people_roles=defaultdict(dict)
    for meeting in meetings:
        meeting_id = meeting.attrib["observation"]
        speakers = meeting.getchildren()
        for speaker in speakers:
            try:
                nxt_letter=speaker.attrib['nxt_agent']
                role = speaker.attrib['role']
                all_people_roles[meeting_id][nxt_letter]=role
            except KeyError:
                assert meeting_id[:-1] not in scenario_ids

    return all_people_roles

all_people_roles = get_roles()

# getting the summary from different meetings

meetings = os.listdir(summary_dir)
def getMeetingDesc(fpath):
    xmlroot = ET.parse(fpath).getroot()
    returndict={}
    categories = xmlroot.getchildren()
    for elem in categories:
        thistag = elem.tag
        thissents = []
        sentences = elem.getchildren()
        for sent in sentences:
            summary_sent_id = sent.attrib['{http://nite.sourceforge.net/}id']
            summary_sent_text = sent.text
            thissents.append({"summary_sent_id": summary_sent_id, "summary": summary_sent_text})
        returndict[thistag] = thissents
    return returndict

abssum_dict={}
for m in meetings:
    meeting_id=m.split(".")[0]
    fpath=os.path.join(summary_dir, m)
    abssum_dict[meeting_id]=getMeetingDesc(fpath)


# this is to get words spoken in different meetings

transcripts = os.listdir(transcript_dir)

def getMeetingTranscript(fpath):
    xmlroot = ET.parse(fpath).getroot()
    l = xmlroot.getchildren()  # this includes words as well as vocal stuff like laugh

    wordList={}

    for word in l:
        try:
            _id=word.attrib['{http://nite.sourceforge.net/}id']
            _id = _id.split("words")[1].split(")")[0]
            _id = int(_id)
        except:
            # 3 instances have wordsx which are laugh and can be ignored
            continue

        try:
            if word.tag=="w":
                this_text = word.text
            else:
                this_text = "@@NONWORD@@"

            if "starttime" in word.attrib:
                this_start = word.attrib['starttime']
            else:
                this_start = None

            wordList[_id]={
                "start":this_start,
                "text":this_text
            }
        except KeyError:
            print (word.attrib)

    return wordList

transcript_dict={}

for t in transcripts:
    meeting_id = t.split(".")[0]
    if not meeting_id in transcript_dict.keys():
        transcript_dict[meeting_id]={}

    participant_id = t.split(".")[1]

    fpath = os.path.join(transcript_dir, t)
    transcript_dict[meeting_id][participant_id] = getMeetingTranscript(fpath)




# get the sentences (dialogue acts) in transcripts from the words

dacts = os.listdir(dialogue_act_dir)
dacts = [d for d in dacts if "dialog-act" in d]

def get_sent_from_href(href):
    '''href is like ES2006b.C.words.xml#id(ES2006b.C.words0)..id(ES2006b.C.words4)'''
    header, words = href.split("#")
    if ".." in words:  # means it is a range
        start, end = words.split("..")
        word_index1=start.split("words")[1].split(")")[0]
        word_index2=end.split("words")[1].split(")")[0]

        return list(range(int(word_index1), int(word_index2)+1))   #+1 because the end is inclusive

    else:
        start=words
        word_index1=start.split("words")[1].split(")")[0]
        return [int(word_index1)]


transcript_sentences=defaultdict(dict)
for dact_file in dacts:
    fpath = os.path.join(dialogue_act_dir, dact_file)
    meeting_id, speaker_id = dact_file.split(".")[:2]

    this_speaker_words = transcript_dict[meeting_id][speaker_id]

    output = transcript_sentences[meeting_id]

    xmlroot = ET.parse(fpath).getroot()

    if meeting_id[:-1] not in scenario_ids:
        print(f"Rejected {meeting_id} because it is not a scenario meeting")
        continue
    else:
        print(f"Accepted {meeting_id}")

    speaker_role_lookup = all_people_roles[meeting_id]

    for elem in xmlroot:
        dact_id = elem.attrib['{http://nite.sourceforge.net/}id']
        children = elem.getchildren()
        for child in children:
            if child.tag=='{http://nite.sourceforge.net/}child':
                try:
                    word_indices = get_sent_from_href( child.attrib["href"] )
                    tokens = [this_speaker_words[i] for i in word_indices if i in this_speaker_words.keys()]

                    output[dact_id] = {"txt": " ".join(
                                                        [t["text"] for t in tokens
                                                                     if t["text"]!="@@NONWORD@@"]
                                                    ),
                                          "speaker": speaker_role_lookup[speaker_id],
                                          "start_timestamp": tokens[0]["start"]}

                except IndexError:
                    print(this_speaker_words)
                    print(len(this_speaker_words))
                    print(word_indices)
                    print(child.attrib["href"])
                    print(dact_file)
                    break


# if two utterances have same timestamp we move one of them slightly ahead
for (meeting_id, this_meeting_sents) in transcript_sentences.items():
    all_ts = set()
    for sent in this_meeting_sents.values():
        ts = sent["start_timestamp"]
        if ts in all_ts:
            print("resolving conflicting timestamps")
            slightly_forwarded_ts = float(ts)+0.0001
            sent["start_timestamp"] = str(slightly_forwarded_ts)

        all_ts.add(sent["start_timestamp"])

    print(f"In {meeting_id} number of clashes = { len(this_meeting_sents)-len(all_ts) }")



# Getting the noteworthy utterances part

summlink_files = glob.glob(extractive_dir+"/*summlink.xml")

abst_to_ext_map=defaultdict(list)

for fpath in summlink_files:
    xmlroot = ET.parse(fpath).getroot()
    links = xmlroot.getchildren()
    for link in links:
        pointer1, pointer2 = link.getchildren()
        if pointer1.attrib["role"]=="extractive" and pointer2.attrib["role"]=="abstractive":
            extractive_pointer = pointer1
            abstractive_pointer = pointer2
        elif pointer1.attrib["role"]=="abstractive" and pointer2.attrib["role"]=="extractive":
            extractive_pointer = pointer2
            abstractive_pointer = pointer1

        extractive_id = extractive_pointer.attrib["href"].split("#id(")[1].split(")")[0]
        abstractive_id = abstractive_pointer.attrib["href"].split("#id(")[1].split(")")[0]

        abst_to_ext_map[abstractive_id].append(extractive_id)



num_hanging_abs=0
num_good_abs=0
bad_meetings=set()

final_summary_dict = {}

for meeting_id, all_section_summaries in abssum_dict.items():
    final_summary_dict[meeting_id]={}
    this_transcript = transcript_sentences[meeting_id]

    for section_name, list_of_sents in all_section_summaries.items():
        final_summary_dict[meeting_id][section_name]=[]
        for sent in list_of_sents:
            sent_id = sent["summary_sent_id"]
            xmin_da_ids = abst_to_ext_map[sent_id]
            xmin_starts = [ this_transcript[da_id]["start_timestamp"] for da_id in xmin_da_ids
                                                              if this_transcript[da_id]["txt"]!=""]
            xmin_starts = sorted(xmin_starts, key=lambda x:float(x))

            if len(xmin_starts)==0:
                num_hanging_abs+=1
                bad_meetings.add(meeting_id)
#                 print(sent["summary"])
                continue

            num_good_abs+=1

            full_dict = {"summary":sent["summary"]}
            full_dict["sorted_xmins"] = xmin_starts
            full_dict["xmin_interval"] = [xmin_starts[0], xmin_starts[-1]]

            final_summary_dict[meeting_id][section_name].append(full_dict)

    final_summary_dict[meeting_id]["num_total_entries"]=sum([len(v) for v in final_summary_dict[meeting_id].values()])

    final_summary_dict[meeting_id]["id"]=meeting_id
    final_summary_dict[meeting_id]["num_bad_entries"]=0


print(f"Number of abs summary sents with no evidence = {num_hanging_abs}")
print(f"Number of abs summary sents with >=1 evidence = {num_good_abs}")
print(f"Number of meetings where at least one summary sent has no evidence marked = {len(bad_meetings)}")



for meeting_id, out_dict in final_summary_dict.items():
    formatted_sents = {}
    all_transcript_sents = list(transcript_sentences[meeting_id].values())
    all_transcript_sents = sorted(all_transcript_sents, key=lambda x:float(x["start_timestamp"]))
    for sent in all_transcript_sents:
        if sent["txt"]!="":
            formatted_sents[sent["start_timestamp"]] = {"speaker": sent["speaker"], "txt": sent["txt"]}

    out_dict["transcript"] = formatted_sents

print("-----------------------------------------------------------------------")
print("Sample datapoint looks like this --------------------------------------")
print("-----------------------------------------------------------------------")

pprint(final_summary_dict["ES2004a"])


all_meeting_ids = list(final_summary_dict.keys())


output_dataset_dir = "../../dataset_ami/root/"
os.mkdir(output_dataset_dir)

meetings_to_write = set(all_meeting_ids)
print(f"I have {len(meetings_to_write)} datapoints to write")

with open(os.path.join(output_dataset_dir,"train.jsonl"), "w") as w:
    for meeting_id in all_meeting_ids:
        if meeting_id[:-1] in train_meetings:  # :-1 to remove the final a,b,c letters
            out_str=json.dumps(final_summary_dict[meeting_id])
            w.write(out_str+"\n")
            meetings_to_write.remove(meeting_id)

with open(os.path.join(output_dataset_dir,"val.jsonl"), "w") as w:
    for meeting_id in all_meeting_ids:
        if meeting_id[:-1] in val_meetings:  # :-1 to remove the final a,b,c letters
            out_str=json.dumps(final_summary_dict[meeting_id])
            w.write(out_str+"\n")
            meetings_to_write.remove(meeting_id)

with open(os.path.join(output_dataset_dir,"test.jsonl"), "w") as w:
    for meeting_id in all_meeting_ids:
        if meeting_id[:-1] in test_meetings:  # :-1 to remove the final a,b,c letters
            out_str=json.dumps(final_summary_dict[meeting_id])
            w.write(out_str+"\n")
            meetings_to_write.remove(meeting_id)

print(f"After writing, I have {len(meetings_to_write)} datapoints still left, potentially because they were non-scenario")
print("They are:", meetings_to_write)

