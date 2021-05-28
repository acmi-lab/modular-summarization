import pandas as pd

def add_result(fpath, method_name, dict_obj):
    df = pd.read_csv(fpath, sep="\t")
    dd = df.loc[0].to_dict()
    del dd['Unnamed: 0']
    dict_obj[method_name]=dd

dict_obj = {}

add_result(fpath="./ami_models/t5_models/ser_entrywise_conditioned_t5_base/test_outputs.jsonl.rougescores.csv",
           method_name="CLUSTER2SENT+T5-BASE (ORACLE)",
           dict_obj=dict_obj)

add_result(fpath="./ami_models/t5_models/ser_entrywise_conditioned_t5_small/test_outputs.jsonl.rougescores.csv",
           method_name="CLUSTER2SENT+T5-SMALL (ORACLE)",
           dict_obj=dict_obj)

add_result(fpath="./ami_models/t5_models/ser_sectionwise_conditioned_t5_small/test_outputs.jsonl.rougescores.csv",
           method_name="EXT2SEC+T5-SMALL (ORACLE)",
           dict_obj=dict_obj)

add_result(fpath="./ami_models/t5_models/ser_allxmins_fullsummary_t5_small/test_outputs.jsonl.rougescores.csv",
           method_name="EXT2NOTE+T5-SMALL (ORACLE)",
           dict_obj=dict_obj)

add_result(fpath="./ami_models/t5_models/ser_allxmins_fullsummary_t5_small/test_outputs.jsonl.rougescores.csv",
           method_name="EXT2NOTE+T5-SMALL (ORACLE)",
           dict_obj=dict_obj)

add_result(fpath="./ami_models/pg_models/ser_fullconversation_fullsummary/test_outputs.jsonl.rougescores.csv",
           method_name="CONV2NOTE+PG  (ORACLE)",
           dict_obj=dict_obj)

add_result(fpath="./ami_models/pg_models/ser_allxmins_fullsummary/test_outputs.jsonl.rougescores.csv",
           method_name="EXT2NOTE+PG  (ORACLE)",
           dict_obj=dict_obj)

add_result(fpath="./ami_models/pg_models/ser_sectionwise_allxmins_sectionsummary/test_outputs.jsonl.rougescores.csv",
           method_name="EXT2SEC+PG  (ORACLE)",
           dict_obj=dict_obj)

add_result(fpath="./ami_models/pg_models/ser_entrywise_summarization/test_outputs.jsonl.rougescores.csv",
           method_name="CLUSTERSENT+PG  (ORACLE)",
           dict_obj=dict_obj)

add_result(fpath="./ami_models/t5_models/ser_entrywise_conditioned_t5_base/test_outputs_on_hlstm.jsonl.rougescores.csv",
           method_name="CLUSTERSENT+T5-BASE  (HLSTM)",
           dict_obj=dict_obj)

add_result(fpath="./ami_models/t5_models/ser_entrywise_conditioned_t5_small/test_outputs_on_hlstm.jsonl.rougescores.csv",
           method_name="CLUSTERSENT+T5-SMALL  (HLSTM)",
           dict_obj=dict_obj)

add_result(fpath="./ami_models/t5_models/ser_sectionwise_conditioned_t5_small/test_outputs_on_hlstm.jsonl.rougescores.csv",
           method_name="EXT2SEC+T5-SMALL  (HLSTM)",
           dict_obj=dict_obj)

add_result(fpath="./ami_models/t5_models/ser_allxmins_fullsummary_t5_small/test_outputs_on_hlstm.jsonl.rougescores.csv",
           method_name="EXT2NOTE+T5-SMALL  (HLSTM)",
           dict_obj=dict_obj)

add_result(fpath="./ami_models/pg_models/ser_allxmins_fullsummary/test_outputs_on_hlstm.jsonl.rougescores.csv",
           method_name="EXT2NOTE+PG  (HLSTM)",
           dict_obj=dict_obj)

add_result(fpath="./ami_models/pg_models/ser_sectionwise_allxmins_sectionsummary/test_outputs_on_hlstm.jsonl.rougescores.csv",
           method_name="EXT2SEC+PG  (HLSTM)",
           dict_obj=dict_obj)

add_result(fpath="./ami_models/pg_models/ser_entrywise_summarization/test_outputs_on_hlstm.jsonl.rougescores.csv",
           method_name="CLUSTER2SENT+PG  (HLSTM)",
           dict_obj=dict_obj)


df_complete = pd.DataFrame(dict_obj)
df_complete = df_complete.transpose()
df_complete = df_complete*100
df_complete = df_complete.round(2)

print(df_complete)


