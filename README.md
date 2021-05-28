This repository contains the code for running modular summarization pipelines as described in the publication
`Krishna K, Khosla K, Bigham J, Lipton ZC. Generating SOAP Notes from Doctor-Patient Conversations." ACL 2021.`

Although we can not release models trained on the confidential medical data,
we have released  models trained on the publicly available AMI dataset.

Download the `ami_models` folder from the link given below and put it at the root of the repository:
https://drive.google.com/drive/folders/12Fkv_JvhJotvDTk2Z5PZYs4POIkqmwBZ?usp=sharing

To reproduce the results, you need to follow the following 3 steps.

```
# step1: downloads and preprocesses AMI dataset
./prepare_data.sh

 # step2: runs the summarization pipelines on the data and computes rouge scores
 # (before running this command, you need to download the models as shown above  )
./predict_ami.sh

# step3: print the results
python show_results.py
```
