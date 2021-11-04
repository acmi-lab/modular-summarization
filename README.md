
This repository contains the code for running modular summarization pipelines as described in the publication  
`Krishna K, Khosla K, Bigham J, Lipton ZC. Generating SOAP Notes from Doctor-Patient Conversations." ACL 2021.`  
  
The paper can be found here : https://aclanthology.org/2021.acl-long.384/

 
###  Instructions

Although we can not release models trained on the confidential medical data,   we have released  models trained on the publicly available AMI dataset.  
To reproduce the results on the AMI dataset, you need to follow the steps listed below. 
For convenience, we have also created a Google Colab notebook [here](https://colab.research.google.com/drive/1P0dp4rctvhSWdzfgml4B7yt3Qketrzst?usp=sharing) that runs these steps on Google's servers (free-of-cost as of June 2021) and produces the summaries and their rouge scores.

**Step1:** Set up the environment by installing the required packages mentioned in `requirements.txt` using pip.

**Step2:** Download the `ami_models` folder from this [link](https://drive.google.com/drive/folders/12Fkv_JvhJotvDTk2Z5PZYs4POIkqmwBZ?usp=sharing) and put it at the root of the repository:

  
**Step3:** Run the following 3 commands to prepare data, run summary generation pipelines, and show the achieved rouge scores.  
  
```bash  
# command1: downloads and preprocesses AMI dataset  
./prepare_data.sh  
  
 # command2: runs the summarization pipelines on the data and computes rouge scores  
 # (before running this command, you need to download the models as shown above)  
./predict_ami.sh  
  
# command3: print the results  
python show_results.py  
```
