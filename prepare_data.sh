mkdir rawAMI
cd rawAMI
wget http://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip
unzip ami_public_manual_1.6.2.zip
cd ..

mkdir dataset_ami

cd dataset_creators/AMI
python create_dataset.py
python variant_tasks_creator.py
python xmin_dataset_creator.py


cd ../../


