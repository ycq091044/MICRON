# Code Instructions
IJCAI2021 - MICRON - Medication Change Prediction

### Citation
```bibtex
@inproceedings{yang2021micron,
    title = {Change Matters: Medication Change Prediction with Recurrent Residual Networks},
    author = {Yang, Chaoqi and Xiao, Cao and Glass, Lucas and Sun, Jimeng},
    booktitle = {Proceedings of the Thirtieth International Joint Conference on
               Artificial Intelligence, {IJCAI} 2021},
    year = {2021}
}
```
### （Reference) Dependency
python 3.7, scipy 1.1.0, pandas 0.25.3, torch 1.4.0, numpy 1.16.5, dill

### Reproductive code folder structure
- data/
    - !!! ``refer to`` https://github.com/ycq091044/SafeDrug ``for more information. The preparation files here are a subset from`` https://github.com/ycq091044/SafeDrug ``and the preprocessing file is a little bit different.``
    - mapping files that collected from external sources
        - drug-atc.csv: this is a CID-ATC file, which gives the mapping from CID code to detailed ATC code (we should truncate later)
        - drug-DDI.csv: this a large file, could be downloaded from https://drive.google.com/file/d/1mnPc0O0ztz0fkv3HF-dpmBb8PLWsEoDz/view?usp=sharing
        - ndc2atc_level4.csv: this is a NDC-RXCUI-ATC5 file, which gives the mapping information
        - ndc2rxnorm_mapping.txt: rxnorm to RXCUI file
    - other files that generated from mapping files and MIMIC dataset (we attach these files here, user could use our provided scripts to generate)
        - data_final.pkl: intermediate result
        - ddi_A_final.pkl: ddi matrix
        - ehr_adj_final.pkl: used in GAMENet baseline (refer to https://github.com/sjy1203/GAMENet)
        - (important) records_final.pkl: 100 patient visit-level record samples. Under MIMIC Dataset policy, we are not allowed to distribute the datasets. Practioners could go to https://physionet.org/content/mimiciii/1.4/ and requrest the access to MIMIC-III dataset and then run our processing script to get the complete preprocessed dataset file.
        - voc_final.pkl: diag/prod/med index to code dictionary
    - dataset processing scripts
        - preprocessing.py: is used to process the MIMIC original dataset
- src/
    - MICRON.py: our model
    - baselines:
        - GAMENet.py
        - Leap.py
        - Retain.py
        - DualNN.py
		- SimNN.py
    - setting file
        - model.py
        - util.py
        - layer.py

### Data Processing

- Go to https://physionet.org/content/mimiciii/1.4/ to download the MIMIC-III dataset (You may need to get the certificate)

  ```python
  cd ./data
  wget -r -N -c -np --user [account] --ask-password https://physionet.org/files/mimiciii/1.4/
  ```

- go into the folder and unzip three main files

  ```python
  cd ./physionet.org/files/mimiciii/1.4
  gzip -d PROCEDURES_ICD.csv.gz # procedure information
  gzip -d PRESCRIPTIONS.csv.gz  # prescription information
  gzip -d DIAGNOSES_ICD.csv.gz  # diagnosis information
  ```

- change the path in processing.py and processing the data to get a complete records_final.pkl

  ```python
  vim processing.py
  
  # line 294~296
  # med_file = './physionet.org/files/mimiciii/1.4/PRESCRIPTIONS.csv'
  # diag_file = './physionet.org/files/mimiciii/1.4/DIAGNOSES_ICD.csv'
  # procedure_file = './physionet.org/files/mimiciii/1.4/PROCEDURES_ICD.csv'
  
  python preprocessing.py
  ```

### Run the code

```python
python MICRON.py
```

configurations:

```shell
usage: MICRON.py [-h] [--Test] [--model_name MODEL_NAME]
                 [--resume_path RESUME_PATH] [--lr LR]
                 [--weight_decay WEIGHT_DECAY] [--dim DIM]

optional arguments:
  -h, --help            show this help message and exit
  --Test                test mode
  --model_name MODEL_NAME
                        model name
  --resume_path RESUME_PATH
                        resume path
  --lr LR               learning rate
  --weight_decay WEIGHT_DECAY
                        learning rate
  --dim DIM             dimension
```

