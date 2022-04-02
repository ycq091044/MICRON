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

### Folder Specification
- data
    - **processing.py**: our data preprocessing file.
    - Input (extracted from external resources)
        - PRESCRIPTIONS.csv: the prescription file from MIMIC-III raw dataset
        - DIAGNOSES_ICD.csv: the diagnosis file from MIMIC-III raw dataset
        - PROCEDURES_ICD.csv: the procedure file from MIMIC-III raw dataset
        - RXCUI2atc4.csv: this is a NDC-RXCUI-ATC4 mapping file, and we only need the RXCUI to ATC4 mapping. This file is obtained from https://github.com/sjy1203/GAMENet, where the name is called ndc2atc_level4.csv.
        - drug-atc.csv: this is a CID-ATC file, which gives the mapping from CID code to detailed ATC code (we will use the prefix of the ATC code latter for aggregation). This file is obtained from https://github.com/sjy1203/GAMENet.
        - rxnorm2RXCUI.txt: rxnorm to RXCUI mapping file. This file is obtained from https://github.com/sjy1203/GAMENet, where the name is called ndc2rxnorm_mapping.csv.
        - drug-DDI.csv: this a large file, containing the drug DDI information, coded by CID. The file could be downloaded from https://drive.google.com/file/d/1mnPc0O0ztz0fkv3HF-dpmBb8PLWsEoDz/view?usp=sharing
    - Output
        - ddi_A_final.pkl: ddi adjacency matrix
        - ddi_matrix_H.pkl: H mask structure (This file is created by **ddi_mask_H.py**)
        - ehr_adj_final.pkl: used in GAMENet baseline (if two drugs appear in one set, then they are connected)
        - records_final.pkl: The final diagnosis-procedure-medication EHR records of each patient, used for train/val/test split.
        - voc_final.pkl: diag/prod/med index to code dictionary
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

> Dataset statistics can be found below

```
#patients  6350
#clinical events  15032
#diagnosis  1958
#med  151
#procedure 1430
#avg of diagnoses  10.5089143161256
#avg of medicines  11.865886109632783
#avg of procedures  3.8436668440659925
#avg of vists  2.367244094488189
#max of diagnoses  128
#max of medicines  68
#max of procedures  50
#max of visit  29
```

### Step 1: Package Dependency

- install the following package
```python
pip install scikit-learn, dill, dnc
```
Note that torch setup may vary according to GPU hardware. Generally, run the following
```python
pip install torch
```
If you are using RTX 3090, then plase use the following, which is the right way to make torch work.
```python
python3 -m pip install --user torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
```

- install other packages if necessary
```python
pip install [xxx] # any required package if necessary, maybe do not specify the version
```

Here is a list of reference versions for all package

```shell
pandas: 1.3.0
dill: 0.3.4
torch: 1.8.0+cu111
rdkit: 2021.03.4
scikit-learn: 0.24.2
numpy: 1.21.1
```

Let us know any of the package dependency issue. Please pay special attention to pandas, some report that a high version of pandas would raise error for dill loading.


### Step 2: Data Processing

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

- download the DDI file and move it to the data folder
  download https://drive.google.com/file/d/1mnPc0O0ztz0fkv3HF-dpmBb8PLWsEoDz/view?usp=sharing
  ```python
  mv drug-DDI.csv ./data
  ```

- processing the data to get a complete records_final.pkl

  ```python
  cd ./data
  vim processing.py
  
  # line 323-325
  # med_file = './physionet.org/files/mimiciii/1.4/PRESCRIPTIONS.csv'
  # diag_file = './physionet.org/files/mimiciii/1.4/DIAGNOSES_ICD.csv'
  # procedure_file = './physionet.org/files/mimiciii/1.4/PROCEDURES_ICD.csv'
  
  python processing.py
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

