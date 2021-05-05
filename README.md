# Code Instructions
IJCAI2021 - MICRON - Medication Change Prediction

### cite
```bibtex
@inproceedings{yang2021micron,
    title = {Change Matters: Medication Change Prediction with Recurrent Residual Networks},
    author = {Yang, Chaoqi and Xiao, Cao and Glass, Lucas and Sun, Jimeng},
    booktitle = {Proceedings of the Thirtieth International Joint Conference on
               Artificial Intelligence, {IJCAI} 2021},
    year = {2021}
}
```
### dependency
python 3.7, scipy 1.1.0, pandas 0.25.3, torch 1.4.0, numpy 1.16.5, dill

### usage
run ```python MICRON.py``` to test the result (we already attach the trained model "Epoch_39_JA_0.5205_DDI_0.06709.model")
- Note that due to the instruction of MIMIC-III dataset, we only attach a sample of processed dataset in the data folder. Users could refer to https://mimic.physionet.org/ to request the original MIMIC-III dataset and use our provided processing scripts to extract the whole processed dataset.

### reproductive code folder structure
- data/
    - mapping files that collected from external sources
        - drug-atc.csv: drug to atc code mapping file
        - drug-DDI.csv: can be downloaded from https://www.dropbox.com/s/8os4pd2zmp2jemd/drug-DDI.csv?dl=0
        - ndc2atc_level4.csv: NDC code to ATC-4 code mapping file
        - ndc2xnorm_mapping.txt: NDC to xnorm mapping file
        - id2drug.pkl: drug ID to drug SMILES string dict
    - other files that is generated from mapping files and the original MIMIC dataset (after download MIMIC-III, user could use preprocessing.py to generate the following)
        - data_final.pkl: intermediate result
        - ddi_matrix.pkl: ddi matrix
        - records_final.pkl: patient visit-level record (this is a sample, please Under MIMIC Dataset policy, we are not allowed to distribute the datasets. Practioners could go to https://mimic.physionet.org/about/mimic/ and requrest the access to MIMIC-III dataset and then run our processing script ```preprocessing.py``` to get the complete preprocessed dataset file.)
        - voc_final.pkl: diag/prod/med dictionary
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


