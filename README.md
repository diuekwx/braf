This model is a mall molecules that inhibit BRAF mutations predictor. It was made from a SVM classification model. The dataset is already provided with for this model under chemical_compounds.csv and will be trained/tested upon running. 
To ensure proeper usage, please ensure Python3 is installed on your machine and then run pip -r requirements.txt

Prediction Instructions:
1. Place input.txt in the same directory as base.py.
2. Enter a valid CID (range limited to 243 as of currently, refer to chemical_compounds.csv for list) on each line.
3. Output will be placed into output.txt with each CID and its corresponding predicted classification.