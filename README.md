# Fine-Grain-Classification
## Overview
Fine-Grain Image Classification on monkey species using simple CNN and transfer learning.

## Dataset
[**10 Monkey Species**](https://www.kaggle.com/slothkong/10-monkey-species/home)

## To run the code
cd into root directory
### Simple CNN
```bash
python3 Code/main.py --Method simple_cnn
```
-  Parameters  
    - Method - method for classification. *Default :- 'tranfer_learning'*

### Tranfer learning
```bash
python3 Code/main.py --Method tranfer_learning
```
-  Parameters  
    - Method - method for classification. *Default :- 'tranfer_learning'*

## Results
### CNN
Accuracy | Confusion Matrix
:-:|:-:
![env](./Results/cnn_acc.png) | ![env](./Results/cnn_cm.png) 

### Tranfer Learning
 &nbsp; | Accuracy | Confusion Matrix
:-:|:-:|:-:
| w/o fine tune     |![env](./Results/tl_acc.png) | ![env](./Results/tf_cm.png) 
| Fine tune |![env](./Results/tl_acc_ft.png) | ![env](./Results/tf_cm_ft.png) 
