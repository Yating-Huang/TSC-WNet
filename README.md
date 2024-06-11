# Tongue Size and Shape Classification Fusing Segmentation Features for Traditional Chinese Medicine Diagnosis
by [Yating Huang](https://Yating-Huang.github.io/)+, Xuechen Li+, Siting Zheng, Zhonglian Li, Sihan Li, Linlin Shen, Changen Zhou, Zhihui Lai.(+ indicates equal contribution)
## Summary:
### Intoduction:
  This repository is for our paper["Tongue Size and Shape Classification Fusing Segmentation Features for Traditional Chinese Medicine Diagnosis"](https://link.springer.com/article/10.1007/s00521-022-08054-y)
  
### Framework:
![](https://github.com/Yating-Huang/TSC-WNet/blob/main/TSC-WNet.png)

## Usage:
### Requirement:
Ubuntu 16.04+pycharm+python3.6+pytorch1.7.1  
### Preprocessing:
Clone the repository:
```
git clone https://github.com/Yating-Huang/TSC-WNet.git
cd TSC-WNet
```
## How to run:
The only thing you should do is enter the data/image.txt and correct the path of the datasets.
then run ~
example:
```
python kfold_class.py 
```
### best_model folder:
After training, the saved model is in this folder.

### the datasets:
the public Tongue dataset
link：https://github.com/BioHit/TongeImageDataset

## Note
* The repository is being updated
* Contact: Yating Huang (huangyating2019@email.szu.edu.cn)
