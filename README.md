# HDCAnet
official code for "Towards blind super-resolution for stereo image: A hybrid degradation-content aware stereo image super-resolution network"

Peiming Lin <sup>1</sup> | [Sumei Li](https://seea.tju.edu.cn/info/1015/1620.htm)<sup>2</sup> | Anqi Liu <sup>3</sup>

Tianjin University

:star: If HDCAnet is helpful to your images or projects, please help star this repo. Thanks! :hugs:


### 📌 TODO
- [   ] release the pretrained models
- [ √ ] release the training code
- [ √ ] release the training and testing data

## ⚙️ Dependencies and Installation
The codes are based on [BasicSR](https://github.com/xinntao/BasicSR).
```
## git clone this repository
git clone https://github.com/LinPeiMing/HDCAnet.git
cd HDCAnet

# create an environment with python >= 3.7
conda create -n hdcanet python=3.7
conda activate hdcanet
pip install -r requirements.txt
python setup.py develop
```
## 🚀 Quick Inference
#### Step 1: Download the pretrained models
- Download the pretrained models.(TODO)
  
You can put the models into `experiment/pretrained_models`.

#### Step 2: Prepare testing data
You can put the testing images in the `datasets/test_datasets`.

#### Step 3: Modify setting
Modify the test set path and pre-training model path in `options/test/HDCAnet.yml`.

#### Step 4: Running testing command
```
sh test.sh
```

## 🌈 Train 

#### Step1: Prepare training data
Modify the training set and validation set path in `options/train/HDCAnet_train_stage1.yml`,`options/train/HDCAnet_train_stage2.yml`,


#### 2. Begin to train
We apply a two-stage training strategy by commenting out the code in test.sh or uncommenting for different training configs.
In the first stage, run this command: 
```
sh train_stage1.sh
```
In the second stage, change the pretrained model path in `options/train/HDCAnet_train_stage2.yml` to the pretrained model path obtained in stage1.
Then run this command:
```
sh train_stage2.sh
```
