# Strongly Pretrained Class Incremental Learning

This is code base for the following paper:

### [Class-Incremental Learning with Strong Pre-trained Models](https://arxiv.org/abs/2204.03634)
Tz-Ying Wu, Gurumurthy Swaminathan, Zhizhong Li, Avinash Ravichandran, Nuno Vasconcelos, Rahul Bhotika, Stefano Soatto

Please read our paper for details!

## Installation
To create a conda environment to run the project, simply run  
`conda env create -f CIL.yml`.

## Data
Create a soft link of the imagenet folder (the root folder that includes train/val image folders) at `prepro/data/imagenet`.

## Experiments
### 800-40-40
```shell
bash scripts/getresults80040.sh -l layer4 -n 10  # for resnet10
bash scripts/getresults80040.sh -l layer4 -n 18  # for resnet18
bash scripts/getresults80040.sh -l fc -n 10  # for resnet10, fc-only
bash scripts/getresults80040.sh -l fc -n 18  # for resnet18, fc-only
```

### 500-50
```shell
bash scripts/getresults50050.sh -l layer4 -n 10
bash scripts/getresults50050.sh -l layer4 -n 18
bash scripts/getresults50050.sh -l fc -n 10
bash scripts/getresults50050.sh -l fc -n 18
```

