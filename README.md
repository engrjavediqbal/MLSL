
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mlsl-multi-level-self-supervised-learning-for/synthetic-to-real-translation-on-gtav-to)](https://paperswithcode.com/sota/synthetic-to-real-translation-on-gtav-to?p=mlsl-multi-level-self-supervised-learning-for)
# MLSL: Multi-Level Self-Supervised Learning for Domain Adaptation with Spatially Independent and Semantically Consistent Labeling (WACV2020)

By Javed Iqbal and Mohsen Ali

### Update
- **2020.12.05**: code release for GTA-5 to Cityscapes Adaptation

### Contents
0. [Introduction](#introduction)
0. [Requirements](#requirements)
0. [Setup](#models)
0. [Usage](#usage)
0. [Results](#results)
0. [Note](#note)
0. [Citation](#citation)

### Introduction
This repository contains the multi-level self-supervised learning framwork for domain adaptation of semantic segmnentation based on the work described in WACV 2020 paper "[MLSL: Multi-Level Self-Supervised Learning for Domain Adaptation with Spatially Independent and Semantically Consistent Labeling]". 
(https://arxiv.org/pdf/1909.13776.pdf).

### Requirements:
The code is tested in Ubuntu 16.04. It is implemented based on [MXNet 1.3.0](https://mxnet.apache.org/install/index.html?platform=Linux&language=Python&processor=GPU) and Python 2.7.12. For GPU usage, the maximum GPU memory consumption is about 7.8 GB in a single GTX 1080.


### Setup
We assume you are working in mlsl-master folder.

0. Datasets:
- Download [GTA-5](https://download.visinf.tu-darmstadt.de/data/from_games/) dataset. 
- Download [Cityscapes](https://www.cityscapes-dataset.com/).
- Put downloaded data in "data" folder.
1. Source pretrained models:
- Download [source model](https://www.dropbox.com/s/idnnk398hf6u3x9/gta_rna-a1_cls19_s8_ep-0000.params?dl=0) trained in GTA-5.
- For ImageNet pre-traine model, download [model in dropbox](https://www.dropbox.com/s/n2eewzy7bn7lhk0/ilsvrc-cls_rna-a1_cls1000_ep-0001.params?dl=0), provided by [official ResNet-38 repository](https://github.com/itijyou/ademxapp).
- Put source trained and ImageNet pre-trained models in "models/" folder
2. Spatial priors 
- Download [Spatial priors](https://www.dropbox.com/s/o6xac8r3z30huxs/prior_array.mat?dl=0) from GTA-5. Spatial priors are only used in GTA2Cityscapes. Put the prior_array.mat in "spatial_prior/gta/" folder.

### Usage
0. Set the PYTHONPATH environment variable:
~~~~
cd mlsl-master
export PYTHONPATH=PYTHONPATH:./
~~~~
1. Self-training for GTA2Cityscapes:
- MLSL(SISC):
~~~~

python issegm/solve_AO.py --num-round 6 --test-scales 2048 --scale-rate-range 0.7,1.3 --dataset gta --dataset-tgt cityscapes --split train --split-tgt train --data-root data/gta --data-root-tgt data/cityscapes --output gta2city/MLSL-SISC --model cityscapes_rna-a1_cls19_s8 --weights models/gta_rna-a1_cls19_s8_ep-0000.params --batch-images 2 --crop-size 500 --origin-size-tgt 2048 --init-tgt-port 0.15 --init-src-port 0.03 --seed-int 0 --mine-port 0.8 --mine-id-number 3 --mine-thresh 0.001 --base-lr 1e-4 --to-epoch 2 --source-sample-policy cumulative --self-training-script issegm/solve_ST1.py --kc-policy cb --prefetch-threads 2 --gpus 0 --with-prior True
~~~~
- MLSL(SISC-PWL):
~~~~
python issegm1/solve_AO.py --num-round 6 --test-scales 2048 --scale-rate-range 0.7,1.3 --dataset gta --dataset-tgt cityscapes --split train --split-tgt train --data-root data/gta --data-root-tgt data/cityscapes --output gta2city/MLSL-SISC-PWL --model cityscapes_rna-a1_cls19_s8 --weights models/gta_rna-a1_cls19_s8_ep-0000.params --batch-images 1 --crop-size 500 --origin-size-tgt 2048 --init-tgt-port 0.15 --init-src-port 0.03 --seed-int 0 --mine-port 0.8 --mine-id-number 3 --mine-thresh 0.001 --base-lr 1e-4 --to-epoch 2 --source-sample-policy cumulative --self-training-script issegm1/solve_ST.py --kc-policy cb --prefetch-threads 2 --gpus 0 --with-prior True

~~~~
3. 
- To run the code, you need to set the data paths of source data (data-root) and target data (data-root-tgt) by yourself. Besides that, you can keep other argument setting as default.

4. Evaluation
- Test in Cityscapes for model compatible with GTA-5 (Initial source trained model as example)
~~~~
python issegm/evaluate.py --data-root DATA_ROOT_CITYSCAPES --output val/gta-city --dataset cityscapes --phase val --weights models/gta_rna-a1_cls19_s8_ep-0000.params --split val --test-scales 2048 --test-flipping --gpus 0 --no-cudnn
~~~~

5. Train in source domain
- Train in GTA-5
~~~~
python issegm/train_src.py --gpus 0,1,2,3 --split train --data-root DATA_ROOT_GTA --output gta_train --model gta_rna-a1_cls19_s8 --batch-images 16 --crop-size 500 --scale-rate-range 0.7,1.3 --weights models/ilsvrc-cls_rna-a1_cls1000_ep-0001.params --lr-type fixed --base-lr 0.0016 --to-epoch 30 --kvstore local --prefetch-threads 16 --prefetcher process --cache-images 0 --backward-do-mirror --origin-size 1914
~~~~

- Train in Cityscapes, please check the [official ResNet-38 repository](https://github.com/itijyou/ademxapp).

### Note
- This code is based on [CBST](https://github.com/yzou2/CBST).
- Due to the randomness, the self-training results may slightly vary in each run. Usually the best results will be obtained in 3rd/4th round. For training in source domain, the best model usually appears during the first 30 epoches. Optimal model appearing in initial stage is also possible.








### Results
A leaderboard for state-of-the-art methods is available [here](https://github.com/engrjavediqbal/udass-leaderboard). Feel free to contact  for adding your published results.

### Citation:
If you found this useful, please cite our [paper](https://openaccess.thecvf.com/content_WACV_2020/papers/Iqbal_MLSL_Multi-Level_Self-Supervised_Learning_for_Domain_Adaptation_with_Spatially_Independent_WACV_2020_paper.pdf). 

>@inproceedings{iqbal2020mlsl,  
>&nbsp; &nbsp; &nbsp;    title={MLSL: Multi-Level Self-Supervised Learning for Domain Adaptation with Spatially Independent and  
>&nbsp; &nbsp; &nbsp;     Semantically Consistent Labeling},  
>&nbsp; &nbsp; &nbsp;     author={Javed Iqbal and Mohsen Ali},  
>&nbsp; &nbsp; &nbsp;     booktitle={The IEEE Winter Conference on Applications of Computer Vision}, 
>&nbsp; &nbsp; &nbsp;     pages={1864--1873}, 
>&nbsp; &nbsp; &nbsp;     year={2020} 
>}


Contact: javed.iqbal@itu.edu.pk
