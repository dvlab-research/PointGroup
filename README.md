# PointGroup
## PointGroup: Dual-Set Point Grouping for 3D Instance Segmentation (CVPR2020)
![overview](https://github.com/llijiang/PointGroup/blob/master/doc/overview.png)

Code for the paper **PointGroup:Dual-Set Point Grouping for 3D Instance Segmentation**, CVPR 2020 (Oral).

**Authors**: Li Jiang, Hengshuang Zhao, Shaoshuai Shi, Shu Liu, Chi-Wing Fu, Jiaya Jia 

[[arxiv]](https://arxiv.org/abs/2004.01658) [[video]](https://youtu.be/HMetye3gmAs)

## Introduction
Instance segmentation is an important task for scene understanding. Compared to the fully-developed 2D, 3D instance segmentation for point clouds have much room to improve. In this paper, we present PointGroup, a new end-to-end bottom-up architecture, specifically focused on better grouping the points by exploring the void space between objects. We design a two-branch network to extract point features and predict semantic labels and offsets, for shifting each point towards its respective instance centroid. A clustering component is followed to utilize both the original and offset-shifted point coordinate sets, taking advantage of their complementary strength. Further, we formulate the ScoreNet to evaluate the candidate instances, followed by the Non-Maximum Suppression (NMS) to remove duplicates.

## Installation

### Requirements
* Python 3.7.0
* Pytorch 1.1.0
* CUDA 9.0

### Virtual Environment
```
conda create -n pointgroup python==3.7
source activate pointgroup
```

### Install `PointGroup`

(1) Clone the PointGroup repository.
```
git clone https://github.com/llijiang/PointGroup.git --recursive 
cd PointGroup
```

(2) Install the dependent libraries.
```
pip install -r requirements.txt
conda install -c bioconda google-sparsehash 
```

(3) For the SparseConv, we apply the implementation of [spconv](https://github.com/traveller59/spconv). The repository is recursively downloaded at step (1). We use the version 1.0 of spconv. 

**Note:** We further modify `spconv\spconv\functional.py` to make `grad_output` contiguous. Make sure you use our modified `spconv`.

* To compile `spconv`, firstly install the dependent libraries. 
```
conda install libboost
conda install -c daleydeng gcc-5 # need gcc-5.4 for sparseconv
```
Add the `$INCLUDE_PATH$` that contains `boost` in `lib/spconv/CMakeLists.txt`. (Not necessary if it could be found.)
```
include_directories($INCLUDE_PATH$)
```

* Compile the `spconv` library.
```
cd lib/spconv
python setup.py bdist_wheel
```

* Run `cd dist` and use pip to install the generated `.whl` file.



(4) Compile the `pointgroup_ops` library.
```
cd lib/pointgroup_ops
python setup.py develop
```
If any header files could not be found, run the following commands. 
```
python setup.py build_ext --include-dirs=$INCLUDE_PATH$
python setup.py develop
```
`$INCLUDE_PATH$` is the path to the folder containing the header files that could not be found.


## Data Preparation

(1) Download the [ScanNet](http://www.scan-net.org/) v2 dataset.

(2) Put the data in the corresponding folders. 
* Copy the files `[scene_id]_vh_clean_2.ply`,  `[scene_id]_vh_clean_2.labels.ply`,  `[scene_id]_vh_clean_2.0.010000.segs.json`  and `[scene_id].aggregation.json`  into the `dataset/scannetv2/train` and `dataset/scannetv2/val` folders according to the ScanNet v2 train/val [split](https://github.com/ScanNet/ScanNet/tree/master/Tasks/Benchmark). 

* Copy the files `[scene_id]_vh_clean_2.ply` into the `dataset/scannetv2/test` folder according to the ScanNet v2 test [split](https://github.com/ScanNet/ScanNet/tree/master/Tasks/Benchmark). 

* Put the file `scannetv2-labels.combined.tsv` in the `dataset/scannetv2` folder.

The dataset files are organized as follows.
```
PointGroup
├── dataset
│   ├── scannetv2
│   │   ├── train
│   │   │   ├── [scene_id]_vh_clean_2.ply & [scene_id]_vh_clean_2.labels.ply & [scene_id]_vh_clean_2.0.010000.segs.json & [scene_id].aggregation.json
│   │   ├── val
│   │   │   ├── [scene_id]_vh_clean_2.ply & [scene_id]_vh_clean_2.labels.ply & [scene_id]_vh_clean_2.0.010000.segs.json & [scene_id].aggregation.json
│   │   ├── test
│   │   │   ├── [scene_id]_vh_clean_2.ply 
│   │   ├── scannetv2-labels.combined.tsv
```

(3) Generate input files `[scene_id]_inst_nostuff.pth` for instance segmentation.
```
cd dataset/scannetv2
python prepare_data_inst.py --data_split train
python prepare_data_inst.py --data_split val
python prepare_data_inst.py --data_split test
```

## Training
```
CUDA_VISIBLE_DEVICES=0 python train.py --config config/pointgroup_run1_scannet.yaml 
```
You can start a tensorboard session by
```
tensorboard --logdir=./exp --port=6666
```

## Inference and Evaluation

(1) If you want to evaluate on validation set, prepare the `.txt` instance ground-truth files as the following.
```
cd dataset/scannetv2
python prepare_data_inst_gttxt.py
```
Make sure that you have prepared the `[scene_id]_inst_nostuff.pth` files before. 

(2) Test and evaluate. 

a. To evaluate on validation set, set `split` and `eval` in the config file as `val` and `True`. Then run 
```
CUDA_VISIBLE_DEVICES=0 python test.py --config config/pointgroup_run1_scannet.yaml
```
An alternative evaluation method is to set `save_instance` as `True`, and evaluate with the ScanNet official [evaluation script](https://github.com/ScanNet/ScanNet/blob/master/BenchmarkScripts/3d_evaluation/evaluate_semantic_instance.py).

b. To run on test set, set (`split`, `eval`, `save_instance`) as (`test`, `False`, `True`). Then run
```
CUDA_VISIBLE_DEVICES=0 python test.py --config config/pointgroup_run1_scannet.yaml
```

c. To test with a pretrained model, run
```
CUDA_VISIBLE_DEVICES=0 python test.py --config config/pointgroup_default_scannet.yaml --pretrain $PATH_TO_PRETRAIN_MODEL$
```

## Pretrained Model
We provide a pretrained model trained on ScanNet v2 dataset. Download it [here](https://drive.google.com/file/d/1wGolvj73i-vNtvsHhg_KXonNH2eB_6-w/view?usp=sharing). Its performance on ScanNet v2 validation set is 35.2/57.1/71.4 in terms of mAP/mAP50/mAP25.


## Visualize
To visualize the point cloud, you should first install [mayavi](https://docs.enthought.com/mayavi/mayavi/installation.html). Then you could visualize by running
```
cd util 
python visualize.py --data_root $DATA_ROOT$ --result_root $RESULT_ROOT$ --room_name $ROOM_NAME$ --room_split $ROOM_SPLIT$ --task $TASK$
```
The visualization task could be `input`, `instance_gt`, `instance_pred`, `semantic_pred` and `semantic_gt`.

## Results on ScanNet Benchmark 
Quantitative results on ScanNet test set at the submisison time.
![scannet_result](https://github.com/llijiang/PointGroup/blob/master/doc/scannet_benchmark.png)

## TODO List
- [ ] Distributed multi-GPU training

## Citation
If you find this work useful in your research, please cite:
```
@article{jiang2020pointgroup,
  title={PointGroup: Dual-Set Point Grouping for 3D Instance Segmentation},
  author={Jiang, Li and Zhao, Hengshuang and Shi, Shaoshuai and Liu, Shu and Fu, Chi-Wing and Jia, Jiaya},
  journal={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2020}
}
```

## Acknowledgement
This repo is built upon several repos, e.g., [SparseConvNet](https://github.com/facebookresearch/SparseConvNet), [spconv](https://github.com/traveller59/spconv) and [ScanNet](https://github.com/ScanNet/ScanNet). 

## Contact
If you have any questions or suggestions about this repo, please feel free to contact me (lijiang@cse.cuhk.edu.hk).


