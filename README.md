# NCSR: Noise Conditional Flow Model for Learning the Super-Resolution Space

Official NCSR training PyTorch Code for the CVPR2021 workshop paper "Noise Conditional Flow Model for Learning the Super-Resolution Space" 

**NCSR: Noise Conditional Flow Model for Learning the Super-Resolution Space**(https://arxiv.org/abs/2106.04428)

## How to use repo
```.bash
git clone --recursive https://github.com/younggeun-kim/NCSR.git
```


## Training

```.bash
cd code
python train.py -opt path/to/Confpath
```
* path/to/Confpath is model parameter script which is in code/confs/~.yml

## Test

```.bash
cd code
python eval.py --scale scale_factor --lrtest_path path/to/LRpath --conf_path path/to/Confpath
```
* To eval with pretrained model, please check model_path in Confpath. 
* Pretriained models should be in code/pretrained_model

## Measure

```.bash
cd code/NTIRE21_Learning_SR_Space
python measure.py OutName path/to/Ground-Truth path/to/Super-Resolution n_samples scale_factor 
```
* path/to/Super-Resolution is code/output_dir. 


Pre-trained weights and README script details will be updated fully soon.
