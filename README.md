# NCSR: Noise Conditional Flow Model for Learning the Super-Resolution Space

Official NCSR training PyTorch Code for the CVPR2021 workshop paper "Noise Conditional Flow Model for Learning the Super-Resolution Space" 
**NCSR: Noise Conditional Flow Model for Learning the Super-Resolution Spacer**(https://arxiv.org/abs/2106.04428)

## Training

```.bash
python train.py -opt path/to/Confpath
```
path/to/Confpath is model parameter script which is in code/confs/~.yml

## Test

```.bash
python eval.py --scale scale_factor --lrtest_path path/to/LRpath --conf_path path/to/Confpath
```
To eval with pretrained model, please check model_path in Confpath. 
Pretriained models should be in code/pretrained_model

## Measure

```.bash
python measure.py OutName path/to/Ground-Truth path/to/Super-Resolution n_samples scale_factor 
```
path/to/Super-Resolution is code/output_dir. 
measure.py is in NTIRE21_Learning_SR_Space


Pre-trained weights and README script details will be updated fully soon.
