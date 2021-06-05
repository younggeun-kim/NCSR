# NCSR

Code for the CVPR2021 workshop paper "Noise Conditional Flow Model for Learning the Super-Resolution Space" 

Pre-trained weights and training script will be made fully soon.

#####

eval.py:
To eval with pretrained model, please check model_path in Confpath. Pretriained models are in code/pretrained_model
python eval.py --scale scale_factor --lrtest_path path/to/LRpath --conf_path path/to/Confpath

scale_factor is 4 or 8
path/to/LRpath is where LR images for test are
path/to/Confpath is model parameter script which is in code/confs/~~~.yml

#####
measure.py:
python measure.py OutName path/to/Ground-Truth path/to/Super-Resolution n_samples scale_factor 
path/to/Super-Resolution is code/output_dir

#####
train.py:
python train.py -opt path/to/Confpath
path/to/Confpath is model parameter script which is in code/confs/~~~.yml


