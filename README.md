# NCSR

Code for the CVPR2021 workshop paper "Noise Conditional Flow Model for Learning the Super-Resolution Space" 

Code will be made fully soon.

code/train.py is training code
code/eval.py saves SR images in code/output_dir by using pretrained model in Confpath
code/NTIRE21_Learning_SR_Space/measure.py is metric code which calculates Diversity, LRPSNR and LPIPS 

#####
eval.py
To eval with pretrained model, please check model_path in Confpath. Pretriained models are in code/pretrained_model
python eval.py --scale scale_factor --lrtest_path path/to/LRpath --conf_path path/to/Confpath

scale_factor is 4 or 8
path/to/LRpath is where LR images for test are
path/to/Confpath is model parameter script which is in code/confs/~~~.yml

#####
measure.py
python measure.py OutName path/to/Ground-Truth path/to/Super-Resolution n_samples scale_factor 
path/to/Super-Resolution is code/output_dir

#####
train.py
python train.py -opt path/to/Confpath
path/to/Confpath is model parameter script which is in code/confs/~~~.yml

Do eval.py to save SR images and do measure.py to calculate scores
4X.yml is X4 conf file and 8X.yml is X8 conf file
