name=$(echo $0 | awk -F'/' '{print $NF}' | awk -F'.' '{print $1}')
echo $name
# Note that progress indicator uses bce loss not MSE
# same as 011_e but no bert
flag="--attn soft --train auglistener --selfTrain 
      --aug tasks/R2R/data/aug_paths.json
      --speaker snap/speaker/state_dict/best_val_unseen_bleu 
      --load snap/agent_aux_011_e/state_dict/best_val_unseen
      --angleFeatSize 128
      --accumulateGrad
      --featdropout 0.4
      --aux_speaker_weight 0.5
      --aux_progress_weight 0.5
      --aux_matching_weight 0.5
      --aux_feature_weight 0.5
      --aux_angle_weight 1
      --subout max --optim rms --lr 1e-4 --iters 200000 --maxAction 35"
mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=$1 python3.6 r2r_src/train.py $flag --name $name 

# Try this with file logging:
# CUDA_VISIBLE_DEVICES=$1 unbuffer python r2r_src/train.py $flag --name $name | tee snap/$name/log



