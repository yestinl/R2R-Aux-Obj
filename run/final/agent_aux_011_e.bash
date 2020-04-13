#name=$(echo $0 | awk -F'/' '{print $NF}' | awk -F'.' '{print $1}')
#echo $name
name=baseline_fengda
# Note that progress indicator uses bce loss not MSE
# same as 011_d but no bert
flag="--attn soft --train listener 
      --featdropout 0.3
      --angleFeatSize 128
      --feedback sample
      --mlWeight 0.2
      --aux_speaker_weight 1
      --aux_progress_weight 1
      --aux_feature_weight 1
      --aux_angle_weight 2
      --subout max --dropout 0.5 --optim rms --lr 1e-4 --iters 80000 --maxAction 35"
mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=$1 python3.6 r2r_src/train.py $flag --name $name

# Try this with file logging:
# CUDA_VISIBLE_DEVICES=$1 unbuffer python r2r_src/train.py $flag --name $name | tee snap/$name/log
