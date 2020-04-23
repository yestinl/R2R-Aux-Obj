name=R2R_Envdrop_0
flag="--attn soft --train listener 
      --featdropout 0.3
      --angleFeatSize 128
      --feedback sample
      --mlWeight 0.2
      --subout max --dropout 0.5 --optim rms --lr 1e-4 --iters 80000 --maxAction 35 --upload"
mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=$0 python3.6 r2r_src/train.py $flag --name $name

# Try this with file logging:
# CUDA_VISIBLE_DEVICES=$3 python r2r_src/train.py $flag --name $name | tee snap/$name/log
