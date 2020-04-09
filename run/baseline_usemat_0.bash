name=baseline_usemat_0
flag="--speWeight 1 --proWeight 1 --angWeight 2 --feaWeight 1 --matWeight 1 --submit "
mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=$3 python3.6 r2r_src/train.py $flag --name $name
