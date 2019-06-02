beta="50
100
"

#0: old wall avoiding agent
#1: new wall avoiding agent
#2: no kl tolerence

var=0
for b in $beta
do
  #name=${f:12:-5}
  CUDA_VISIBLE_DEVICES=$(($var%8)) python train_vision.py --beta $b --int 14 --datadir train_record/stage1new &> train_bash_log/betavae_b${b}_14 &
  echo "${var}th name: $0 assigned to $(($var%8))th gpu"
  var=$((var+1))
  CUDA_VISIBLE_DEVICES=$(($var%8)) python train_vision.py --beta $b --int 15 --datadir train_record/stage1new &> train_bash_log/betavae_b${b}_14 &
  echo "${var}th name: $0 assigned to $(($var%8))th gpu"
  var=$((var+1))
  CUDA_VISIBLE_DEVICES=$(($var%8)) python train_vision.py --beta $b --int 16 --datadir train_record/stage1new &> train_bash_log/betavae_b${b}_14 &
  echo "${var}th name: $0 assigned to $(($var%8))th gpu"
  var=$((var+1))
  CUDA_VISIBLE_DEVICES=$(($var%8)) python train_vision.py --beta $b --int 17 --datadir train_record/stage1new &> train_bash_log/betavae_b${b}_14 &
  echo "${var}th name: $0 assigned to $(($var%8))th gpu"
  var=$((var+1))
 # CUDA_VISIBLE_DEVICES=$(($var%8)) python train_vision.py --beta $b --int 24 --datadir train_record/stage1new &> train_bash_log/betavae_b${b}_24 &
 # echo "${var}th name: $0 assigned to $(($var%8))th gpu"
 # var=$((var+1))
done
