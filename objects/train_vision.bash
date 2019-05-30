
beta="0.05
0.5
5
0.005"

#0: old wall avoiding agent
#1: new wall avoiding agent
#2: no kl tolerence

var=0
for b in $beta
do
  #name=${f:12:-5}
  CUDA_VISIBLE_DEVICES=$(($var%8)) python train_vision.py --beta $b --int 20 --datadir train_record/stage1new &> train_bash_log/betavae_b${b}_20 &
  echo "${var}th name: $0 assigned to $(($var%8))th gpu"
  var=$((var+1))
  CUDA_VISIBLE_DEVICES=$(($var%8)) python train_vision.py --beta $b --int 21 --datadir train_record/stage1new &> train_bash_log/betavae_b${b}_21 &
  echo "${var}th name: $0 assigned to $(($var%8))th gpu"
  var=$((var+1))
  CUDA_VISIBLE_DEVICES=$(($var%8)) python train_vision.py --beta $b --int 22 --datadir train_record/stage1new &> train_bash_log/betavae_b${b}_22 &
  echo "${var}th name: $0 assigned to $(($var%8))th gpu"
  var=$((var+1))
  CUDA_VISIBLE_DEVICES=$(($var%8)) python train_vision.py --beta $b --int 23 --datadir train_record/stage1new &> train_bash_log/betavae_b${b}_23 &
  echo "${var}th name: $0 assigned to $(($var%8))th gpu"
  var=$((var+1))
 # CUDA_VISIBLE_DEVICES=$(($var%8)) python train_vision.py --beta $b --int 24 --datadir train_record/stage1new &> train_bash_log/betavae_b${b}_4 &
 # echo "${var}th name: $0 assigned to $(($var%8))th gpu"
 # var=$((var+1))
done
