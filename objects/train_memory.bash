beta="
1"

#0: old wall avoiding agent
#1: new wall avoiding agent
#2: no kl tolerence

var=0
for b in $beta
do
  #name=${f:12:-5}
  CUDA_VISIBLE_DEVICES=$(($var%8)) python train_memory.py --beta $b --int 20 &> train_bash_log/rnn_b${b}_20_all &
  echo "${var}th name: $0 assigned to $(($var%8))th gpu"
  var=$((var+1))
  CUDA_VISIBLE_DEVICES=$(($var%8)) python train_memory.py --beta $b --int 21 &> train_bash_log/rnn_b${b}_21_all &
  echo "${var}th name: $0 assigned to $(($var%8))th gpu"
  var=$((var+1))
  CUDA_VISIBLE_DEVICES=$(($var%8)) python train_memory.py --beta $b --int 22 &> train_bash_log/rnn_b${b}_22_all &
  echo "${var}th name: $0 assigned to $(($var%8))th gpu"
  var=$((var+1))
  CUDA_VISIBLE_DEVICES=$(($var%8)) python train_memory.py --beta $b --int 23 &> train_bash_log/rnn_b${b}_23_all &
  echo "${var}th name: $0 assigned to $(($var%8))th gpu"
  var=$((var+1))
 # CUDA_VISIBLE_DEVICES=$(($var%8)) python train_memory.py --beta $b --int 24 &> train_bash_log/rnn_b${b}_24_all &
 # echo "${var}th name: $0 assigned to $(($var%8))th gpu"
 # var=$((var+1))
done
