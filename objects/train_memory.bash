beta="
200"

#0: old wall avoiding agent
#1: new wall avoiding agent
#2: no kl tolerence

var=0
#CUDA_VISIBLE_DEVICES=2 python train_memory.py --beta 100 --int 4 &> train_bash_log/rnn_b100_4_all &
for b in $beta
do
  #name=${f:12:-5}
  CUDA_VISIBLE_DEVICES=$(($var%8)) python train_memory.py --beta $b --int 10 &> train_bash_log/rnn_b${b}_10_all &
  echo "${var}th name: $0 assigned to $(($var%8))th gpu"
  var=$((var+1))
  CUDA_VISIBLE_DEVICES=$(($var%8)) python train_memory.py --beta $b --int 11 &> train_bash_log/rnn_b${b}_11_all &
  echo "${var}th name: $0 assigned to $(($var%8))th gpu"
  var=$((var+1))
  CUDA_VISIBLE_DEVICES=$(($var%8)) python train_memory.py --beta $b --int 12 &> train_bash_log/rnn_b${b}_12_all &
  echo "${var}th name: $0 assigned to $(($var%8))th gpu"
  var=$((var+1))
  CUDA_VISIBLE_DEVICES=$(($var%8)) python train_memory.py --beta $b --int 13 &> train_bash_log/rnn_b${b}_13_all &
  echo "${var}th name: $0 assigned to $(($var%8))th gpu"
  var=$((var+1))
 # CUDA_VISIBLE_DEVICES=$(($var%8)) python train_memory.py --beta $b --int 24 &> train_bash_log/rnn_b${b}_24_all &
 # echo "${var}th name: $0 assigned to $(($var%8))th gpu"
 # var=$((var+1))
done
