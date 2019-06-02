beta="
1"

#0: old wall avoiding agent
#1: new wall avoiding agent
#2: no kl tolerence

var=4
#CUDA_VISIBLE_DEVICES=2 python train_memory.py --beta 100 --int 4 &> train_bash_log/rnn_b100_4_all &
for b in $beta
do
  #name=${f:12:-5}
  CUDA_VISIBLE_DEVICES=$(($var%8)) python train_memory.py --beta $b --int 20 --datadir train_record/stagett --name tt &> train_bash_log/rnn_b${b}_20_tt &
  echo "${var}th name: $0 assigned to $(($var%8))th gpu"
  var=$((var+1))
  CUDA_VISIBLE_DEVICES=$(($var%8)) python train_memory.py --beta $b --int 21 --datadir train_record/stagett --name tt &> train_bash_log/rnn_b${b}_21_tt &
  echo "${var}th name: $0 assigned to $(($var%8))th gpu"
  var=$((var+1))
  CUDA_VISIBLE_DEVICES=$(($var%8)) python train_memory.py --beta $b --int 22 --datadir train_record/stagett --name tt &> train_bash_log/rnn_b${b}_22_tt &
  echo "${var}th name: $0 assigned to $(($var%8))th gpu"
  var=$((var+1))
  CUDA_VISIBLE_DEVICES=$(($var%8)) python train_memory.py --beta $b --int 23 --datadir train_record/stagett --name tt &> train_bash_log/rnn_b${b}_23_tt &
  echo "${var}th name: $0 assigned to $(($var%8))th gpu"
  var=$((var+1))
 # CUDA_VISIBLE_DEVICES=$(($var%8)) python train_memory.py --beta $b --int 24 &> train_bash_log/rnn_b${b}_24_all &
 # echo "${var}th name: $0 assigned to $(($var%8))th gpu"
 # var=$((var+1))
done
