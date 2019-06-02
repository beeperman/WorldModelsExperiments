beta="1
"

#0: old wall avoiding agent
#1: new wall avoiding agent
#2: no kl tolerence

var=0
#b=100
#CUDA_VISIBLE_DEVICES=$(($var%8)) python train_controller.py --loadrnn --beta $b --int 4 &> train_bash_log/controller_b${b}_4_rnn &
for b in $beta
do
  name=${f:12:-5}
  CUDA_VISIBLE_DEVICES=$(($var%8)) python train_controller.py --beta $b --int 20 --seed 1 &> train_bash_log/controller_b${b}_20_nornn_s1 &
  echo "${var}th name: $0 assigned to $(($var%8))th gpu"
  var=$((var+1))
  CUDA_VISIBLE_DEVICES=$(($var%8)) python train_controller.py --beta $b --int 21 --seed 1 &> train_bash_log/controller_b${b}_21_nornn_s1 &
  echo "${var}th name: $0 assigned to $(($var%8))th gpu"
  var=$((var+1))
  CUDA_VISIBLE_DEVICES=$(($var%8)) python train_controller.py --beta $b --int 22 --seed 1 &> train_bash_log/controller_b${b}_22_nornn_s1 &
  echo "${var}th name: $0 assigned to $(($var%8))th gpu"
  var=$((var+1))
  CUDA_VISIBLE_DEVICES=$(($var%8)) python train_controller.py --beta $b --int 23 --seed 1 &> train_bash_log/controller_b${b}_23_nornn_s1 &
  echo "${var}th name: $0 assigned to $(($var%8))th gpu"
  var=$((var+1))
  #CUDA_VISIBLE_DEVICES=$(($var%8)) python train_controller.py --beta $b --int 4 &> train_bash_log/controller_b${b}_4_nornn &
  #echo "${var}th name: $0 assigned to $(($var%8))th gpu"
  #var=$((var+1))
done
