beta="50
"

#0: old wall avoiding agent
#1: new wall avoiding agent
#2: no kl tolerence

var=1
#b=100
#CUDA_VISIBLE_DEVICES=$(($var%8)) python train_controller.py --loadrnn --beta $b --int 4 &> train_bash_log/controller_b${b}_4_rnn &
for b in $beta
do
  name=${f:12:-5}
  CUDA_VISIBLE_DEVICES=$(($var%8)) python train_controller.py --beta $b --int 0 &> train_bash_log/controller_b${b}_0_nornn &
  echo "${var}th name: $0 assigned to $(($var%8))th gpu"
  var=$((var+1))
  CUDA_VISIBLE_DEVICES=$(($var%8)) python train_controller.py --beta $b --int 1 &> train_bash_log/controller_b${b}_1_nornn &
  echo "${var}th name: $0 assigned to $(($var%8))th gpu"
  var=$((var+1))
  CUDA_VISIBLE_DEVICES=$(($var%8)) python train_controller.py --beta $b --int 2 &> train_bash_log/controller_b${b}_2_nornn &
  echo "${var}th name: $0 assigned to $(($var%8))th gpu"
  var=$((var+1))
  CUDA_VISIBLE_DEVICES=$(($var%8)) python train_controller.py --beta $b --int 3 &> train_bash_log/controller_b${b}_3_nornn &
  echo "${var}th name: $0 assigned to $(($var%8))th gpu"
  var=$((var+1))
  CUDA_VISIBLE_DEVICES=$(($var%8)) python train_controller.py --beta $b --int 4 &> train_bash_log/controller_b${b}_4_nornn &
  echo "${var}th name: $0 assigned to $(($var%8))th gpu"
  var=$((var+1))
done
