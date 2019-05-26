
beta="50
100
200"

var=0
for b in $beta
do
  #name=${f:12:-5}
  CUDA_VISIBLE_DEVICES=$(($var%8)) python train_vision.py --beta $b --int 0 --datadir train_record/stage1 &> train_bash_log/betavae_b${b}_0
  echo "${var}th name: $0 assigned to $(($var%8))th gpu"
  var=$((var+1))
  CUDA_VISIBLE_DEVICES=$(($var%8)) python train_vision.py --beta $b --int 1 --datadir train_record/stage1 &> train_bash_log/betavae_b${b}_1
  echo "${var}th name: $0 assigned to $(($var%8))th gpu"
  var=$((var+1))
  CUDA_VISIBLE_DEVICES=$(($var%8)) python train_vision.py --beta $b --int 2 --datadir train_record/stage1 &> train_bash_log/betavae_b${b}_2
  echo "${var}th name: $0 assigned to $(($var%8))th gpu"
  var=$((var+1))
  CUDA_VISIBLE_DEVICES=$(($var%8)) python train_vision.py --beta $b --int 3 --datadir train_record/stage1 &> train_bash_log/betavae_b${b}_3
  echo "${var}th name: $0 assigned to $(($var%8))th gpu"
  var=$((var+1))
  CUDA_VISIBLE_DEVICES=$(($var%8)) python train_vision.py --beta $b --int 4 --datadir train_record/stage1 &> train_bash_log/betavae_b${b}_4
  echo "${var}th name: $0 assigned to $(($var%8))th gpu"
  var=$((var+1))
done