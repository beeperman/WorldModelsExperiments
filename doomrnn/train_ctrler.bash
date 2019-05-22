#echo "name: $1"
#python series.py --modeldir tf_beta_vae --name $1 &>bash_log/$1
#python rnn_train.py --modeldir tf_beta_rnn --name $1 &>>bash_log/$1

FILES=tf_beta_vae/*
var=0
for f in $FILES
do
  name=${f:12:-5}
  #python series.py --modeldir tf_beta_vae --name $name &>bash_log/$name && CUDA_VISIBLE_DEVICES=$(($var%8)) python rnn_train.py --modeldir tf_beta_rnn --name $name &>>bash_log/$name &
  #CUDA_VISIBLE_DEVICES=$(($var%8)) python rnn_train.py --modeldir tf_beta_rnn --name $name &>>bash_log/$name &
  if [ $var -gt 20 ]
  then
    echo "$var $name"
    python train.py --name $name --beta &>bash_log/${name}_ctrler
  fi
  var=$((var+1))
done
