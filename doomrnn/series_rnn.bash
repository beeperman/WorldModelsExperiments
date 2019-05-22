#echo "name: $1"
#python series.py --modeldir tf_beta_vae --name $1 &>bash_log/$1
#python rnn_train.py --modeldir tf_beta_rnn --name $1 &>>bash_log/$1

#FILES=tf_beta_vae/*
FILES="b10.0_2
b20.0_3
b20.0_4
b30.0_2
b40.0_4
vae"
var=0
for f in $FILES
do
  #name=${f:12:-5}
  name=$f
  echo "${var}th name: $name assigned to $(($var%8))th gpu"
  #python series.py --modeldir tf_beta_vae --name $name &>bash_log/$name && CUDA_VISIBLE_DEVICES=$(($var%8)) python rnn_train.py --modeldir tf_beta_rnn --name $name &>>bash_log/$name &
  CUDA_VISIBLE_DEVICES=$(($var%8)) python rnn_train.py --modeldir tf_beta_rnn --name $name &>>bash_log/$name &
  var=$((var+1))
done
