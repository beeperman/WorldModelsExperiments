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
  echo "$var $name"
  CUDA_VISIBLE_DEVICES=-1 python model.py doomreal norender beta_log/doomrnn.cma.16.64_${name}.json &>bash_log/${name}_test
  var=$((var+1))
done
