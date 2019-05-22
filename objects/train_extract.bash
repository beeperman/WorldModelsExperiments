export CUDA_VISIBLE_DEVICES=""

for i in `seq 1 64`;
do
  echo worker $i
  python train_extract.py &
  sleep 1.0
done
