python Clasification_train.py --encoder MaxEncoder --dataset mnist --epoch 10
python Clasification_train.py --encoder FSEncoder --dataset mnist --epoch 10
python Clasification_train.py --encoder SumEncoder --dataset mnist --epoch 10
python Clasification_train.py --encoder MeanEncoder --dataset mnist --epoch 10

python Clasification_train.py --encoder MaxEncoder --dataset modelnet10 --epoch 10
python Clasification_train.py --encoder FSEncoder --dataset modelnet10 --epoch 10
python Clasification_train.py --encoder SumEncoder --dataset modelnet10 --epoch 10
python Clasification_train.py --encoder MeanEncoder --dataset modelnet10 --epoch 10

python DeepSetPrediction_train.py --encoder MaxEncoderDSPN --dataset mnist --epoch 50
python DeepSetPrediction_train.py --encoder FSEncoderDSPN --dataset mnist --epoch 50
python DeepSetPrediction_train.py --encoder MaxEncoderDSPN --dataset mnist --epoch 50

python DeepSetPrediction_train.py --encoder MaxEncoderDSPN --dataset modelnet10 --epoch 50
python DeepSetPrediction_train.py --encoder FSEncoderDSPN --dataset modelnet10 --epoch 50
python DeepSetPrediction_train.py --encoder MaxEncoderDSPN --dataset modelnet10 --epoch 10