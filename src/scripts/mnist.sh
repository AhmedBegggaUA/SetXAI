#MNIST CLASSIFICATION
python3 ../Clasification_train.py --encoder MaxEncoder --epoch 50 --store
python3 ../Clasification_train.py --encoder SumEncoder --epoch 50 --store
python3 ../Clasification_train.py --encoder MeanEncoder --epoch 50 --store
python3 ../Clasification_train.py --encoder FSEncoder --epoch 50 --store
#MNIST REGRESION

