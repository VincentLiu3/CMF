#!/bin/sh
python3 cmf.py --train data/ml-1m/train.txt --test data/ml-1m/test.txt --user data/ml-1m/user.txt --item data/ml-1m/item.txt --out ml_1m.txt --alphas '0.6-0.4' --k 10 --reg 0.1 --lr 0.1 --tol 1.0 --verbose 1 &
python3 cmf.py --train data/ml-1m/train.txt --test data/ml-1m/test.txt --user data/ml-1m/user.txt --item data/ml-1m/item.txt --out ml_1m.txt --alphas '0.6-0.4' --k 10 --reg 0.01 --lr 0.1 --tol 1.0 --verbose 1 &

python3 cmf.py --train data/ml-1m/train.txt --test data/ml-1m/test.txt --user data/ml-1m/user.txt --out ml_1m.txt --alphas '0.6-0.4' --k 10 --reg 0.1 --lr 0.1 --tol 1.0 --verbose 1 &
python3 cmf.py --train data/ml-1m/train.txt --test data/ml-1m/test.txt --user data/ml-1m/user.txt --out ml_1m.txt --alphas '0.6-0.4' --k 10 --reg 0.01 --lr 0.1 --tol 1.0 --verbose 1 &

python3 cmf.py --train data/ml-1m/train.txt --test data/ml-1m/test.txt --item data/ml-1m/item.txt --out ml_1m.txt --alphas '0.6-0.4' --k 10 --reg 0.1 --lr 0.1 --tol 1.0 --verbose 1 &
python3 cmf.py --train data/ml-1m/train.txt --test data/ml-1m/test.txt --item data/ml-1m/item.txt --out ml_1m.txt --alphas '0.6-0.4' --k 10 --reg 0.01 --lr 0.1 --tol 1.0 --verbose 1 &
wait