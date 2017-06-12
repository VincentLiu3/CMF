#!/bin/sh
for para_K in 0.1 0.01 0.001
do
	for para_reg in 0.1 0.01 0.001
	do
		for para_lr in 0.1 0.01 0.001
		do
			python cmf.py --train data/ml-1m/train.txt --test data/ml-1m/test.txt --user data/ml-1m/user.txt --item data/ml-1m/item.txt --out ml-1m.txt --alphas '0.4-0.3-0.3' --k $para_K --reg $para_reg --lr $para_lr --tol 1.0 --verbose 1 &
			python cmf.py --train data/ml-1m/train.txt --test data/ml-1m/test.txt --user data/ml-1m/user.txt --out ml-1m.txt --alphas '0.5-0.5' --k $para_K --reg $para_reg --lr $para_lr --tol 1.0 --verbose 1 &
			python cmf.py --train data/ml-1m/train.txt --test data/ml-1m/test.txt --item data/ml-1m/item.txt --out ml-1m.txt --alphas '0.5-0.5' --k $para_K --reg $para_reg --lr $para_lr --tol 1.0 --verbose 1 &
			wait
		done
	done
done