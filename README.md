# Collective Matrix Factorization
This is a python implementation of Collective Matrix Factorization using Newton's method.

# Input Data Format
In this code, each relation is stored as a matrix in coordinate format. There are some examples in **data/**.
```
row,col,value
394,264,2
464,201,3
111,198,4
```

# Quick Usage
```
$ python3 cmf.py --train data/ml-1m/train.txt --test data/ml-1m/test.txt --user data/ml-1m/user.txt --item data/ml-1m/item.txt --out ml-1m.txt --alphas '0.5-0.5-0.5' --link log_dense --k 8 --reg 0.1 --lr 0.1 --iter 50 --tol 0
```
You have to use python3 to run this code. Type **python3 cmf.py --help** for more details about the parameters.  

# Reference 
```
* Singh, Ajit P., and Geoffrey J. Gordon. Relational learning via collective matrix factorization. Proceedings of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2008.
```
