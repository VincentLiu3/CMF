# Collective Matrix Factorization
This is an implmentation of Collective Matrix Factorization using Newton's method.

# Data Format
In this model, each relation is stored as a matrix in coordinate format. 
```
row,col,value
394,264,2
464,201,3
111,198,4
```

# Quick Usage
```
$ python cmf.py --train data/yelp/train.txt --test data/yelp/test.txt --user data/yelp/user.txt --item data/yelp/item.txt --out yelp.txt --alphas '0.6-0.2-0.2' --k 10 --reg 0.1 --lr 0.1 --tol 1 --verbose 1
```
You can type **$ python cmf.py --help** for more details about the parameters.  

# Reference 
```
Singh, Ajit P., and Geoffrey J. Gordon. Relational learning via collective matrix factorization. Proceedings of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2008.
```
