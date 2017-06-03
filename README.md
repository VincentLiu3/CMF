# Collective Matrix Factorization
This is an implmentation of Collective Matrix Factorization using Newton methond.

# Data Format
In RecSys, each dataset usually has three relations (rating data, user features and item features). Each relation is stored as a matrix in coordinate format. That is, it has three columns: row, col, value. 
```
3944,2641,2
4644,2015,3
1119,1980,4
```

# Quick Usage
```
$ python3 cmf.py --out both.txt --alphas '0.8-0.1-0.1' --k 10 --reg 0.1 --lr 0.1 --tol 1.0 --verbose 1
```
You can type **$ python3 cmf.py --help** for more details about the parameters.  

# Grid Search
Write parameters you would like to search in 'cmf.sh', and run the command below.
```
$ nohup ./cmf.sh &
```

# Reference 
```
* Singh, Ajit P., and Geoffrey J. Gordon. Relational learning via collective matrix factorization. Proceedings of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2008.
```
