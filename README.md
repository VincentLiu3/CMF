# Collective Matrix Factorization
This is an implmentation of Collective Matrix Factorization using Newton methond to minimize objective loss.

# Data Format
Each relation is stored as a matrix in coordinate format. That is, it has three columns: row, col, value. For example, 
```
3944,2641,2
4644,2015,3
1119,1980,4
```

# Quick Usage
```
$ python3 cmf.py --alphas '0.8-0.1-0.1' --k 10 --reg 0.1 --lr 0.1
```
You can type **"$ python3 cmf.py --help"** for more details about the parameters.  

# Grid Search
```
$ python3 grid.py
```

# Reference 
* Singh, Ajit P., and Geoffrey J. Gordon. Relational learning via collective matrix factorization. Proceedings of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2008.
