#!/bin/sh
python3 cmf.py --a '0.8-0.1-0.1' --k 20 --reg 0.1 --lr 0.1 --tol 0.5
python3 cmf.py --a '0.8-0.1-0.1' --k 25 --reg 0.1 --lr 0.1 --tol 0.5
python3 cmf.py --a '0.8-0.1-0.1' --k 30 --reg 0.1 --lr 0.1 --tol 0.5

python3 cmf.py --a '0.8-0.1-0.1' --k 10 --reg 0.05 --lr 0.1 --tol 0.5
python3 cmf.py --a '0.8-0.1-0.1' --k 15 --reg 0.05 --lr 0.1 --tol 0.5
python3 cmf.py --a '0.8-0.1-0.1' --k 20 --reg 0.05 --lr 0.1 --tol 0.5
python3 cmf.py --a '0.8-0.1-0.1' --k 25 --reg 0.05 --lr 0.1 --tol 0.5
python3 cmf.py --a '0.8-0.1-0.1' --k 30 --reg 0.05 --lr 0.1 --tol 0.5

python3 cmf.py --a '0.8-0.1-0.1' --k 10 --reg 0.01 --lr 0.1 --tol 0.5
python3 cmf.py --a '0.8-0.1-0.1' --k 15 --reg 0.01 --lr 0.1 --tol 0.5
python3 cmf.py --a '0.8-0.1-0.1' --k 20 --reg 0.01 --lr 0.1 --tol 0.5
python3 cmf.py --a '0.8-0.1-0.1' --k 25 --reg 0.01 --lr 0.1 --tol 0.5
python3 cmf.py --a '0.8-0.1-0.1' --k 30 --reg 0.01 --lr 0.1 --tol 0.5

