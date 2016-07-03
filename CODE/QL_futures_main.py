import QL_futures_func as QL
import csv
import math
import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

length = rank + 1
episode = 3
# print(rrl2.increList)
Q = QL.algo(length, episode)
sumsr = QL.test(length,Q)
