import QL_futures_func as QL
import csv
import math
import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

#DATA reading
with open('../DATA/AAPL.csv') as csvfile:
    testData = csv.reader(csvfile, delimiter=',')
    priceList = []
    for row in testData:
        priceList.append(float(row[1]))

n = len(priceList)
increList = QL.increment(priceList)
logincreList = QL.logincrement(priceList)

n = 1000
priceList = np.zeros(n)

for i in range(0,n):
	priceList[i] = np.sin(i*math.pi/100) + 2

plt.plot(priceList)
plt.xlabel('Date')
plt.ylabel('Total Return')
plt.title('L')
plt.show()

com = 0.3
priceState = np.floor(np.sign(logincreList)/2) + 1

def OsumReturn(pos, inc):
    sr = np.zeros(n)
    srsquire = np.zeros(n)
    srtotal = np.zeros(n)
    for i in range(1, n):
        sr[i] = (pos[i-1] * inc[i] - com * abs(pos[i] - pos[i-1]))
        srsquire[i] = sr[i] ** 2
        srtotal[i] = sr.sum()
    sumsr = sr.sum()
    sumsrsq = srsquire.sum()
    # print(sr)
    return sumsr, sumsrsq, srtotal

def sumReturn(pos, inc):
    sr = np.zeros(n)
    srsquire = np.zeros(n)
    srtotal = np.zeros(n)
    tt = 0
    for i in range(1, n):
        sr[i] = (pos[i-1] * inc[i] - com * abs(pos[i] - pos[i-1]))
        # Loss limit control: cover position when sum loss in 5 bar exceeds 200
        # This is also a parameter
        if (sr[i] + sr[i-1] + sr[i-1]) < -150:
            pos[i+1:i+5] = 0
        #tt += 0.5 * abs(pos[i] - pos[i-1])
        srsquire[i] = sr[i] ** 2
        srtotal[i] = sr.sum()
    sumsr = sr.sum()
    sumsrsq = srsquire.sum()
    # print(sr)
    return sumsr, sumsrsq, srtotal

length = rank + 1
episode = 10
# print(rrl2.increList)
Q = QL.algo(length, episode)
