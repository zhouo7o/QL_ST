import tf_rrl2_futures as tff
import csv
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

with open('ltrain.csv') as csvfile:
    testData = csv.reader(csvfile, delimiter=',')
    newpriceList = []
    for row in testData:
        newpriceList.append(float(row[1]))

length = 4
coef = 0.5
weight = coef * np.ones([1, length+2])
posVec = tff.posVector(weight, length)
num = len(newpriceList)

def positionVector(length,n):
    posVector = np.zeros([1, n])
    for i in range(length, n):
        pos = 1
        # Restriction to control trading times:
        # Hold position for at least 5 bars
        if posVector[0][i-1] - posVector[0][i-2] == 0 and \
           posVector[0][i-2] - posVector[0][i-3] == 0 and \
           posVector[0][i-3] - posVector[0][i-4] == 0 and \
           posVector[0][i-4] - posVector[0][i-5] == 0 and \
           posVector[0][i-5] - posVector[0][i-6] == 0:
            holdEnough = 1
        else:
            holdEnough = 0
        # Time Control, Trading Period ;9:30 A.M - 2:30 P.M
        if (i+1) % 225 <= 30 or (i+1) % 225 >= 195:
            posVector[0][i] == 0
        # 0.9 is also a parameter, served as the signal significance
        elif abs(pos) > 0.9 and holdEnough == 1:
            posVector[0][i] = np.sign(pos)
        else:
            posVector[0][i] = posVector[0][i-1]
            
    print(len(posVector[0]))
    print(n)
    return posVector

positionVec = positionVector(length,num)
tradeSig = positionVec
sumsr, sumsrsq, total, tt = tff.sumReturn(tradeSig, tff.increList)
plt.plot(tradeSig.transpose())
plt.ylabel('Trading Signal')
plt.show()
print("Total Return: ", total[0][-1])
print("Trading Times: ", tt)
plt.plot(total.transpose())
plt.xlabel('Date')
plt.ylabel('Total Return')
plt.title('L')
plt.show()
