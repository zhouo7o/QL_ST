import csv
import numpy as np
import math
import matplotlib.pyplot as plt


#   Load Data
with open('ltrain.csv') as csvfile:
    trainData = csv.reader(csvfile, delimiter=',')
    priceList = []
    for row in trainData:
        priceList.append(float(row[1]))

#   Calculate the increment series
def increment(list):
    increList = []
    lengthList = len(list)
    increList.append(0)
    for i in range(1, lengthList):
        increList.append(list[i] - list[i-1])
    return increList

# increList = increment(priceList)
increList = [x * 5 for x in increment(priceList)]
# print(increList)

#   Calculate increment vector of length n
def incVector(length, index, list):
    incVector = list[index - length + 1: index + 1]
    return incVector

n = len(priceList)
#   Set the length of increment series we want to use
#   length = 10

#   Calculate the value of position
#   Formal parameter list here is (1, increVector)
def position(list, weight, prepos):
    weightList = np.append([1], np.append(np.array(list), [prepos]))
    weightList = weightList.reshape([len(list)+2, 1])
    pos = np.dot(weight, weightList)
    return np.tanh(pos), weightList

#   Set the initial weight
#   weight = 0.1 * np.ones([1, length+2])

#   Calculate the positions vector
def posVector(wt, length):
    posVec = np.zeros([1, n])
    for i in range(length, n):
        posList = incVector(length, i, increList)
        prepos = posVec[0][i-1]
        pos, wl = position(posList, wt, prepos)
        # Restriction to control trading times:
        # Hold position for at least 5 bars
        if posVec[0][i-1] - posVec[0][i-2] == 0 and \
           posVec[0][i-2] - posVec[0][i-3] == 0 and \
           posVec[0][i-3] - posVec[0][i-4] == 0 and \
           posVec[0][i-4] - posVec[0][i-5] == 0 and \
           posVec[0][i-5] - posVec[0][i-6] == 0:
            holdEnough = 1
        else:
            holdEnough = 0
        # Time Control, Trading Period ;9:30 A.M - 2:30 P.M
        if (i+1) % 225 <= 30 or (i+1) % 225 >= 195:
            posVec[0][i] == 0
        # 0.9 is also a parameter, served as the signal significance
        elif abs(pos) > 0.9 and holdEnough == 1:
            posVec[0][i] = np.sign(pos)
        else:
            posVec[0][i] = posVec[0][i-1]
            
    print(len(posVec[0]))
    print(n)
    return posVec

#   Calculate the sum of return
def sumReturn(pos, inc):
    sr = np.zeros([1, n])
    srsquire = np.zeros([1, n])
    srtotal = np.zeros([1, n])
    tt = 0
    for i in range(1, n):
        sr[0][i] = (pos[0][i-1] * inc[i] - com * abs(pos[0][i] - pos[0][i-1]))
        # Loss limit control: cover position when sum loss in 5 bar exceeds 200
        # This is also a parameter
        if (sr[0][i] + sr[0][i-1] + sr[0][i-1]) < -150:
            pos[0][i+1:i+5] = 0
        tt += 0.5 * abs(pos[0][i] - pos[0][i-1])
        srsquire[0][i] = sr[0][i] ** 2
        srtotal[0][i] = sr.sum()
    sumsr = sr.sum()
    sumsrsq = srsquire.sum()
    # print(sr)
    return sumsr, sumsrsq, srtotal, tt

# Calculate the sharp ratio
def sharpRatio(weight, length,posVec):
    sumsr, sumsrsq, total, tt = sumReturn(posVec, increList)
    A = sumsr / float(n)
    B = sumsrsq / float(n)
    sharpRatio = A / math.sqrt(B - (A**2))
    return sharpRatio, float(A), float(B)


#   Calculate the gradient of position vector
def posGradient(weight, length,posVec):
    posVec = posVector(weight, length)
    posGrad = np.zeros([n, length+2])
    for i in range(length, n):
        posList = incVector(length, i, increList)
        prepos = posVec[0][i-1]
        pos, wl = position(posList, weight, prepos)
        # print(np.shape(posGrad[i-1, ]))
        posGrad[i, ] = (1 - pos ** 2) * (wl.transpose() + posGrad[i-1, ] * weight[0, length+1])
    return posGrad

#   Calculate the gradient of Sharp Ratio
def sharpGradient(weight, length, posVec):
    pGrad = posGradient(weight, length,posVec)
    sRatio, A, B = sharpRatio(weight, length,posVec)
    dST_dA = (B - A**2) ** (-0.5) + (A ** 2) * (B - A**2) ** (-1.5)
    dST_dB = (-0.5) * A * ((B - A**2) ** (-1.5))
    dA_dRi = 1.0 / float(n)
    sharpGrad = np.zeros([n, length+2])
    for i in range(length, n):
        dB_dRi = 2 * dA_dRi * increList[i]
        dRt_dFt = - com * np.sign(posVec[0, i] - posVec[0, i-1])
        dRt_dFpret = (increList[i] + com * np.sign(posVec[0, i] - posVec[0, i-1]))
        sharpGrad[i, ] = (dST_dA * dA_dRi + dST_dB * dB_dRi) * \
                         (dRt_dFt * pGrad[i, ] + dRt_dFpret * pGrad[i-1, ])
    sGrad = sharpGrad.sum(axis=0)
    return sGrad

def learning(length, step):
    coef = float(input("Please enter the initial weight coefficient (0.01-1.00): "))
    speed = float(input("Please enter the learning Speed (0.1-5.0): "))
    weight = coef * np.ones([1, length+2])
    for i in range(1, step+1):
        posVec = posVector(weight, length)
        spratio, a, b = sharpRatio(weight, length, posVec)
        # if spratio >= 0.15 and i >= 300:
        #     break
        #if i % 10 == 0:
        print("Reinforcement Learning {0} th step, Sharp Ratio: {1}".format(i, spratio))
        spGrad = sharpGradient(weight, length, posVec)
        weight += max(0.01, speed * (0.99 ** i)) * spGrad
    return weight
# The weight returned is a new one updated in the last step
# So the sharp ratio is not the same as showed

com = 3    # Default Commission Rate
# This is actually larger because here we charged twice for a complete transaction
# in both building position and covering position
#weight = learning(14, 10)   # Example: train with 14 bars and 10 steps
#tradeSig = posVector(weight, 14)
#sumsr, sumsrsq, total, tt = sumReturn(tradeSig, increList)
#plt.plot(tradeSig.transpose())
#plt.ylabel('Trading Signal')
#plt.show()
#print("Total Return: ", total[0][-1])
#print("Trading Times: ", tt)
#plt.plot(total.transpose())
#plt.xlabel('Date')
#plt.ylabel('Total Return')
#plt.title('L')
#plt.show()


