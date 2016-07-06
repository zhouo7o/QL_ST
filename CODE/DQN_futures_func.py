import csv
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import os

def pause():
    programPause = raw_input("Press the <ENTER> key to continue...")

def priceGeneration(n,length):
	priceList = np.zeros(n)
	for i in range(0,n):
		priceList[i] = np.sin(i*math.pi/100) + 2 + 0.1*(random.random()-0.5)
	
	increList = increment(priceList)
	logincreList = logincrement(priceList)
	
	priceList[0] = 10
	for i in range(1, n):
		if random.random() >= 0.3:
			increList[i] = 1
		else:
			increList[i] = -1
		priceList[i] = priceList[i-1] + increList[i]
					
	plt.plot(priceList)
	plt.xlabel('Date')
	plt.ylabel('Price')
	plt.title(['# of bars = ', length])
	plt.show()
	return priceList, increList

#   Calculate the increment series
def increment(list):
    increList = []
    lengthList = len(list)
    increList.append(0)
    for i in range(1, lengthList):
        increList.append(list[i] - list[i-1])
    return increList

#   Calculate the increment series in log
def logincrement(list):
    increList = []
    lengthList = len(list)
    increList.append(0)
    for i in range(1, lengthList):
        increList.append(math.log(list[i] / list[i-1]))
    return increList

#############################################################
#increList = [x * 5 for x in increment(priceList)]
#############################################################

#   Calculate increment vector of length
def stateVector(length, index, list):
    stateVector = list[index - length + 1 : index + 1]
    indxQstate = 0
    for i in range(1 , length + 1):
		indxQstate += stateVector[i-1] * math.pow(2,i-1)
		
    indxQstate = int(indxQstate)
    return stateVector, indxQstate

def OsumReturn(pos, inc, n):
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

def sumReturn(pos, inc, n):
    com = 0.3
    sr = np.zeros(n)
    srsquire = np.zeros(n)
    srtotal = np.zeros(n)
    tt = 0
    for i in range(1, n):
        sr[i] = (pos[i-1] * inc[i] - com * abs(pos[i] - pos[i-1]))
        # Loss limit control: cover position when sum loss in 5 bar exceeds 200
        # This is also a parameter
        #if (sr[i] + sr[i-1] + sr[i-1]) < -150:
        #    pos[i+1:i+5] = 0
        #tt += 0.5 * abs(pos[i] - pos[i-1])
        srsquire[i] = sr[i] ** 2
        srtotal[i] = sr.sum()
    sumsr = sr.sum()
    sumsrsq = srsquire.sum()
    # print(sr)
    return sumsr, sumsrsq, srtotal

# Calculate the sharp ratio
def sharpeRatio(posVec, increList, n):
    sumsr, sumsrsq, total = sumReturn(posVec, increList, n)
    A = sumsr / float(n)
    B = sumsrsq / float(n)
    sharpeRatio = 0.0
    #print(A,B)
    if A*B != 0.0:
		sharpeRatio = float(A) / math.sqrt(B - (A**2))
    #print(sharpeRatio)
    return sharpeRatio

def actionSelection(Q, indxQs, positionVector, increList, j, n, i, step):
	sharpeSel = np.zeros(3)
	
	for k in range(0,3):
		positionVector[j] = k - 1
		sharpeSel[k] = sharpeRatio(positionVector, increList, n)
			
	epsilon = float((i-1)*n + j)/float(n*step)
	#print(epsilon)
	seed = np.random.rand(2)
	if seed[0] <= epsilon:
		action = np.argmax(sharpeSel) - 1
		#maxQ = np.argmax(Q, axis = 1) - 1
		#action = maxQ[indxQs]
	else:
		action = np.random.random_integers(-1,1)				
	
	# Hold position for at least 5 bars
	if positionVector[j-1] - positionVector[j-2] == 0 and \
	positionVector[j-2] - positionVector[j-3] == 0 and \
	positionVector[j-3] - positionVector[j-4] == 0 and \
	positionVector[j-4] - positionVector[j-5] == 0 and \
	positionVector[j-5] - positionVector[j-6] == 0:
		holdEnough = 1
	else:
		holdEnough = 0

	#holdEnough = 1
	#if (j+1) % 225 <= 30 or (j+1) % 225 >= 195:
	#	positionVector_j = 0
	if abs(action) > 0.5 and holdEnough == 1:
		positionVector_j = action
	else:
		positionVector_j = positionVector[j-1]
	return positionVector_j

def QactionSelection(QpositionVector, Q, indxQs, j):
	maxQ = np.argmax(Q, axis = 1) - 1
	action = maxQ[indxQs]
	
	if QpositionVector[j-1] - QpositionVector[j-2] == 0 and \
    QpositionVector[j-2] - QpositionVector[j-3] == 0 and \
    QpositionVector[j-3] - QpositionVector[j-4] == 0 and \
    QpositionVector[j-4] - QpositionVector[j-5] == 0 and \
    QpositionVector[j-5] - QpositionVector[j-6] == 0:
		holdEnough = 1
	else:
		holdEnough = 0

	#holdEnough = 1				
	#if (j+1) % 225 <= 30 or (j+1) % 225 >= 195:
	#	QpositionVector_j = 0
	if abs(action) > 0.5 and holdEnough == 1:
		QpositionVector_j = action
	else:
		QpositionVector_j = QpositionVector[j-1]
	return QpositionVector_j
