import csv
import numpy as np
import math
import matplotlib.pyplot as plt

#   Load Data
with open('../DATA/ltest.csv') as csvfile:
    trainData = csv.reader(csvfile, delimiter=',')
    priceList = []
    for row in trainData:
        priceList.append(float(row[1]))
n = len(priceList)

n = 1000
priceList = np.zeros(n)

for i in range(0,n):
	priceList[i] = np.sin(i*math.pi/100) + 2

priceList = priceList * 1
plt.plot(priceList)
plt.xlabel('Date')
plt.ylabel('Total Return')
plt.title('L')
plt.show()

#   Calculate the increment series
def increment(list):
    increList = []
    lengthList = len(list)
    increList.append(0)
    for i in range(1, lengthList):
        increList.append(list[i] - list[i-1])
    return increList

increList = increment(priceList)

#   Calculate the increment series in log
def logincrement(list):
    increList = []
    lengthList = len(list)
    increList.append(0)
    for i in range(1, lengthList):
        increList.append(math.log(list[i] / list[i-1]))
    return increList

logincreList = logincrement(priceList)

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

com = 0.3

# Calculate the sharp ratio
def sharpeRatio(posVec):
    sumsr, sumsrsq, total = sumReturn(posVec, increList)
    A = sumsr / float(n)
    B = sumsrsq / float(n)
    sharpeRatio = 0.0
    #print(A,B)
    if A*B != 0.0:
		sharpeRatio = float(A) / math.sqrt(B - (A**2))
    #print(sharpeRatio)
    return sharpeRatio

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
    return posVec

def actionSelection(positionVector, j):
	sharpeSel = np.zeros(3)
	action = 0
	for k in range(0,3):
		positionVector[j] = k - 1
		sharpeSel[k] = sharpeRatio(positionVector)
			
	epsilon = 0.9
	seed = np.random.rand(2)
	if seed[0] <= epsilon:
		action = np.argmax(sharpeSel) - 1
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

	if (j+1) % 225 <= 30 or (j+1) % 225 >= 195:
		positionVector_j = 0
	# 0.9 is also a parameter, served as the signal significance
	#elif abs(pos) > 0.9 and holdEnough == 1:
		#positionVector_j = np.sign(pos)
	#else:
		#positionVector_j = positionVector[j-1]
	elif abs(action) > 0.5 and holdEnough == 1:
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
				
	if (j+1) % 225 <= 30 or (j+1) % 225 >= 195:
		QpositionVector_j = 0
	elif abs(action) > 0.5 and holdEnough == 1:
		QpositionVector_j = action
	else:
		QpositionVector_j = QpositionVector[j-1]
	return QpositionVector_j

def algo(length,step):
    Q = np.zeros([int(math.pow(2,length)), 3]) #{STATE: # of length of bar, Action: {-1, 0, 1}}
    Q_old = Q
    speed = 0.01; #alpha learning rate
    sharpRatio = np.zeros(n)
    gamma = 0.5  #discounted future reward rate: immediate vs. future
    sharpe = np.zeros(n)
    priceState = np.floor(np.sign(increList)/2) + 1
    reward = 0.0
    QpositionVector = np.zeros(n)
    positionVector = np.zeros(n)
    for i in range(1, step + 1):
        print('episode = ', i)
        print(np.argmax(Q, axis = 1) - 1)
        #print(Q)
        #print(Q_old)
        print('Residual =',np.sum(Q*1.0 - Q_old)/(int(math.pow(2,length))))
        Q_old = Q*1.0
        ##########################################################################
        sumsr, sumsrsq, total = sumReturn(positionVector, increList)
        print('Optimized Profit =', sumsr)
        sumsr, sumsrsq, total = sumReturn(QpositionVector, increList)
        print('Q-Sharpe Profit =', sumsr)        
        QpositionVector = np.zeros(n)
        positionVector = np.zeros(n)
        ##########################################################################
        for j in range(length, n-1):
			#print(j)
			#########################################state & action current
			stateVec, indxQs = stateVector(length, j, priceState)
			#print(stateVec, indxQs)
			#action option - 1
			#action = np.random.random_integers(-1,1)
			#positionVector[j] = action
			#action option - 2
			positionVector[j] = actionSelection(positionVector, j)			
			#########################################reward
			sharpe[j] = sharpeRatio(positionVector)
			#print(j, sharpe[j])
			reward = sharpe[j] - sharpe[j - 1]
			########################################next state
			nstateVec, nindxQs = stateVector(length, j+1, priceState)
			Qmax = np.max(Q[nindxQs,])
			########################################
			indxQa = int(positionVector[j] + 1)
			Q[indxQs, indxQa] += speed*(reward + gamma*Qmax - Q[indxQs, indxQa])
			#print(indxQs)
			QpositionVector[j] = QactionSelection(QpositionVector, Q_old, indxQs, j)
    print(step*(n-length),np.sum(Q))
    print(Q)
    print(np.argmax(Q, axis = 1) - 1)
    sumsr, sumsrsq, total = sumReturn(positionVector, increList)
    #print(QpositionVector)
    plt.plot(total.transpose())
    plt.xlabel('Date')
    plt.ylabel('Total Return')
    plt.title('L')
    plt.show()
    return Q

def test(length,Q):
    nn = 1000
    TESTpriceList = np.zeros(nn)
    
    for i in range(0,nn):
		TESTpriceList[i] = math.cos(i*math.pi/100)*math.cos(i*math.pi/100) + 2
    
    TESTpriceList = TESTpriceList * 1
    plt.plot(TESTpriceList)
    plt.xlabel('Date')
    plt.ylabel('Total Return')
    plt.title('L')
    plt.show()
    QpositionVector = np.zeros(nn)
    QincreList = increment(TESTpriceList)
    priceState = np.floor(np.sign(QincreList)/2) + 1
    
    for j in range(length, nn-1):
		stateVec, indxQs = stateVector(length, j, priceState)
		QpositionVector[j] = QactionSelection(QpositionVector, Q, indxQs, j)

    #print(QpositionVector)
    sumsr, sumsrsq, total = sumReturn(QpositionVector, QincreList)
    print('Profit =', sumsr)
    print(n, len(total))
    plt.plot(total.transpose())
    plt.xlabel('Date')
    plt.ylabel('Total Return')
    plt.title('L')
    plt.show()
    return sumsr

