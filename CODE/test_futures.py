import rrl2_futures
import csv
import matplotlib.pyplot as plt

with open('ltest.csv') as csvfile:
    testData = csv.reader(csvfile, delimiter=',')
    newpriceList = []
    for row in testData:
        newpriceList.append(float(row[1]))


# print(rrl2.increList)
weight = rrl2_futures.learning(14, 100) # Test with 14 bars and 10 steps
rrl2_futures.n = len(newpriceList)
# rrl2_futures.increList = rrl2_futures.increment(newpriceList)
rrl2_futures.increList = [x * 5 for x in rrl2_futures.increment(newpriceList)]
#print(rr2_futures.increList)
tradeSig = rrl2_futures.posVector(weight, 14)
sumsr, sumsrsq, total, tt = rrl2_futures.sumReturn(tradeSig, rrl2_futures.increList)
#plt.plot(newpriceList)
#plt.ylabel('price')
#plt.show()
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
