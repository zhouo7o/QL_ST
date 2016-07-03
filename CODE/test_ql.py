import ql
import csv
import matplotlib.pyplot as plt

with open('AAPL_test.csv') as csvfile:
    testData = csv.reader(csvfile, delimiter=',')
    priceList = []
    for row in testData:
        priceList.append(float(row[1]))

# print(rrl2.increList)
weight = ql.qlearning(21, 1000)
ql.n = len(priceList)
ql.increList = ql.increment(priceList)
# rrl2.increList = [x * 5 for x in rrl2.increment(priceList)]
tradeSig = ql.posVector(weight, 21)
sumsr, sumsrsq, total = ql.sumReturn(tradeSig, ql.increList)
#plt.plot(tradeSig.transpose())
#plt.ylabel('Trading Signal')
#plt.show()
#print(total)
plt.plot(total.transpose())
plt.xlabel('Date')
plt.ylabel('Total Return')
plt.title('AAPL')
plt.show()
