# Developed by Louis Langhardt and Mathias Overby

import tensorflow as tf
import matplotlib.pyplot as plt


# Prices input
prices = [[1, 3, 2, 3, 4, 2, 5, 4, 3], [8,7,6,7,6,5,6,5,4], [8,7,6,7,8,6,7,8,6],
         [13,14,16,10,8,9,12,6,9,4,7], [14,10,15,10,12,14,15,13,12,10,12,13,14.5,14],
         [3,6,4,7,6,4,6,8,6,7,9,6,7], [10,9,6,11,10,10,7,13,12,10,10,9,8,12,14,10]]


# HELPER FUNCTIONS
# Returns a list with len(j) items 0..len(j)-1
def getIndices(j):

    return ([i for i in range(len(j))])

# Gets the prices with indices as input
def getPrices(j, prices):
    return [prices[i] for i in j]

# Does a regression analysis
def regressionAnalysis(x_values, y_values):
    tf.reset_default_graph()

    y = tf.placeholder(dtype=tf.float32, shape=None)

    x = tf.placeholder(dtype=tf.float32, shape=None)
    a = tf.Variable(1.0, dtype=tf.float32)
    b = tf.Variable(1.0, dtype=tf.float32)

    func = a * x + b

    error = func - y
    squared_error = tf.square(error)
    loss = tf.reduce_mean(squared_error)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.005)
    train = optimizer.minimize(loss)

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        for i in range(3000):

            sess.run(train, feed_dict={x: x_values, y: y_values})

        a, b = sess.run([a, b])

        return float("%.2f" % a), float("%.2f" % b)

# Applies a linear funtion to a x value
def applyFunc(a, b, x):

    return a * x + b

# Splits a list in two, over and under a linear funtion (a, b)
def splitList(indices, myList, a, b):
    over = []
    under = []

    for index in indices:

        price = myList[index]

        if price > applyFunc(a, b, index):
            over.append(index)
        else:
            under.append(index)

    return over, under

# Displays the graph with args as (a, b) values
def showPlot(indices, prices, *args):
    plt.plot(indices, prices)
    plt.xlabel("time")
    plt.ylabel("price")
    for (a, b) in args:
        funcApplied = list(map((lambda x: a * x + b), indices))
        plt.plot(funcApplied)
    plt.show()



def middleLine(prices):
    # Regression analysis of the whole graph
    indices = getIndices(prices)
    a1, b1 = regressionAnalysis(indices, prices)
    # Splits the list in two, over and under the function  (indices)
    over, under = splitList(indices, prices, a1, b1)
    return indices, over, under

def resistanceLine(prices, over):

    # RESISTANCE LINE
    # The regression analysis for top points
    overPrices = getPrices(over, prices)
    a2, b2 = regressionAnalysis(over, overPrices)
    tempB = b2 - 0.25
    over, _ = splitList(over, prices, a2, tempB)


    # Gets the max top points for the second function
    topPoints = []
    tempPoints = []
    lastI = -1

    for i in over:
        if not len(tempPoints) == 0 and not lastI + 1 == i:
            topPoints.append(max(tempPoints))
            tempPoints = []

        price = prices[i]

        if price > applyFunc(a2, tempB, i):
            tempPoints.append((price, i))
        elif not len(tempPoints) == 0:
            topPoints.append(max(tempPoints))
            tempPoints = []

        if i == over[-1] and not len(tempPoints) == 0:
            topPoints.append(max(tempPoints))

        lastI = i

    # Splits the list into prices and indices
    spikesPrices = [i[0] for i in topPoints]
    spikesIndices = [i[1] for i in topPoints]

    # Resistance line
    a3, b3 = regressionAnalysis(spikesIndices, spikesPrices)

    return a3, b3

def supportLine(prices, under):
    # SUPPORT LINE
    # Creates the second regression analysis for support line
    underPrices = getPrices(under, prices)
    a4, b4 = regressionAnalysis(under, underPrices)
    tempB = b4 + 0.25
    _, under = splitList(under, prices, a4, tempB)

    # Gets all the lowest points
    lowPoints = []
    tempPoints = []
    lastI = -1
    for i in under:
        if not len(tempPoints) == 0 and not lastI + 1 == i:
            lowPoints.append(min(tempPoints))
            tempPoints = []

        price = prices[i]

        if price < applyFunc(a4, tempB, i):
            tempPoints.append((price,i))
        elif not len(tempPoints) == 0:
            lowPoints.append(min(tempPoints))
            tempPoints = []

        lastI = i
        if i == under[-1] and not len(tempPoints) == 0:
            lowPoints.append(min(tempPoints))

    # Splits the indices and prices up
    dropPrices = [i[0] for i in lowPoints]
    dropIndices = [i[1] for i in lowPoints]

    # Support line
    a5, b5 = regressionAnalysis(dropIndices, dropPrices)

    return a5, b5




# Calculates and displays the support and resistance lines
for pricesList in prices:
    allIndices, topIndices, buttonIndices = middleLine(pricesList)
    ra, rb = resistanceLine(pricesList, topIndices)
    sa, sb = supportLine(pricesList, buttonIndices)
    showPlot(allIndices, pricesList, (ra, rb), (sa, sb))

