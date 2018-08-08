# Developed by Louis Langhardt and Mathias Overby

import tensorflow as tf
import matplotlib.pyplot as plt


def getIndices(j):

    return [i for i in range(len(j))]

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

        for i in range(2000):

            sess.run(train, feed_dict={x: x_values, y: y_values})

        a, b = sess.run([a, b])

        return float("%.2f" % a), float("%.2f" % b)

def applyFunc(a, b, x):

    return a * x + b

def splitList(indices, myList, a, b):
    overPrices = []
    overIndices = []
    underPrices = []
    underIndices = []

    for index in indices:

        price = myList[index]

        if price > applyFunc(a, b, index):
            overPrices.append(price)
            overIndices.append(index)
        else:
            underPrices.append(price)
            underIndices.append(index)

    return overPrices, overIndices, underPrices, underIndices

def showPlot(indices, prices, *args):
    plt.plot(indices, prices)
    plt.xlabel("time")
    plt.ylabel("price")
    for (a, b) in args:
        funcApplied = list(map((lambda x: a * x + b), indices))
        plt.plot(funcApplied)
    plt.show()


prices = [1, 3, 2, 3, 4, 2, 5, 4, 3]

# 1. regression analysis for all points
indices = getIndices(prices)
orgIndices = indices
a1, b1 = regressionAnalysis(indices, prices)
# showPlot(indices, prices, (a1, b1))
overPrices, overIndices, underPrices, underIndices = splitList(indices, prices, a1, b1)

# 2a. regression analysis for top points
# Depending on the volatility of the stock adjust how many times you do regression analysis
# Or change the b value to move the function down
# Or move the function down no matter what to make sure also to get the points a bit under the function(change b)
a2, b2 = regressionAnalysis(overIndices, overPrices)
print(a2, b2)
# showPlot(orgIndices, prices, (a1, b1), (a2, b2))

# Gets the max top points for the second function

topPoints = []

for index in overIndices:

    price = prices[index]

    tempPoints = []

    if price > applyFunc(a2, b2, index):
        tempPoints.append(index)
    elif not len(tempPoints) == 0:
        topPoints.append(max(tempPoints))
        tempPoints = []

print(topPoints)



