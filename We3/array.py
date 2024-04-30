import numpy
import matplotlib.pyplot as plt

x = numpy.random.uniform(0.0, 5.0, 250000)


y = numpy.random.uniform(5.0, 1.0, 250000)

plt.hist(x, 5)
plt.show()

plt.hist(y, 100)
plt.show()