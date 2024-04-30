import numpy

speed = [99,86,87,88,111,86,103,87,94,78,77,85,86]

x = numpy.mean(speed)
y = numpy.median(speed)
z = numpy.std(speed)
v = numpy.var(speed)
print("The mean is", x, "the median is ",y, "the standard deviation is", z, "The variabale is ",v)