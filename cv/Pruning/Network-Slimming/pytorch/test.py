import matplotlib.pyplot as plt
import math
x=[]
lr=0.010
for epoch in range(0,100):
    x.append(lr*((epoch+1) / 5) if (epoch+1) <= 5 else lr*0.5 * (
                    math.cos((epoch - 5) / (101 - 5) * math.pi) + 1))
print(x)
m=[x for x in range(100)]
t=x[-1]
print(t)
plt.plot(m,x)
plt.show()