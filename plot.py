

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np




# parallel time ## using 64 core machine

x=[20,32,44,56,110]
y1=[0.95,1.08,1.21,1.34,1.99]
y2=[0.94,1.03,1.19,1.29,1.79]
y3=[0.92,0.99,1.17,1.28,1.65]


## ratio
y4=[1.01,1.05,1.02,1.04,1.11]
y5=[1.03,1.09,1.03,1.05,1.21]
plt.plot(x,y4,label="30% pruned")
plt.plot(x,y5,label="50% pruned")
plt.xlabel("Model Version")
plt.ylabel("Running time (ms) Ratio")
plt.title("Pruned Model Runtime Ratio on TPU")
plt.legend()
plt.savefig("final1.png")
plt.show()

# plt.plot(x,y1,label="Unpruned")
# plt.plot(x,y2,label="30% pruned")
# plt.plot(x,y3,label="50% pruned")
# plt.xlabel("Model Version")
# plt.ylabel("Running time (ms)")
# plt.title("Pruned Model Runtime on TPU")
# plt.legend()
# plt.savefig("final.png")
# plt.show()













