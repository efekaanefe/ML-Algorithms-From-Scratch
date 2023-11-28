import matplotlib.pyplot as plt
from math import sqrt

data = {
    "r":[(-1,3),(-2,2),(-3,1), (0, 1.5)],
    "b":[(2,1),(1,2),(3,0),(0, -0.5)]
}

# # plot the data
# for k,v in data.items():
#     for x1, x2 in v:
#         plt.scatter(x1,x2,c=k)

# plt.show()

# linearlize
X = []
y = []
for k,v in data.items():
   for X_ in v:
      X.append(X_)
      y.append(k)

# print(X, y)    

x_predict = (0,0) # features to be predicted
k = 1 # k of kNN
distances = [] # [(distance, index), ...]
for i, x_i in enumerate(X):
    distance = sqrt((x_predict[0]-x_i[0])**2+(x_predict[1]-x_i[1])**2)
    distances.append((distance, i))

print(distances)


