import matplotlib.pyplot as plt
from math import sqrt

data = {
    "r":[(-1,3),(-2,2),(-3,1), (0, 1.5)],
    "b":[(2,1),(1,2),(3,0),(0, -0.5)]
}
x_predict = (0,0) # features to be predicted

# TODO: try it with n labeled data
# TODO: if all the x_predict is in the middle of neighbours, it shouldn't decide 

# # plot the data
# for k,v in data.items():
#     for x1, x2 in v:
#         plt.scatter(x1,x2,c=k)
# plt.scatter(x_predict[0],x_predict[1], c="g")
# plt.show()

# linearlize
X = []
y = []
for k,v in data.items():
   for X_ in v:
      X.append(X_)
      y.append(k)
# print(X, y)    

k = 3 # k of kNN
distances = [] # [(distance, index), ...]
for i, x_i in enumerate(X):
    distance = sqrt((x_predict[0]-x_i[0])**2+(x_predict[1]-x_i[1])**2)
    distances.append((distance, i))
# print(distances)

distances_ordered = sorted(distances, key=lambda x: x[0]) # increasing by distance
# print(distances_ordered)

closest_k_nearest_neighbour_indeces = []
for i in range(k):
   _, index = distances_ordered[i]
   closest_k_nearest_neighbour_indeces.append(index)
# print(closest_k_nearest_neighbour_indeces)

closest_k_nearest_neighbours = []
for index in closest_k_nearest_neighbour_indeces:
   closest_k_nearest_neighbours.append(y[index])
print(closest_k_nearest_neighbours)

predicted_label = max(set(closest_k_nearest_neighbours), key=closest_k_nearest_neighbours.count)
print(f"I predict ({x_predict[0]}, {x_predict[1]}) as: {predicted_label}")

for k,v in data.items():
    for x1, x2 in v:
        plt.scatter(x1,x2,c=k)
plt.scatter(x_predict[0],x_predict[1], c=predicted_label, marker="x")
plt.show()
