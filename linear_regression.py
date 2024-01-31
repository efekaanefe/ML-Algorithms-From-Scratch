import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)


# Parameters
num_points = 50  # Number of data points
slope_actual = 2         # Actual slope of the line
intercept_actual = 5     # Actual intercept of the line
noise_std = 2     # Standard deviation of the noise

x = np.linspace(0, 10, num_points)  # X
linear_data = slope_actual * x + intercept_actual

noise = np.random.normal(loc=0, scale=noise_std, size=num_points)  
data_with_noise = linear_data + noise  # y

plt.plot(x, linear_data, label='Linear Data', color='blue')
plt.scatter(x, data_with_noise, label='Data with Noise', color='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Data with Noise')
plt.legend()
plt.grid(True)
#plt.show()

# Linear Regression 2D: Feature # 1
m = 1  # slope
c = 0 # intercept

lr = 0.01
max_iter = 1000

for iter in range(max_iter):
    y_predict = m * x + c
    error = y_predict - data_with_noise
    mse = np.sum(error**2)/num_points # mean square error
    dm = (2/num_points) * np.sum(error * x)
    dc = (2/num_points) * np.sum(error)   
    m -= lr * dm
    c -= lr * dc

print(f"Actual:", slope_actual, intercept_actual)
print("Approximated: ", round(m,2), round(c,2))

# Linear Regression N-D: Feature # 2
# m = np.ones()
c = 0 

lr = 0.01
max_iter = 1000

for iter in range(max_iter):
    y_predict = m * x + c
    error = y_predict - data_with_noise
    mse = np.sum(error**2)/num_points 
    dm = (2/num_points) * np.sum(error * x)
    dc = (2/num_points) * np.sum(error)   
    m -= lr * dm
    c -= lr * dc

print(x.shape)