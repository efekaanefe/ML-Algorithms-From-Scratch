from sklearn.datasets import make_classification, make_regression
import matplotlib.pyplot as plt
import numpy as np

class DataGenerator:
    def __init__(self, num_samples, num_features, num_classes=None):
        self.num_samples = num_samples
        self.num_features = num_features
        self.num_classes = num_classes

    def get_data(self, type = "classification"):
        if type == "classification":
            X, y = make_classification(n_samples=self.num_samples, 
                                       n_features=self.num_features, 
                                       n_classes=self.num_classes, 
                                       n_clusters_per_class=1, 
                                       n_redundant=0, 
                                       n_informative=2, 
                                       random_state=42)
        if type == "regression":
            X, y = make_regression(n_samples=self.num_samples, 
                                       n_features=self.num_features, 
                                       noise = 10,
                                       random_state=42)
 
        return X, y
        
    def get_radial_data(self, num_samples_per_circle = 100, num_circles = 2, radius_range = (5, 8), noise=0.1):
        X = []
        y = []
        for i in range(num_circles):
            radius = np.random.uniform(*radius_range)
            theta = np.linspace(0, 2 * np.pi, num_samples_per_circle)
            x_circle = radius * np.cos(theta) + np.random.normal(0, noise, num_samples_per_circle)
            y_circle = radius * np.sin(theta) + np.random.normal(0, noise, num_samples_per_circle)
            X.extend(np.column_stack((x_circle, y_circle)))
            y.extend([i] * num_samples_per_circle)
        return np.array(X), np.array(y)


class DataPlotter:
    def __init__(self):
        pass

    def plot_data(self, X, y, type = "classification", num_classes = 3): 
        if type == "classification":
            # works for num_features = 2, and num_class < 3 
            plt.figure(figsize=(8, 6))
            colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan']
            for i in range(num_classes):
                plt.scatter(X[y == i][:, 0], X[y == i][:, 1], color=colors[i], label=f'Class {i}')
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.title('Classification Data')
            plt.legend()
            plt.grid(True)
            plt.show()

        if type == "regression":
            plt.figure(figsize=(8, 6))
            plt.plot(y, color='blue', label='Data points')
            plt.xlabel('Index')
            plt.ylabel('y')
            plt.title('Regression Data')
            plt.legend()
            plt.grid(True)
            plt.show()
            

def plot_decision_boundary_numerically(predict_func, X, y, title = "decision boundary", is_plotting_data = True):
    # Create meshgrid for visualization
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    
    # Predict classes for each point in meshgrid
    Z = np.c_[xx.ravel(), yy.ravel()]
    Z = predict_func(Z)
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary
    plt.contourf(xx, yy, Z, alpha=0.2)
    if is_plotting_data:
        plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()
    

if __name__ == "__main__":
    ## hyper parameters
    num_samples = 50
    num_features = 3
    num_classes = 2
    #type = "regression"
    type = "classification"
    
    gen = DataGenerator(num_samples, num_features, num_classes)
    X, y = gen.get_data(type)
    
    plotter = DataPlotter()
    plotter.plot_data(X,y, type = type)
