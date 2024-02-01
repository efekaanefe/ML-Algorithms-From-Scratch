from sklearn.datasets import make_classification, make_regression


class DataGenerator:
    def __init__(self, num_samples, num_features, num_classes):
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
        

if __name__ == "__main__":
    num_samples = 50,
    num_features = 3
    num_classes = 2
    gen = DataGenerator(num_samples, num_features, num_classes)
    X, y = gen.get_data()