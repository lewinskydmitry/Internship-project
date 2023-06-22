import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

class data_level():
    def __init__(self, X, y) -> None:
        self.X = X
        self.y = y
        self.ROS_data = None
        self.RUS_data = None

    def prepare_data(self, *data):
        if len(data) == 1:
            # Count the number of samples in each class
            df = data[0]
            X = df[:,:-1]
            y = df[:,-1]
        else:
            X = data[0]
            y = data[1]

        return np.array(X), np.array(y)
    
    def ROS(self, p=1.):
        X, y = self.prepare_data(self.X, self.y)

        class_counts = np.bincount(y)
        minority_class = np.argmin(class_counts)
        majority_class = np.argmax(class_counts)
        minority_indices = np.where(y == minority_class)[0]

        sampling_amount =  int(class_counts[majority_class] * p) - class_counts[minority_class]
        minority_indices = np.where(y == minority_class)[0]
        oversampled_indices = np.random.choice(minority_indices, size=sampling_amount, replace=True)

        X_oversampled = np.concatenate((X, X[oversampled_indices]), axis=0)
        y_oversampled = np.concatenate((y, y[oversampled_indices]), axis=0)

        return X_oversampled, y_oversampled
    

    def RUS(self, p=0.5):
        X, y = self.prepare_data(self.X, self.y)

        class_counts = np.bincount(y)
        minority_class = np.argmin(class_counts)
        majority_class = np.argmax(class_counts)
        majority_indices = np.where(y != minority_class)[0]

        sampling_amount =  class_counts[majority_class] - int(class_counts[minority_class] / p)

        undersampled_indices = np.random.choice(majority_indices, size=sampling_amount, replace=False)

        X_undersampled = np.concatenate((X[~undersampled_indices], X[y == minority_class]), axis=0)
        y_undersampled = np.concatenate((y[~undersampled_indices], y[y == minority_class]), axis=0)

        return X_undersampled, y_undersampled


    def SMOTE(self, p=1., k=4, *data):
        # Convert input to numpy arrays if needed
        X = np.array(X)
        y = np.array(y)

        # Count the number of samples in each class
        class_counts = np.bincount(y)
        minority_class = np.argmin(class_counts)
        majority_class = np.argmax(class_counts)

        # Calculate the desired number of synthetic samples
        target_count = int(class_counts[majority_class] * p) - class_counts[minority_class]

        # Initialize arrays to store synthetic samples
        synthetic_samples = np.zeros((target_count, X.shape[1]))

        # Find k nearest neighbors for each minority class sample
        nn = NearestNeighbors(n_neighbors=k+1)
        nn.fit(X[y == minority_class])
        indices = nn.kneighbors(return_distance=False)

        # Generate synthetic samples
        for i in range(target_count):
            # Randomly choose a minority class sample
            idx = np.random.choice(len(indices))
            sample = X[y == minority_class][idx]

            # Randomly choose one of its k nearest neighbors
            nn_idx = np.random.choice(indices[idx])

            # Calculate the difference between the sample and its neighbor
            diff = X[nn_idx] - sample

            # Generate a synthetic sample
            synthetic_samples[i] = sample + np.random.random() * diff

        # Combine original and synthetic samples
        X_resampled = np.vstack((X, synthetic_samples))
        y_resampled = np.concatenate((y, [minority_class] * target_count))

        return X_resampled, y_resampled


