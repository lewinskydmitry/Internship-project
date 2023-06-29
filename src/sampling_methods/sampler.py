import copy
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import OneSidedSelection

def copy_data(X, y):
        return np.array(X).copy(), np.array(y).copy()

class DataSampler():

    @staticmethod
    def ROS(X, y, p=1.):
        X, y = copy_data(X, y)

        class_counts = np.bincount(y)
        minority_class = np.argmin(class_counts)
        majority_class = np.argmax(class_counts)
        minority_indices = np.where(y == minority_class)[0]

        sampling_amount =  int(class_counts[majority_class] * p) - class_counts[minority_class]
        oversampled_indices = np.random.choice(minority_indices, size=sampling_amount, replace=True)

        X_oversampled = np.concatenate((X, X[oversampled_indices]), axis=0)
        y_oversampled = np.concatenate((y, y[oversampled_indices]), axis=0)
        y_oversampled = y_oversampled.reshape(-1,1)

        ROS_data = np.concatenate((X_oversampled, y_oversampled), axis = 1)
        np.random.shuffle(ROS_data)

        return ROS_data
    
    @staticmethod
    def RUS(X, y, p=1.):
        X, y = copy_data(X, y)

        class_counts = np.bincount(y)
        minority_class = np.argmin(class_counts)
        majority_class = np.argmax(class_counts)
        majority_indices = np.where(y != minority_class)[0]

        sampling_amount =  class_counts[majority_class] - int(class_counts[minority_class] / p)
        undersampled_indices = np.random.choice(majority_indices, size=sampling_amount, replace=False)

        X_undersampled = np.concatenate((X[~undersampled_indices], X[y == minority_class]), axis=0)
        y_undersampled = np.concatenate((y[~undersampled_indices], y[y == minority_class]), axis=0)
        y_undersampled = y_undersampled.reshape(-1,1)

        RUS_data = np.concatenate((X_undersampled, y_undersampled), axis = 1)
        np.random.shuffle(RUS_data)

        return RUS_data

    @staticmethod
    def SMOTE(X, y, p = 1.):
        X, y = copy_data(X, y)
        sm = SMOTE(random_state=42, sampling_strategy = p) # type: ignore
        X_res, y_res = sm.fit_resample(X, y) # type: ignore
        y_res = np.array(y_res).reshape(-1,1)
        SMOTE_data = np.concatenate((X_res, y_res), axis = 1)
        np.random.shuffle(SMOTE_data)

        return SMOTE_data

    @staticmethod
    def OSS(X, y):
        X, y = copy_data(X, y)
        oss = OneSidedSelection(random_state=42)
        X_res, y_res = oss.fit_resample(X, y) # type: ignore
        y_res = np.array(y_res).reshape(-1,1)
        OSS_data = np.concatenate((X_res, y_res), axis = 1)
        np.random.shuffle(OSS_data)

        return OSS_data
