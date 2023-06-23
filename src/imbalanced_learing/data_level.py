import copy
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import OneSidedSelection

class data_level():
    def __init__(self, X, y, p, random_seed) -> None:
        self.X = X
        self.y = y
        self.data = {}
        self.p = p
        self.randoom_seed = random_seed
        np.random.seed(random_seed)


    def prepare_data(self, X, y):
        return np.array(X).copy(), np.array(y).copy()
    
    
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
        ROS_data = np.concatenate((X_oversampled, y_oversampled), axis = 1)
        np.random.shuffle(ROS_data)
        self.data['ROS_data'] = ROS_data

        return ROS_data
    

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

        RUS_data = np.concatenate((X_undersampled, y_undersampled), axis = 1)
        np.random.shuffle(RUS_data)
        self.data['RUS_data'] = RUS_data

        return RUS_data


    def SMOTE(self, p = 1.):
        X, y = self.prepare_data(self.X, self.y)
        sm = SMOTE(random_state=42, sampling_strategy = p) # type: ignore
        X_res, y_res = sm.fit_resample(X, y) # type: ignore
        y_res = np.array(y_res).reshape(-1,1)
        SMOTE_data = np.concatenate((X_res, y_res), axis = 1)
        np.random.shuffle(SMOTE_data)
        self.data['SMOTE_data'] = SMOTE_data

        return SMOTE_data

    def OSS(self):
        X, y = self.prepare_data(self.X, self.y)
        oss = OneSidedSelection(random_state=42)
        X_res, y_res = oss.fit_resample(X, y) # type: ignore
        y_res = np.array(y_res).reshape(-1,1)
        OSS_data = np.concatenate((X_res, y_res), axis = 1)
        np.random.shuffle(OSS_data)
        self.data['OSS_data'] = OSS_data

        return OSS_data