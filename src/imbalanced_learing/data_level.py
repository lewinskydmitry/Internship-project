import pandas as pd
import numpy as np

class data_level():
    def __init__(self) -> None:
        pass

    def ROS(self, p=1., *data):
        if len(data) == 1:
            # Count the number of samples in each class
            df = data[0]
            class_counts = df['Machine failure'].value_counts()

            # Determine the minority and majority class
            minority_class = class_counts.idxmin()
            majority_class = class_counts.idxmax()

            # Upsample the minority class with replacement
            minority_df = df[df['Machine failure'] == minority_class]
            sampling_amount = int(class_counts[majority_class] - class_counts[minority_class]*p)
            upsampled_df = pd.concat( [df] + [minority_df.sample(n=sampling_amount, replace=True, random_state = 0)], axis=0)

            # Shuffle the upsampled data
            upsampled_df = upsampled_df.sample(frac=1, random_state = 0).reset_index(drop = True)
            return upsampled_df
        else:
            df = pd.concat([data[0], data[1]], axis=1)
            # Count the number of samples in each class
            class_counts = df['Machine failure'].value_counts()

            # Determine the minority and majority class
            minority_class = class_counts.idxmin()
            majority_class = class_counts.idxmax()

            # Upsample the minority class with replacement
            minority_df = df[df['Machine failure'] == minority_class]
            sampling_amount = int(class_counts[majority_class] - class_counts[minority_class]*p)
            upsampled_df = pd.concat( [df] + [minority_df.sample(n=sampling_amount, replace=True, random_state = 0)], axis=0)

            # Shuffle the upsampled data
            upsampled_df = upsampled_df.sample(frac=1, random_state = 0).reset_index(drop = True)
            return upsampled_df.drop(columns = ['Machine failure']), upsampled_df['Machine failure']
        

    def RUS(self, p=1., *data):
        if len(data) == 1:
            # Count the number of samples in each class
            df = data[0]
            class_counts = df['Machine failure'].value_counts()

            # Determine the minority and majority class
            minority_class = class_counts.idxmin()
            majority_class = class_counts.idxmax()

            sampling_amount = int(class_counts[majority_class] - class_counts[minority_class]*p)
            samples_to_remove = df[df['Machine failure'] == majority_class].sample(n=sampling_amount).index
            unsampled_df = df.drop(samples_to_remove)

            # Shuffle the upsampled data
            unsampled_df = unsampled_df.sample(frac=1, random_state = 0).reset_index(drop = True)

            return unsampled_df
        
        else:
            df = pd.concat([data[0], data[1]], axis=1)
            # Count the number of samples in each class
            class_counts = df['Machine failure'].value_counts()

            # Determine the minority and majority class
            minority_class = class_counts.idxmin()
            majority_class = class_counts.idxmax()

            # Unsample the minority class with replacement
            sampling_amount = int(class_counts[majority_class] - class_counts[minority_class]*p)
            samples_to_remove = df[df['Machine failure'] == majority_class].sample(n=sampling_amount).index
            unsampled_df = df.drop(samples_to_remove)

            # Shuffle the upsampled data
            unsampled_df = unsampled_df.sample(frac=1, random_state = 0).reset_index(drop = True)

            return unsampled_df.drop(columns = ['Machine failure']), unsampled_df['Machine failure']


