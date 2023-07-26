import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from sklearn.preprocessing import StandardScaler
import joblib

random_seed = 42
torch.manual_seed(random_seed)
generator = torch.Generator()
generator.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# DEFINE BASELINE MODEL AND DATASET
class BaselineClassifier(nn.Module):
    def __init__(self, num_features, init_param, random_seed = 42):
        super(BaselineClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(num_features, init_param),
            nn.BatchNorm1d(init_param),
            nn.ReLU(),
            nn.Linear(init_param, init_param),
            nn.BatchNorm1d(init_param),
            nn.ReLU(),
            nn.Linear(init_param, init_param),
            nn.BatchNorm1d(init_param),
            nn.ReLU(),
            nn.Linear(init_param, int(init_param/2)),
            nn.BatchNorm1d(int(init_param/2)),
            nn.ReLU(),
            nn.Linear(int(init_param/2), int(init_param/4)),
            nn.BatchNorm1d(int(init_param/4)),
            nn.ReLU(),
            nn.Linear(int(init_param/4), int(init_param/8)),
            nn.BatchNorm1d(int(init_param/8)),
            nn.ReLU(),
            nn.Linear(int(init_param/8), int(init_param/16)),
            nn.BatchNorm1d(int(init_param/16)),
            nn.ReLU(),
            nn.Linear(int(init_param/16), int(init_param/32)),
            nn.BatchNorm1d(int(init_param/32)),
            nn.ReLU(),
            nn.Linear(int(init_param/32), int(init_param/64)),
            nn.BatchNorm1d(int(init_param/64)),
            nn.ReLU(),
            nn.Linear(int(init_param/64), int(init_param/64)),
            nn.BatchNorm1d(int(init_param/64)),
            nn.ReLU(),
            nn.Linear(int(init_param/64), 2)
        )
        self.random_seed = random_seed

        self._initialize_weights()

    def forward(self, x):
        x = self.classifier(x)
        x = F.softmax(x, dim=1)
        return x
    
    def _initialize_weights(self):
        for module in self.modules():
            torch.manual_seed(self.random_seed)
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    



# Loading data for testing
df_train = pd.read_csv('data/prepared_data.csv')
df_train = df_train.drop(columns = ['Machine failure'])
df_train = np.array(df_train)
# df_train = torch.tensor(df_train.values, dtype=torch.float32)
# df_train = df_train.to(device)


INIT_PARAM = 512 # Initial parameters 
features_amount = df_train.shape[1]

# Initialize model
model = BaselineClassifier(features_amount, INIT_PARAM) # Create model
model_weights_path = 'logs/classifiers/BL_512_1024_state_dict.pth'
model.load_state_dict(torch.load(model_weights_path)) # Load weights of the trained model
model = model.to(device)
model.eval()



# Functions for testing :
scaler = joblib.load('C:/Users/dimaf/Documents/GitHub/Internship-project/logs/classifiers/scaler.pkl') 

def get_data(df):
    """
    This function simulates that we take the one row of the data
    """
    idx = np.random.randint(0, len(df))
    one_row_data = df_train[idx,:].reshape(1,-1)
    one_row_data = scaler.transform(one_row_data)
    one_row_data = torch.tensor(one_row_data, dtype=torch.float32)
    one_row_data.to(device)
    return one_row_data


def return_pred(one_row_data):
    """
    This function takes one row of the data and return prediction
    """
    # Take output
    with torch.no_grad():
        result = model(one_row_data)
    return result

# testing result
one_row_data = get_data(df_train)
result = return_pred(one_row_data)
print(result)