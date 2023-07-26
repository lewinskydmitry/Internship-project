
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import argparse
import logging
import time
from  BaseExecutor import AbstractBaseExecutor

from sklearn.preprocessing import StandardScaler
import joblib


from numpy import asarray


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

  

class LiveDataSimultation() :

    def __init__(self,configuration) : 
        self.live_data = None 
        self.index = 0 
        self.excluded_columns=None

        csv_file = configuration.get("csv_file",None) 
        if csv_file is None :
            raise Exception(
                    "Invalid configuration,missing  'csv file'")   
        excluded_columns = configuration.get("excluded_column",[]) 
        self.live_data = pd.read_csv(csv_file)
        self.live_data = self.live_data.drop(columns = excluded_columns)
        return 
    
    def get_data(self) :
        """
        This function simulates that we take the one row of the data
        """
        self.index = self.index + 1 
        if self.index > (len(self.live_data)-1) :
             self.index = 0 
        row = self.live_data[self.index:self.index+1]
        return row



class Prediction(AbstractBaseExecutor) : 
    
    def __init__(self,configuration) : 
        super().__init__()
        self.model = None
        self.device = None
        self.columns_order = None
        self.columns_prediction = None
        # take a trained scaler
        self.scaler = joblib.load('../logs/classifiers/scaler.pkl') 


        path_weights = configuration.get("path_weights",None) 
        if path_weights == None :
                raise Exception(
                    "Invalid configuration,missing section 'path_weights'")

        self.service_name = configuration.get("name","predict_maint_dnn")
        self.columns_order = configuration.get("columns_order",[])
        self.columns_prediction = configuration.get("columns_prediction",[])
        
        random_seed = 42
        torch.manual_seed(random_seed)
        generator = torch.Generator()
        generator.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        features_amount = configuration.get("features_amount",None)
        init_param = configuration.get("initial_parameters")

        # Initialize model
        self.model = BaselineClassifier(features_amount, init_param) # Create model

        self.model.load_state_dict(torch.load(path_weights,map_location = torch.device("cpu"))) # Load weights of the trained model
        self.model = self.model.to(self.device)
        self.model.eval() 
 
        logging.debug("end of init Predictions")
        return 

    def run (self, df) :

        logging.debug("transformation dataframe to tensor_data")
        df = df.reindex(columns=self.columns_order)
        # Use the trained scaler
        df = self.scaler.transform(df)
        
        tensor_data = torch.tensor(df.values, dtype=torch.float32)
        tensor_data = tensor_data.to(self.device)
        logging.debug("tensor_data {}".format(tensor_data))
        #run a prediction
        with torch.no_grad():
            predictions = self.model(tensor_data)
            logging.debug("predictions :{}".format(predictions))      
        df = pd.DataFrame(predictions).astype("float")
        df.columns= self.columns_prediction
        # build output dataframe
        return df
    
   

def serve(simulation,prediction,sleep):
    # Start the server
    
    while True :
        one_row_data =simulation.get_data()
        logging.info("live data \n{}".format(one_row_data))
        result = prediction.run(one_row_data)
        logging.info("prediction \n{}".format(result))
        time.sleep(sleep)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Service Python",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c","--configuration",help="yaml configuration file",default="./configuration.yaml")
    parser.add_argument("-s","--sleep",help="execution period",default=1.0)
    parser.add_argument("-n","--name",help="service name",default="Deep Neural Network")
    parser.add_argument("-l","--loglevel",help="log level",default=logging.INFO)
    args = parser.parse_args()

    with open (args.configuration,'rt') as file : 
        configuration = yaml.safe_load(file.read())
    config_simulation = configuration.get("live_data_injestion",[])
    config_predictions = configuration.get("predictions",[])
    # Start the server.
    logging.basicConfig( encoding='utf-8', level=args.loglevel)
    serve(simulation=LiveDataSimultation(config_simulation),prediction=Prediction(config_predictions),sleep=args.sleep)