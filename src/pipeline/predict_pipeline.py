import sys
import os
import pandas as pd
from src.components.mlpro.exception import CustomException
from src.components.mlpro.utils import load_object
import mlflow
import joblib

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            # Data pre-processing can be done
            model = joblib.load('artifacts\model.pkl')
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            print("Before Loading")
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")
            print(features.head())
            data_scaled = preprocessor.transform(features)
           
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(  self,
        Address: str,
        Sent_tnx: int,
        Received_Tnx: int,                                 
        Number_of_Created_Contracts: int,
        Unique_Received_From_Addresses: int,
        Unique_Sent_To_Addresses: int,
        Avg_min_between_sent_tnx: float,
        Avg_min_between_received_tnx: float,
        Time_Diff_between_first_and_last_Mins: float,
        min_value_received: float,
        max_value_received: float,
        avg_val_received: float,
        min_val_sent: float,
        max_val_sent: float,
        avg_val_sent: float,
        min_value_sent_to_contract: float,
        max_val_sent_to_contract: float,
        total_Ether_sent: float,
        total_ether_received: float,
        total_ether_sent_contracts: float,
        total_ether_balance: float,
        Total_ERC20_tnxs: int,
        ERC20_total_Ether_received: float,
        ERC20_total_ether_sent: float,
        ERC20_total_Ether_sent_contract: float,
        ERC20_uniq_sent_addr: int,
        ERC20_uniq_rec_addr: int,
        ERC20_uniq_sent_addr_1: int,
        ERC20_uniq_rec_contract_addr: int,
        ERC20_min_val_rec: float,
        ERC20_max_val_rec: float,
        ERC20_avg_val_rec: float,
        ERC20_min_val_sent: float,
        ERC20_max_val_sent: float,
        ERC20_avg_val_sent: float,
        ERC20_uniq_sent_token_name: int,
        ERC20_uniq_rec_token_name: int,
        ERC20_most_sent_token_type: str,
        ERC20_most_rec_token_type: str):
    
        self.Address = Address
        self.Sent_tnx = Sent_tnx
        self.Received_Tnx = Received_Tnx
        self.Number_of_Created_Contracts = Number_of_Created_Contracts
        self.Unique_Received_From_Addresses = Unique_Received_From_Addresses
        self.Unique_Sent_To_Addresses = Unique_Sent_To_Addresses
        self.Avg_min_between_sent_tnx = Avg_min_between_sent_tnx
        self.Avg_min_between_received_tnx = Avg_min_between_received_tnx
        self.Time_Diff_between_first_and_last_Mins = Time_Diff_between_first_and_last_Mins
        self.min_value_received = min_value_received
        self.max_value_received = max_value_received
        self.avg_val_received = avg_val_received
        self.min_val_sent = min_val_sent
        self.max_val_sent = max_val_sent
        self.avg_val_sent = avg_val_sent
        self.min_value_sent_to_contract = min_value_sent_to_contract
        self.max_val_sent_to_contract = max_val_sent_to_contract
        self.total_Ether_sent = total_Ether_sent
        self.total_ether_received = total_ether_received
        self.total_ether_sent_contracts = total_ether_sent_contracts
        self.total_ether_balance = total_ether_balance
        self.Total_ERC20_tnxs = Total_ERC20_tnxs
        self.ERC20_total_Ether_received = ERC20_total_Ether_received
        self.ERC20_total_ether_sent = ERC20_total_ether_sent
        self.ERC20_total_Ether_sent_contract = ERC20_total_Ether_sent_contract
        self.ERC20_uniq_sent_addr = ERC20_uniq_sent_addr
        self.ERC20_uniq_rec_addr = ERC20_uniq_rec_addr
        self.ERC20_uniq_sent_addr_1 = ERC20_uniq_sent_addr_1
        self.ERC20_uniq_rec_contract_addr = ERC20_uniq_rec_contract_addr
        self.ERC20_min_val_rec = ERC20_min_val_rec
        self.ERC20_max_val_rec = ERC20_max_val_rec
        self.ERC20_avg_val_rec = ERC20_avg_val_rec
        self.ERC20_min_val_sent = ERC20_min_val_sent
        self.ERC20_max_val_sent = ERC20_max_val_sent
        self.ERC20_avg_val_sent = ERC20_avg_val_sent
        self.ERC20_uniq_sent_token_name = ERC20_uniq_sent_token_name
        self.ERC20_uniq_rec_token_name = ERC20_uniq_rec_token_name
        self.ERC20_most_sent_token_type = ERC20_most_sent_token_type
        self.ERC20_most_rec_token_type = ERC20_most_rec_token_type

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Address": [self.Address],
                "Sent_tnx": [self.Sent_tnx],
                "Received_Tnx": [self.Received_Tnx],
                "Number_of_Created_Contracts": [self.Number_of_Created_Contracts],
                "Unique_Received_From_Addresses": [self.Unique_Received_From_Addresses],
                "Unique_Sent_To_Addresses": [self.Unique_Sent_To_Addresses],
                "Avg_min_between_sent_tnx": [self.Avg_min_between_sent_tnx],
                "Avg_min_between_received_tnx": [self.Avg_min_between_received_tnx],
                "Time_Diff_between_first_and_last_(Mins)": [self.Time_Diff_between_first_and_last_Mins],
                "min_value_received": [self.min_value_received],
                "max_value_received": [self.max_value_received],
                "avg_val_received": [self.avg_val_received],
                "min_val_sent": [self.min_val_sent],
                "max_val_sent": [self.max_val_sent],
                "avg_val_sent": [self.avg_val_sent],
                "min_value_sent_to_contract": [self.min_value_sent_to_contract],
                "max_val_sent_to_contract": [self.max_val_sent_to_contract],
                "total_Ether_sent": [self.total_Ether_sent],
                "total_ether_received": [self.total_ether_received],
                "total_ether_sent_contracts": [self.total_ether_sent_contracts],
                "total_ether_balance": [self.total_ether_balance],
                "Total_ERC20_tnxs": [self.Total_ERC20_tnxs],
                "ERC20_total_Ether_received": [self.ERC20_total_Ether_received],
                "ERC20_total_ether_sent": [self.ERC20_total_ether_sent],
                "ERC20_total_Ether_sent_contract": [self.ERC20_total_Ether_sent_contract],
                "ERC20_uniq_sent_addr": [self.ERC20_uniq_sent_addr],
                "ERC20_uniq_rec_addr": [self.ERC20_uniq_rec_addr],
                "ERC20_uniq_sent_addr.1": [self.ERC20_uniq_sent_addr_1],
                "ERC20_uniq_rec_contract_addr": [self.ERC20_uniq_rec_contract_addr],
                "ERC20_min_val_rec": [self.ERC20_min_val_rec],
                "ERC20_max_val_rec": [self.ERC20_max_val_rec],
                "ERC20_avg_val_rec": [self.ERC20_avg_val_rec],
                "ERC20_min_val_sent": [self.ERC20_min_val_sent],
                "ERC20_max_val_sent": [self.ERC20_max_val_sent],
                "ERC20_avg_val_sent": [self.ERC20_avg_val_sent],
                "ERC20_uniq_sent_token_name": [self.ERC20_uniq_sent_token_name],
                "ERC20_uniq_rec_token_name": [self.ERC20_uniq_rec_token_name],
                "ERC20_most_sent_token_type": [self.ERC20_most_sent_token_type],
                "ERC20_most_rec_token_type": [self.ERC20_most_rec_token_type]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)
