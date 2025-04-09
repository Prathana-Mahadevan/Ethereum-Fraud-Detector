import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline
data = {
    'Avg_min_between_sent_tnx': [0.0],
    'Avg_min_between_received_tnx': [0.0],
    'Time_Diff_between_first_and_last_Mins': [0.0],
    'Sent_tnx': [0],
    'Received_Tnx': [0],
    'Number_of_Created_Contracts': [0],
    'Unique_Received_From_Addresses': [0],
    'Unique_Sent_To_Addresses': [0],
    'min_value_received': [0.0],
    'max_value_received': [0.0],
    'avg_val_received': [0.0],
    'min_val_sent': [0.0],
    'max_val_sent': [0.0],
    'avg_val_sent': [0.0],
    'min_value_sent_to_contract': [0.0],
    'max_val_sent_to_contract': [0.0],
    'total_Ether_sent': [0.0],
    'total_ether_received': [0.0],
    'total_ether_sent_contracts': [0.0],
    'total_ether_balance': [0.0],
    'Total_ERC20_tnxs': [0],
    'ERC20_total_Ether_received': [0.0],
    'ERC20_total_ether_sent': [0.0],
    'ERC20_total_Ether_sent_contract': [0.0],
    'ERC20_uniq_sent_addr': [0],
    'ERC20_uniq_rec_addr': [0],
    'ERC20_uniq_sent_addr.1': [0],
    'ERC20_uniq_rec_contract_addr': [0],
    'ERC20_min_val_rec': [0.0],
    'ERC20_max_val_rec': [0.0],
    'ERC20_avg_val_rec': [0.0],
    'ERC20_min_val_sent': [0.0],
    'ERC20_max_val_sent': [0.0],
    'ERC20_avg_val_sent': [0.0],
    'ERC20_uniq_sent_token_name': [0.0],
    'ERC20_uniq_rec_token_name': [0.0],
    'ERC20_most_sent_token_type': ['No Token Usage'],
    'ERC20_most_rec_token_type': ['No Token Usage']
}

df = pd.DataFrame(data)

print(df)
print("Before Prediction")

predict_pipeline=PredictPipeline()
print("Mid Prediction")
results=predict_pipeline.predict(df)
print("after Prediction")
print(results)


columns = [
    'Avg_min_between_sent_tnx', 'Avg_min_between_received_tnx', 'Time_Diff_between_first_and_last_Mins', 'Sent_tnx', 'Received_Tnx', 'Number_of_Created_Contracts', 'Unique_Received_From_Addresses', 'Unique_Sent_To_Addresses',
    'min_value_received',
    'max_value_received',
    'avg_val_received',
    'min_val_sent',
    'max_val_sent',
    'avg_val_sent',
    'min_value_sent_to_contract',
    'max_val_sent_to_contract',
    'total_Ether_sent',
    'total_ether_received',
    'total_ether_sent_contracts',
    'total_ether_balance',
    'Total_ERC20_tnxs',
    'ERC20_total_Ether_received',
    'ERC20_total_ether_sent',
    'ERC20_total_Ether_sent_contract',
    'ERC20_uniq_sent_addr',
    'ERC20_uniq_rec_addr',
    'ERC20_uniq_sent_addr.1',
    'ERC20_uniq_rec_contract_addr',
    'ERC20_min_val_rec',
    'ERC20_max_val_rec',
    'ERC20_avg_val_rec',
    'ERC20_min_val_sent',
    'ERC20_max_val_sent',
    'ERC20_avg_val_sent',
    'ERC20_uniq_sent_token_name',
    'ERC20_uniq_rec_token_name',
    'ERC20_most_sent_token_type',
    'ERC20_most_rec_token_type'
]