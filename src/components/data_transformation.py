import sys
from dataclasses import dataclass
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler,OrdinalEncoder

from mlpro.exception import CustomException
from mlpro.logger import logging
import os

from mlpro.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function si responsible for data transformation
        
        '''
        try:
            logging.info('Data Transformation initiated')
            numerical_features = ['Avg_min_between_sent_tnx', 'Avg_min_between_received_tnx', 'Time_Diff_between_first_and_last_Mins', 'Sent_tnx', 'Received_Tnx', 'Number_of_Created_Contracts', 'Unique_Received_From_Addresses', 'Unique_Sent_To_Addresses', 'min_value_received', 'max_value_received', 'avg_val_received', 'min_val_sent', 'max_val_sent', 'avg_val_sent', 'min_value_sent_to_contract', 'max_val_sent_to_contract', 'total_Ether_sent', 'total_ether_received', 'total_ether_sent_contracts', 'total_ether_balance', 'Total_ERC20_tnxs', 'ERC20_total_Ether_received', 'ERC20_total_ether_sent', 'ERC20_total_Ether_sent_contract', 'ERC20_uniq_sent_addr', 'ERC20_uniq_rec_addr', 'ERC20_uniq_sent_addr.1', 'ERC20_uniq_rec_contract_addr', 'ERC20_min_val_rec', 'ERC20_max_val_rec', 'ERC20_avg_val_rec', 'ERC20_min_val_sent', 'ERC20_max_val_sent', 'ERC20_avg_val_sent', 'ERC20_uniq_sent_token_name', 'ERC20_uniq_rec_token_name']
            categorical_features = ['ERC20_most_sent_token_type', 'ERC20_most_rec_token_type']
            
            num_pipeline=Pipeline(
            steps=[
                
                ('imputer',SimpleImputer()),
                ('scaler', StandardScaler())
            ])
            cat_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder',OrdinalEncoder()),
                ('scaler',StandardScaler())
                ]

            )        

            logging.info(f"Categorical columns: {categorical_features}")
            logging.info(f"Numerical columns: {numerical_features}")

            preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipeline,numerical_features),
            ('cat_pipeline',cat_pipeline,categorical_features)
            ])
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")
            #PREPROCESSING CAN BE DONE

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name= 'FLAG'
            numerical_features = ['Avg_min_between_sent_tnx', 'Avg_min_between_received_tnx', 'Time_Diff_between_first_and_last_Mins', 'Sent_tnx', 'Received_Tnx', 'Number_of_Created_Contracts', 'Unique_Received_From_Addresses', 'Unique_Sent_To_Addresses', 'min_value_received', 'max_value_received', 'avg_val_received', 'min_val_sent', 'max_val_sent', 'avg_val_sent', 'min_value_sent_to_contract', 'max_val_sent_to_contract', 'total_Ether_sent', 'total_ether_received', 'total_ether_sent_contracts', 'total_ether_balance', 'Total_ERC20_tnxs', 'ERC20_total_Ether_received', 'ERC20_total_ether_sent', 'ERC20_total_Ether_sent_contract', 'ERC20_uniq_sent_addr', 'ERC20_uniq_rec_addr', 'ERC20_uniq_sent_addr.1', 'ERC20_uniq_rec_contract_addr', 'ERC20_min_val_rec', 'ERC20_max_val_rec', 'ERC20_avg_val_rec', 'ERC20_min_val_sent', 'ERC20_max_val_sent', 'ERC20_avg_val_sent', 'ERC20_uniq_sent_token_name', 'ERC20_uniq_rec_token_name']
            categorical_features = ['ERC20_most_sent_token_type', 'ERC20_most_rec_token_type']

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            print(input_feature_train_df)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
