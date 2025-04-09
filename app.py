import json
from flask import Flask, jsonify, render_template, request
import pandas as pd
import psycopg2
import logging
import joblib

from src.pipeline.predict_pipeline import CustomData,PredictPipeline

app = Flask(__name__, template_folder="templates",static_folder="styles_folder")

# Function to connect to the PostgreSQL database
def connect_to_db():
    conn = psycopg2.connect(
        host='localhost',
        dbname='edf_test',
        user='postgres',
        port=5433,
        password='admin123'
    )
    return conn


def truncate_table():
    connection = connect_to_db()
    cursor = connection.cursor()
    cursor.execute("TRUNCATE TABLE public.test;")
    connection.commit()
    cursor.close()
    connection.close()

def insert_data_into_db():
    try:
        # Read data from Excel into a DataFrame
        df = pd.read_csv("artifacts\{}".format('test.csv'))
        df = df.drop(columns=['FLAG'])

        # Connect to the database
        connection = connect_to_db()
        cursor = connection.cursor()
    
        with connection.cursor() as cursor:
        # Iterate over the DataFrame rows and insert into the database
            for index, row in df.iterrows():
                # Replace 'public.test' with the name of your table
                # and 'column1', 'column2', ... with the column names in your table
                cursor.execute("""
                    INSERT INTO public.test ("Time_Diff_between_first_and_last_Mins", "Sent_tnx",
                                            "Received_Tnx", "Number_of_Created_Contracts", "Avg_min_between_sent_tnx",
                                            "Avg_min_between_received_tnx", "Unique_Received_From_Addresses",
                                            "Unique_Sent_To_Addresses", "min_value_received", "max_value_received",
                                            "avg_val_received", "min_val_sent", "max_val_sent", "avg_val_sent",
                                            "min_value_sent_to_contract", "max_val_sent_to_contract", "total_Ether_sent",
                                            "total_ether_received", "total_ether_sent_contracts", "total_ether_balance",
                                            "Total_ERC20_tnxs", "ERC20_total_Ether_received", "ERC20_total_ether_sent",
                                            "ERC20_total_Ether_sent_contract", "ERC20_uniq_sent_addr", "ERC20_uniq_rec_addr",
                                            "ERC20_uniq_sent_addr.1", "ERC20_uniq_rec_contract_addr", "ERC20_min_val_rec",
                                            "ERC20_max_val_rec", "ERC20_avg_val_rec", "ERC20_min_val_sent",
                                            "ERC20_max_val_sent", "ERC20_avg_val_sent", "ERC20_uniq_sent_token_name",
                                            "ERC20_uniq_rec_token_name", "ERC20_most_sent_token_type",
                                            "ERC20_most_rec_token_type")
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                """, (
                    row['Time_Diff_between_first_and_last_Mins'],
                    row['Sent_tnx'], row['Received_Tnx'], row['Number_of_Created_Contracts'],
                    row['Avg_min_between_sent_tnx'], row['Avg_min_between_received_tnx'],
                    row['Unique_Received_From_Addresses'], row['Unique_Sent_To_Addresses'],
                    row['min_value_received'], row['max_value_received'], row['avg_val_received'],
                    row['min_val_sent'], row['max_val_sent'], row['avg_val_sent'],
                    row['min_value_sent_to_contract'], row['max_val_sent_to_contract'],
                    row['total_Ether_sent'], row['total_ether_received'], row['total_ether_sent_contracts'],
                    row['total_ether_balance'], row['Total_ERC20_tnxs'], row['ERC20_total_Ether_received'],
                    row['ERC20_total_ether_sent'], row['ERC20_total_Ether_sent_contract'], row['ERC20_uniq_sent_addr'],
                    row['ERC20_uniq_rec_addr'], row['ERC20_uniq_sent_addr.1'], row['ERC20_uniq_rec_contract_addr'],
                    row['ERC20_min_val_rec'], row['ERC20_max_val_rec'], row['ERC20_avg_val_rec'],
                    row['ERC20_min_val_sent'], row['ERC20_max_val_sent'], row['ERC20_avg_val_sent'],
                    row['ERC20_uniq_sent_token_name'], row['ERC20_uniq_rec_token_name'],
                    row['ERC20_most_sent_token_type'], row['ERC20_most_rec_token_type']
                ))


            # Commit changes and close the cursor and connection
        connection.commit()
        cursor.close()
        connection.close()

        logging.info('Data inserted successfully!')
        print( 'Data inserted successfully!' )

        
    except Exception as e:
        logging.error(f'Error inserting data: {str(e)}')
        print( f'Error inserting data: {str(e)}' )



def predict_fraud(data):
    predict_pipeline=PredictPipeline()
    results=predict_pipeline.predict(data)
    # model = load_model()
    # print(model)
    # predictions = model.predict(data)
    # print(predictions)
    return results

# Route to fetch data from the database
@app.route('/fetch_data')
def fetch_data():
    try:
        connection = connect_to_db()
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM public.test;")
        result = cursor.fetchall()
        cursor.close()
        connection.close()

        # formatting the result
        final_lst = [list(row) for row in result]

        return jsonify(final_lst)
    except Exception as e:
        return jsonify(str(e))

# Route to predict fraud transactions
@app.route('/predict_data')
def predict_data():
    try:
        connection = connect_to_db()
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM public.test;")
        result = cursor.fetchall()
        cursor.close()
        connection.close()

        # formatting the result
        final_lst = [list(row) for row in result]

        print( final_lst )
        print("Hello I am in Predict_data")
        # data = json.loads(request.data)
        # print("Data loaded")
        data_df = pd.DataFrame(final_lst)
        print(data_df.head())
        data_df.columns = [
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
        # print(data_df)
        # predictions = predict_fraud(data_df)

        # data_df = pd.read_csv('artifacts\\test.csv')
        # data_df = data_df.drop( columns=['FLAG'], axis=1)
        print("data loaded")
        predictions = predict_fraud(data_df)
        print( "prediction: ", predictions )

        # # Highlighting fraud transactions
        # for idx, pred in enumerate(predictions):
        #     if pred == '1':  # '1' represents fraud transactions
        #         data_df.iloc[idx] = data_df.iloc[idx].apply(lambda x: f'<span style="background-color: red">{x}</span>')
        #         print(data_df.iloc[idx])


        # result = data_df.values.tolist()
        result = predictions.tolist()
        return jsonify(result)
    except Exception as e:
        print( e )
        return jsonify(str(e))

# Route to render the HTML page
@app.route('/')
def display_data():
    return render_template('homee.html')

if __name__ == '__main__':
    app.run(debug=True, port=5080)
    
#@app.route('/')
#def display_data():
#    return render_template('homee.html')
        




'''from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('home.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
       data = CustomData(
        Time_Diff_between_first_and_last_Mins=request.form.get('Time_Diff_between_first_and_last_Mins'),
        Sent_tnx=request.form.get('Sent_tnx'),
        Received_Tnx=request.form.get('Received_Tnx'),
        Number_of_Created_Contracts=request.form.get('Number_of_Created_Contracts'),
        Avg_min_between_sent_tnx=request.form.get('Avg_min_between_sent_tnx'),
        Avg_min_between_received_tnx=request.form.get('Avg_min_between_received_tnx'),
        Unique_Received_From_Addresses=request.form.get('Unique_Received_From_Addresses'),
        Unique_Sent_To_Addresses=request.form.get('Unique_Sent_To_Addresses'),
        min_value_received=request.form.get('min_value_received'),
        max_value_received=request.form.get('max_value_received'),
        avg_val_received=request.form.get('avg_val_received'),
        min_val_sent=request.form.get('min_val_sent'),
        max_val_sent=request.form.get('max_val_sent'),
        avg_val_sent=request.form.get('avg_val_sent'),
        min_value_sent_to_contract=request.form.get('min_value_sent_to_contract'),
        max_val_sent_to_contract=request.form.get('max_val_sent_to_contract'),
        total_Ether_sent=request.form.get('total_Ether_sent'),
        total_ether_received=request.form.get('total_ether_received'),
        total_ether_sent_contracts=request.form.get('total_ether_sent_contracts'),
        total_ether_balance=request.form.get('total_ether_balance'),
        Total_ERC20_tnxs=request.form.get('Total_ERC20_tnxs'),
        ERC20_total_Ether_received=request.form.get('ERC20_total_Ether_received'),
        ERC20_total_ether_sent=request.form.get('ERC20_total_ether_sent'),
        ERC20_total_Ether_sent_contract=request.form.get('ERC20_total_Ether_sent_contract'),
        ERC20_uniq_sent_addr=request.form.get('ERC20_uniq_sent_addr'),
        ERC20_uniq_rec_addr=request.form.get('ERC20_uniq_rec_addr'),
        ERC20_uniq_sent_addr_1=request.form.get('ERC20_uniq_sent_addr.1'),
        ERC20_uniq_rec_contract_addr=request.form.get('ERC20_uniq_rec_contract_addr'),
        ERC20_min_val_rec=request.form.get('ERC20_min_val_rec'),
        ERC20_max_val_rec=request.form.get('ERC20_max_val_rec'),
        ERC20_avg_val_rec=request.form.get('ERC20_avg_val_rec'),
        ERC20_min_val_sent=request.form.get('ERC20_min_val_sent'),
        ERC20_max_val_sent=request.form.get('ERC20_max_val_sent'),
        ERC20_avg_val_sent=request.form.get('ERC20_avg_val_sent'),
        ERC20_uniq_sent_token_name=request.form.get('ERC20_uniq_sent_token_name'),
        ERC20_uniq_rec_token_name=request.form.get('ERC20_uniq_rec_token_name'),
        ERC20_most_sent_token_type=request.form.get('ERC20_most_sent_token_type'),
        ERC20_most_rec_token_type=request.form.get('ERC20_most_rec_token_type')
    )

    pred_df=data.get_data_as_data_frame()
    print(pred_df)
    print("Before Prediction")

    predict_pipeline=PredictPipeline()
    print("Mid Prediction")
    results=predict_pipeline.predict(pred_df)
    print("after Prediction")
    return render_template('home.html',results=results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0")'''


