import pandas as pd  
import os  
  
print('Checking if data exists...')  
data_path = 'data/filtered_complaints.csv'  
print(f'Looking for file at: {data_path}')  
print(f'Current working directory: {os.getcwd()}')  
print(f'File exists: {os.path.exists(data_path)}')  
if os.path.exists(data_path):  
    print('Data file exists!' )  
    try:  
        df = pd.read_csv(data_path)  
        print(f'Shape: {df.shape}')  
        print(f'Columns: {list(df.columns)}')  
        print(f'Products: {df["Product"].unique()}')  
    except Exception as e:  
        print(f'Error reading CSV: {e}')  
else:  
    print('Data file does not exist') 
