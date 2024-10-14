# Converts the string data in the 'Type' field into numerical values

import pandas as pd

def convert_and_save(input_file, output_file):
    df = pd.read_csv(input_file)
    
    # Define the mapping for the Target column
    target_mapping = {
        'No Diabetes': 0,
        'Prediabetic': 1,
        'Type 1 Diabetes': 2,
        'Type 2 Diabetes': 3,
        'Secondary Diabetes': 4,
        'Type 3c Diabetes (Pancreatogenic Diabetes)' : 5
    }
    
    # Convert the Target column using the mapping
    df['Target'] = df['Target'].map(target_mapping)
    
    # Save the modified dataframe to a new CSV file
    df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")

# Example usage
input_file_path = '../../datasets/main.csv'  
output_file_path = '../../datasets/data.csv' 
convert_and_save(input_file_path, output_file_path)

