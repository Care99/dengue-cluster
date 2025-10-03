import os
import pandas as pd

# Directory containing your CSV files
csv_directory = '/home/cesar/Documents/dengue-cluster/Graficas'

# Create a new Excel writer object
excel_writer = pd.ExcelWriter('output.xlsx', engine='xlsxwriter')

# Loop through all CSV files in the directory
i=0
for csv_file in sorted(os.listdir(csv_directory)):
    if csv_file.endswith('.csv'):
        i=i+1
        # Read the CSV file into a DataFrame
        df = pd.read_csv(os.path.join(csv_directory, csv_file))
        
        # Remove the '.csv' extension to use as the sheet name
        sheet_name = f'{os.path.splitext(csv_file)[0][:5]}{i}'
        
        # Write the DataFrame to a sheet in the Excel file
        df.to_excel(excel_writer, sheet_name=sheet_name, index=False)

# Save the Excel file
excel_writer.close()

print("Excel file created successfully with a sheet for each CSV!")
