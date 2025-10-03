import csv
import re

# Input data string
with open("knn_time_series_sarima.txt","rt") as file_wrapper:
    data_string = file_wrapper.read()
# Use regex to find all lists
rows = re.findall(r'\[.*?\]', data_string)
# Process each row and write to a separate CSV file
for i, row in enumerate(rows):
    # Convert the row to a list of integers
    row_data = eval(row)
    
    # Create the CSV file name
    output_file = f'knn_time_series_sarima{i+1}.csv'
    
    # Write the row to the CSV file
    with open(output_file, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(row_data)

print("CSV files created successfully!")
