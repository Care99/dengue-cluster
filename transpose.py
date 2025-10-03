import pandas as pd

# Path to the input Excel file
input_file = 'output.xlsx'

# Path to the output Excel file
output_file = 'resultado.xlsx'

# Create a new Excel writer object for the output file
excel_writer = pd.ExcelWriter(output_file, engine='openpyxl')

# Load the input Excel file
xls = pd.ExcelFile(input_file)

# Loop through each sheet and transpose the data
for sheet_name in xls.sheet_names:
    # Read the sheet into a DataFrame
    df = pd.read_excel(input_file, sheet_name=sheet_name)
    
    # Transpose the data
    transposed_df = df.T
    
    # Write the transposed data to a new sheet in the output file
    transposed_df.to_excel(excel_writer, sheet_name=sheet_name, index=False)

# Save the transposed data to the output file
excel_writer.close()

print("All sheets transposed and saved successfully!")
