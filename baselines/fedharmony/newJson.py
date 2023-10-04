import csv
import json

# Define the CSV file path
csv_file_path = '5320_ABIDE_Phenotypics_20230801.csv'

# Create a dictionary to store the anonymized ID and age data
age_dictionary = {}

# Open the CSV file and read the data
with open(csv_file_path, 'r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    next(csv_reader)

    for row in csv_reader:
        anonymized_id = row['Anonymized ID']
        age = float(row['AgeAtScan'])
        
        # Store the data in the dictionary
        age_dictionary[anonymized_id] = age

# Define the JSON file path
json_file_path = 'age_dictionary_ABIDE.json'

# Write the dictionary to a JSON file
with open(json_file_path, 'w') as json_file:
    json.dump(age_dictionary, json_file)

print(f"Data has been extracted and stored in '{json_file_path}'.")
