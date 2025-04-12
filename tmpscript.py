import csv
import json

def csv_to_json(csv_file_path, json_file_path):
    data = []

    with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            data.append(row)

    with open(json_file_path, mode='w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=2)

    print(f"Converted '{csv_file_path}' to '{json_file_path}' successfully.")

# Example usage

z=input()
csv_to_json(z+'.csv', z+'.json')