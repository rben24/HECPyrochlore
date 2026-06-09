import json
import csv
import sys
from pathlib import Path

_HERE    = Path(__file__).resolve().parent
_PROJECT = _HERE.parent.parent
DATA_DIR  = _PROJECT / 'data' / 'raw'
INPUT_FILE = DATA_DIR / 'aflow_pyrochlore_data_extra.json'
OUTPUT_FILE = DATA_DIR / 'aflow_pyrochlore_data_extra.csv'

def json_to_csv(json_file, csv_file):
    """
    Convert JSON data to CSV format.

    Args:
        json_file (str): Path to the input JSON file
        csv_file (str): Path to the output CSV file
    """
    try:
        # Read the JSON file
        with open(json_file, 'r') as f:
            data = json.load(f)

        # Check if data is a list
        if not isinstance(data, list):
            print("Error: JSON data must be a list of objects")
            return False

        if len(data) == 0:
            print("Error: JSON data is empty")
            return False

        # Get all unique keys from all objects
        all_keys = set()
        for item in data:
            all_keys.update(item.keys())

        # Sort keys for consistent column order
        fieldnames = sorted(list(all_keys))

        # Write to CSV file
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            # Write header
            writer.writeheader()

            # Write data rows
            writer.writerows(data)

        print(f"✓ Successfully converted {json_file} to {csv_file}")
        print(f"✓ Total records: {len(data)}")
        print(f"✓ Total columns: {len(fieldnames)}")
        return True

    except FileNotFoundError:
        print(f"Error: File '{json_file}' not found")
        return False
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{json_file}'")
        return False
    except Exception as e:
        print(f"Error: {str(e)}")
        return False


def json_to_csv_flat(json_file, csv_file):
    """
    Convert JSON data to CSV, flattening list fields.
    """
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)

        if not isinstance(data, list) or len(data) == 0:
            print("Error: JSON must contain a non-empty list")
            return False

        # Flatten list fields into comma-separated strings
        flattened_data = []
        for item in data:
            flat_item = {}
            for key, value in item.items():
                if isinstance(value, list):
                    flat_item[key] = ','.join(map(str, value))
                else:
                    flat_item[key] = value
            flattened_data.append(flat_item)

        # Get fieldnames
        fieldnames = sorted(list(flattened_data[0].keys()))

        # Write to CSV
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flattened_data)

        print(f"✓ Converted {json_file} to {csv_file}")
        print(f"✓ Records: {len(flattened_data)}, Columns: {len(fieldnames)}")
        return True

    except Exception as e:
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    # json_to_csv(str(INPUT_FILE), str(OUTPUT_FILE))
    json_to_csv_flat(str(INPUT_FILE), str(OUTPUT_FILE))
