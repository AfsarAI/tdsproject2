
import csv
import json
from datetime import datetime

def to_snake_case(name):
    return name.lower().replace(' ', '_')

data = []
with open('messy.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        processed_row = {}
        for key, value in row.items():
            snake_key = to_snake_case(key)
            if snake_key == 'id':
                processed_row[snake_key] = int(value)
            elif snake_key == 'joined':
                dt_object = None
                # Try parsing with different formats
                try:
                    dt_object = datetime.strptime(value, '%Y-%m-%d %H:%M:%S') # For dates with time
                except ValueError:
                    try:
                        dt_object = datetime.strptime(value, '%Y-%m-%d') # For YYYY-MM-DD
                    except ValueError:
                        try:
                            dt_object = datetime.strptime(value, '%m/%d/%y') # For MM/DD/YY
                        except ValueError:
                            try:
                                dt_object = datetime.strptime(value, '%d %b %Y') # For DD Mon YYYY
                            except ValueError:
                                pass # If none match, dt_object remains None

                if dt_object:
                    # Since all input dates are date-only, format to YYYY-MM-DD
                    processed_row[snake_key] = dt_object.strftime('%Y-%m-%d')
                else:
                    # Fallback if no date format matched (shouldn't happen with valid data)
                    processed_row[snake_key] = value # Keep original value or raise error
            elif snake_key == 'value':
                processed_row[snake_key] = int(float(value)) # Handle potential floats and whitespace
            else:
                processed_row[snake_key] = value
        data.append(processed_row)

# Sort by 'id' ascending
sorted_data = sorted(data, key=lambda x: x['id'])

json_output = json.dumps(sorted_data, indent=None) # No indent for submission
print(json_output)
