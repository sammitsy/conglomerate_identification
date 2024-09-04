import pandas as pd
import re

# Load the CSV file
data = pd.read_csv('conquest_search/non-congloms_conquest.csv')

# Compile the regular expression pattern to find compound names with '(+)', '(-)', 'S', or 'R'
pattern = re.compile(r'\(\+\)|\(-\)|[SR]|\+-')

# Function to determine row exclusion based on the presence of any forbidden pattern in the targeted column
def exclude_row(row):
    try:
        # Check the specified column for the pattern, and ensure the output is always a boolean
        system_match = bool(pattern.search(str(row['Title'])))
        return system_match
    except KeyError:
        # If specified column is not found, return False to not exclude the row
        return False


# Apply the function to each row and filter the dataframe
filtered_data = data[~data.apply(exclude_row, axis=1)]

# For checking purposes
print(filtered_data.head())
print(len(filtered_data))

# Save the filtered data to a new CSV file
filtered_data.to_csv('non-congloms_filtered_punc.csv', index=False)




