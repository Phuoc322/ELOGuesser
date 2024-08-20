import re

# Define the regular expression pattern to capture lines of interest
pattern = re.compile(
    r'\[WhiteElo\s+"\d+"\]\n\[BlackElo\s+"\d+"\]\n\n1\..*?(?=\s*(?:0-1|1-0|1/2-1/2)\s*$)',
    re.MULTILINE | re.DOTALL
)

# Read the file content
with open('games from 100 to 2100.pgn', 'r') as file:
    content = file.read()

# Find all matching lines based on the pattern
matches = pattern.findall(content)

# Write the filtered lines to a new file
with open('filtered_output.pgn', 'w') as output_file:
    output_file.write('\n'.join(matches))
    
# TODO
# Extract White and Black Elo from file and the moves that were played, i and i+1 are Elo, i + 3 are the moves