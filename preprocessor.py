import re

from evaluator import *

# takes PGN files of chess games and outputs the ELO of White, Black and the moves that were played
# filters every game that does not contain either whit's or black's ELO
def preprocess(file):
    # Define the regular expression pattern to capture lines of interest
    pattern = re.compile(
        r'\[WhiteElo\s+"\d+"\]\n\[BlackElo\s+"\d+"\]\n\n1\..*?(?=\s*(?:0-1|1-0|1/2-1/2)\s*$)',
        re.MULTILINE | re.DOTALL
    )

    # Read the file content
    with open(file, 'r') as f:
        content = f.read()

    # Find all matching lines based on the pattern
    matches = pattern.findall(content)

    # Write the filtered lines to a new file
    with open('filtered_output.pgn', 'w') as output_file:
        output_file.write('\n'.join(matches))