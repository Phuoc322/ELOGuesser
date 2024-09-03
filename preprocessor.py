import re
import chess.pgn
import os

import time

# takes PGN files of chess games and outputs the ELO of White, Black and the moves that were played
# filters every game that does not contain either whit's or black's ELO
def preprocess(file, output_file):
    # Define the regular expression pattern to capture lines of interest
    pattern = re.compile(
        r'\[WhiteElo\s\"\d+\"\]\n\[BlackElo\s\"\d+\"\]\n\n1\.[^\n]+',
        re.DOTALL
    )

    # Read the file content
    with open(file, 'r') as f:
        content = f.read()

    # Find all matching lines based on the pattern
    matches = pattern.findall(content)

    # Write the filtered lines to a new file
    with open(output_file, 'w') as output_file:
        output_file.write('\n'.join(matches))


def preprocess_data(input_file, output_file):
    # open input file to read and output file to write
    pgn_file = open(input_file)
    curr_game = chess.pgn.read_game(pgn_file)

    # create and open the output file
    with open(output_file, 'w') as out_file:
        # iterate through all games (segments)
        while curr_game is not None:
            # check if the game has a variant
            cond1 = "Variant" not in curr_game.headers.keys()
            # check if the game ended due to time limit
            cond2 = curr_game.headers["Termination"] != "Time forfeit"
            # check if any player is not listed (has unknown elo value)
            cond3 = curr_game.headers["WhiteElo"] != '?' and curr_game.headers["BlackElo"] != '?'

            if cond1 and cond2 and cond3:
                # gather all information for output file
                elo_w_val = curr_game.headers["WhiteElo"]
                elo_b_val = curr_game.headers["BlackElo"]
                res = curr_game.headers["Result"]
                moves = curr_game.mainline_moves().__str__().replace('#', '')

                elo_w = f"[WhiteElo \"{elo_w_val}\"]"
                elo_b = f"[BlackElo \"{elo_b_val}\"]"

                # write entries to file
                out_file.write(elo_w + "\n")
                out_file.write(elo_b + "\n")
                out_file.write("\n")
                out_file.write(moves + " " + res + "\n")

            curr_game = chess.pgn.read_game(pgn_file)

if __name__ == "__main__":
    start_time = time.time()
    preprocess_data("data/lichess_db_standard_rated_2013-01.pgn", "data/output.pgn")
    print("Execution time --- %s seconds ---" % (time.time() - start_time))
