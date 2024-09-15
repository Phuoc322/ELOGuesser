import re
import chess.pgn
import os

import time
import matplotlib.pyplot as plt
import numpy as np

from evaluator import evaluate_positions_from_pgn_string


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
    with (open(output_file, 'w') as out_file):
        # iterate through all games (segments)
        while curr_game is not None:
            # check if termination header exists and if it is not "Time forfeit"
            termination = False
            if "Termination" in curr_game.headers.keys():
                termination = curr_game.headers["Termination"] != "Time forfeit"

            # check if elo values are numeric
            w_c = curr_game.headers["WhiteElo"].isnumeric()
            b_c = curr_game.headers["BlackElo"].isnumeric()

            # if the elo value is not a number set value to unreachable value to skip this game
            w = int(curr_game.headers["WhiteElo"]) if w_c else 0
            b = int(curr_game.headers["BlackElo"]) if b_c else 0

            # filter the string move list to get number of moves
            lst = curr_game.mainline_moves().__str__().split()
            move_lst = [s for s in lst if "." not in s]

            # check if play time is available
            game_time = 0
            if 'TimeControl' in curr_game.headers.keys():
                if curr_game.headers["TimeControl"].split('+')[0] != '-':
                    game_time = int(curr_game.headers["TimeControl"].split('+')[0])

            # gather all filter conditions
            conditions = [
                # check if the game has a variant
                "Variant" not in curr_game.headers.keys(),

                # check if the game ended due to time limit
                termination,

                # check if any player is not listed (has unknown elo value)
                curr_game.headers["WhiteElo"] != '?' and curr_game.headers["BlackElo"] != '?',

                # check if both players have elo value between 500 - 2500
                500 <= w <= 2500 and 500 <= b <= 2500,

                # check for games with limited numbers of moves (x < 80)
                len(move_lst) < 80,

                # check for games with at least 10 min play time
                game_time >= 600
            ]

            if all(conditions):
                # gather all information for output file
                elo_w_val = curr_game.headers["WhiteElo"]
                elo_b_val = curr_game.headers["BlackElo"]
                res = curr_game.headers["Result"]
                moves = curr_game.mainline_moves().__str__()

                elo_w = f"[WhiteElo \"{elo_w_val}\"]"
                elo_b = f"[BlackElo \"{elo_b_val}\"]"

                # write entries to file
                out_file.write(elo_w + "\n")
                out_file.write(elo_b + "\n")
                out_file.write("\n")
                out_file.write(moves + " " + res + "\n")

            curr_game = chess.pgn.read_game(pgn_file)


def visualize_dataset(dataset, bins):
    white = np.array([row[0][0] for row in dataset])
    black = np.array([row[0][1] for row in dataset])

    # visualize histogram
    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

    # We can set the number of bins with the *bins* keyword argument.
    axs[0].hist(white, bins=bins)
    axs[0].set_title("WhiteElo")
    axs[0].set_xlabel("ELO Value")
    axs[0].set_ylabel("Frequency")

    axs[1].hist(black, bins=bins)
    axs[1].set_title("BlackElo")
    axs[1].set_xlabel("ELO Value")
    axs[1].set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig('./data/subplots_histogram.png')

    plt.show()

def balance_dataset(dataset, bins, num_games, pgn_file, depth):
    # get histograms of white and black elo values
    white = np.array([row[0][0] for row in dataset])
    black = np.array([row[0][1] for row in dataset])

    w_count, w_bin_edges = np.histogram(white, bins=bins)
    b_count, b_bin_edges = np.histogram(black, bins=bins)

    # get max and min values (number of occurrence of bin class)
    w_max, w_min = np.max(w_count), np.min(w_count)
    b_max, b_min = np.max(b_count), np.min(b_count)

    with open(pgn_file, 'r') as f:
        cnt = num_games
        write = False
        for i, line in enumerate(f):
            if w_max - w_max * 0.1 <= np.min(w_count) and b_max - b_max * 0.1 <= np.min(b_count):
                break

            # start adding games at the last line being processed
            if i >= num_games * 4:
                if i % 4 == 0:
                    white_elo = int(re.findall(r'\d+', line)[0])
                if i % 4 == 1:
                    black_elo = int(re.findall(r'\d+', line)[0])
                if i % 4 == 2:
                    # check in which range the elo value belongs
                    w_idx = np.digitize(white_elo, w_bin_edges)-1
                    b_idx = np.digitize(black_elo, b_bin_edges)-1

                    # if elo value is higher than last bin edge, set it to the last index
                    w_idx = bins - 1 if w_idx == bins else w_idx
                    b_idx = bins - 1 if b_idx == bins else b_idx

                    # find all indices of max occurrence
                    w_max_idx = np.where(w_count == np.max(w_count))[0]
                    b_max_idx = np.where(b_count == np.max(b_count))[0]

                    # check if both white elo and black elo are not any of the max occurrences
                    conditions = [
                        np.isin(w_idx, w_max_idx, invert=True),
                        np.isin(b_idx, b_max_idx, invert=True)
                    ]

                    # add elo values to dataset and increment count of bin class
                    if all(conditions):
                        dataset.append([(white_elo, black_elo)])
                        w_count[w_idx] += 1
                        b_count[b_idx] += 1
                        write = True

                # only write evaluation of moves if current game was added to dataset
                if i % 4 == 3 and write:
                    dataset[cnt].append(evaluate_positions_from_pgn_string(line[:-1], depth))
                    write = False

    return dataset, w_count, b_count



if __name__ == "__main__":
    start_time = time.time()
    preprocess_data("data/lichess_db_standard_rated_2013-12.pgn", "data/output_all_filter.pgn")
    print("Execution time --- %s seconds ---" % (time.time() - start_time))
