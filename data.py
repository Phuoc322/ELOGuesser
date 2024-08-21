import re

from preprocessor import *

def create_dataset(num_samples, depth):
    dataset = []
    with open('filtered_output.pgn', 'r') as f:
        white_elo = 0
        black_elo = 0
        cnt = 0
        for i, line in enumerate(f):
            if i % 4 == 0:
                white_elo = int(re.findall(r'\d+', line)[0])
            if i % 4 == 1:
                black_elo = int(re.findall(r'\d+', line)[0])
            if i % 4 == 2:
                dataset.append([(white_elo, black_elo)])
            if i % 4 == 3:
                # passes moves, game number (starting from 0) and analysing depth 
                dataset[cnt].append(evaluate_positions_from_pgn_string(line[:-2], int((i - 3)/4), depth))
                cnt += 1
            if i == num_samples * 4:
                break
                
    return dataset