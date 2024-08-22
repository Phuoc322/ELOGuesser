import chess
import chess.engine
import chess.pgn

import numpy

import torch
import torch.nn as nn

import io
import re
import json

def evaluate_positions_from_pgn_string(pgn_string, depth):
    # Use StringIO to mimic a file object with the PGN string
    if isinstance(pgn_string, tuple):
        pgn_io = io.StringIO(pgn_string[0])
    else:
        pgn_io = io.StringIO(pgn_string)
    
    evaluations = load_evaluations(pgn_string, depth)
    if evaluations != False:
        return evaluations
    
    # Parse the game from the PGN string
    game = chess.pgn.read_game(pgn_io)
    
    # Initialize a chess board
    board = game.board()
    
    # Load the chess engine
    engine = chess.engine.SimpleEngine.popen_uci("stockfish")
    
    evaluations = []
    # Iterate over all moves in the game and evaluate the position after each move
    for i, move in enumerate(game.mainline_moves()):
        
        # Push the move to the board
        board.push(move)
        
        # Evaluate the position
        evaluation = engine.analyse(board, chess.engine.Limit(depth=depth))
        
        # Get the evaluation score
        score = evaluation["score"].white()
        
        if score.is_mate():
            evaluations.append(score.mate() * 5)
        else:
            # Convert centipawn score to a float
            float_score = score.score() / 100.0
            evaluations.append(float_score)

    save_evaluations(pgn_string, depth, evaluations)
    engine.quit()
    return evaluations

# print(evaluate_positions_from_pgn_string("1. e4 a5 2. Bc4 d5 3. Bxd5 c5 4. Qf3 g6 5. d3 Qd6 6. Nh3 Qd8 7. Be3 c4 8. Nc3 Nc6 9. Qxf7+ Kd7 10. Be6+ Kc7 11. Nb5+ Kb8 12. Bf4+ Ne5 13. Bxe5+ Qd6 14. Bxd6+ exd6 15. Qc7#", depth=10))
# print(evaluate_positions_from_pgn_string("1. f3 e5 2. Na3 Qf6 3. Nb1 Bc5 4. Na3 Nc6 5. Nb1 Nh6 6. Na3 Nf5 7. Nb1 Qh4+ 8. g3 Nxg3 9. e3 Nxh1+ 10. Ke2 Nd4+ 11. exd4 Qf2+ 12. Kd3 e4+ 13. Kxe4 Qxd4+ 14. Kf5 Be7 15. a3 g6#", depth=10))

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
                # pgn_string and analysing depth 
                dataset[cnt].append(evaluate_positions_from_pgn_string(line[:-1], depth))
                cnt += 1
            if i == num_samples * 4:
                break
                
    return dataset

def save_evaluations(pgn_string, depth, evaluation):
    # dictionary with evaluations for each game
    # key: pgn_string
    # value: depth, evaluations
    
    #read
    with open("evaluated_positions.json", 'r') as f:
        evaluations = json.load(f)
    
    # update dictionary
    evaluations.update({pgn_string: (depth, evaluation)})
    
    # write
    with open("evaluated_positions.json", 'w') as f:
        json.dump(evaluations, f, separators=(',', ': '))
        
def load_evaluations(pgn_string, depth):
    # returns False if dictionary either does not contain the evaluation or
    # the requested analysis depth is larger than the depth with which the evaluations was made
    # otherwise return the evaluation
    
    # read
    with open("evaluated_positions.json", 'r') as f:
        evaluations = json.load(f)
    
    if pgn_string in evaluations:
        if depth <= evaluations[pgn_string][0]:
            return evaluations[pgn_string][1]
        else:
            return False
    else:
        return False
    
def initialize_evaluations(file):
    with open(file, 'w') as f:
        evaluations = {"1. j0": (-1, [0])}
        # initialize with arbitrary data
        json.dump(evaluations, f, separators=(',', ': '))