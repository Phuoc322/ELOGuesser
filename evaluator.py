import chess
import chess.engine
import chess.pgn

import numpy

import torch
import torch.nn as nn

import io

def evaluate_positions_from_pgn_string(pgn_string):
    # Use StringIO to mimic a file object with the PGN string
    if isinstance(pgn_string, tuple):
        pgn_io = io.StringIO(pgn_string[0])
    else:
        pgn_io = io.StringIO(pgn_string)
    
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
        evaluation = engine.analyse(board, chess.engine.Limit(time=0.005))
        
        # Get the evaluation score
        score = evaluation["score"].relative
        
        if score.is_mate():
            evaluations.append(-score.mate() * 100 if (i % 2 == 0) else score.mate() * 100)
        else:
            # Convert centipawn score to a float
            float_score = score.score() / 100.0
            evaluations.append(float_score)
            
    engine.quit()
    return evaluations

def save_evaluations(sample_num):
    # dictionary with evaluations for each game
    # key: sample_num
    # value: depth, moves
    dict