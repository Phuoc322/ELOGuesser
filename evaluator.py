import chess
import chess.engine
import chess.pgn

import numpy

import torch
import torch.nn as nn

import io

def evaluate_positions_from_pgn_string(pgn_string):
    # Use StringIO to mimic a file object with the PGN string
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
        evaluation = engine.analyse(board, chess.engine.Limit(time=0.1))
        
        # Get the evaluation score
        score = evaluation["score"].relative
        
        # Display the board and the evaluation
        print(board)
        if score.is_mate():
            print(f"Mate in {score.mate()}\n")
            evaluations.append(-score.mate() * 100 if (i % 2 == 0) else score.mate() * 100)
        else:
            # Convert centipawn score to a float
            float_score = score.score() / 100.0
            print(f"Evaluation: {float_score:.2f}\n")
            evaluations.append(float_score)
            
    engine.quit()
    return evaluations