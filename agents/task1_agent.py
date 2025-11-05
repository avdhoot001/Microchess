from minichess.chess.fastchess import Chess
from .base_agent import BaseAgent
import random
import numpy as np
from minichess.chess.fastchess_utils import piece_matrix_to_legal_moves

class Task1Agent(BaseAgent):
    def __init__(self, name="MiniMaxAgent"):
        super().__init__(name)
        self.piece_values = {0: 100, 1: 320, 2: 330, 3: 500, 4: 900, 5: 20000}
        
    def evaluate_board(self, chess_obj: Chess):
        score = 0
        dims = chess_obj.dims
        center = (dims[0] / 2, dims[1] / 2)
        current_turn = chess_obj.turn
        
        for i in range(dims[0]):
            dist_x = abs(i - center[0])
            for j in range(dims[1]):
                dist = dist_x + abs(j - center[1])
                center_bonus = 5 - dist
                
                for color in [0, 1]:
                    piece = chess_obj.piece_at(i, j, color)
                    if piece != -1:
                        sign = 1 if color == current_turn else -1
                        score += sign * (self.piece_values[piece] + center_bonus)
        
        return score
    
    def move(self, chess_obj: Chess):
        moves, proms = chess_obj.legal_moves()
        legal_moves = piece_matrix_to_legal_moves(moves, proms)
        
        if not legal_moves:
            return None
        max_moves_to_evaluate = 10
        if len(legal_moves) > max_moves_to_evaluate:
            sampled_moves = random.sample(legal_moves, max_moves_to_evaluate)
        else:
            sampled_moves = legal_moves
        
        best_move = None
        best_score = -float("inf")
        original_turn = chess_obj.turn
        
        for origin, deltas, promo in sampled_moves:
            i, j = origin
            dx, dy = deltas
            sim_board = chess_obj.copy()
            sim_board.make_move(i, j, dx, dy, promo)
            
            opp_moves, opp_proms = sim_board.legal_moves()
            opp_legal = piece_matrix_to_legal_moves(opp_moves, opp_proms)
            
            if not opp_legal:
                score = 999999 if sim_board.turn != original_turn else -999999
            else:
                max_opp_samples = min(5, len(opp_legal))
                worst_reply_score = float("inf")
                
                for o_origin, o_deltas, o_promo in random.sample(opp_legal, max_opp_samples):
                    oi, oj = o_origin
                    odx, ody = o_deltas
                    opp_board = sim_board.copy()
                    opp_board.make_move(oi, oj, odx, ody, o_promo)
                    
                    eval_score = self.evaluate_board(opp_board)
                    worst_reply_score = min(worst_reply_score, eval_score)
                score = worst_reply_score
            
            if score > best_score:
                best_score = score
                best_move = ((i, j), (dx, dy), promo)
        
        # Fallback
        if best_move is None:
            best_move = random.choice(legal_moves)
        
        return best_move
    
    def reset(self):
        pass


