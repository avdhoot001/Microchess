
from minichess.chess.fastchess import Chess
from .base_agent import BaseAgent
import random
import numpy as np
from minichess.chess.fastchess_utils import piece_matrix_to_legal_moves

class Task1Agent(BaseAgent):
    def __init__(self, name="MiniMaxAgent"):
        super().__init__(name)

    def evaluate_board(self, chess_obj: Chess):
        """Simplified static board evaluation using piece_at()"""
        piece_values = {0: 100, 1: 320, 2: 330, 3: 500, 4: 900, 5: 20000}
        score = 0
        center = (chess_obj.dims[0] / 2, chess_obj.dims[1] / 2)

        for color in [0, 1]:
            sign = 1 if color == chess_obj.turn else -1  # we evaluate from current player's POV
            for i in range(chess_obj.dims[0]):
                for j in range(chess_obj.dims[1]):
                    piece = chess_obj.piece_at(i, j, color)
                    if piece != -1:
                        dist = abs(i - center[0]) + abs(j - center[1])
                        score += sign * (piece_values.get(piece, 0) + (5 - dist))
        return score

    def move(self, chess_obj: Chess):
        moves, proms = chess_obj.legal_moves()
        legal_moves = piece_matrix_to_legal_moves(moves, proms)

        if not legal_moves:
            return None

        best_move = None
        best_score = -float("inf")

        for origin, deltas, promo in legal_moves:
            i, j = origin
            dx, dy = deltas

            # simulate our move
            sim_board = chess_obj.copy()
            sim_board.make_move(i, j, dx, dy, promo)

            # opponent's replies
            opp_moves, opp_proms = sim_board.legal_moves()
            opp_legal = piece_matrix_to_legal_moves(opp_moves, opp_proms)

            if not opp_legal:
                # opponent has no moves (likely checkmate)
                score = 999999 if sim_board.turn != chess_obj.turn else -999999
            else:
                # assume opponent picks the move that minimizes our evaluation
                worst_reply_score = float("inf")
                for o_origin, o_deltas, o_promo in random.sample(opp_legal, min(10, len(opp_legal))):
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

        # fallback
        if best_move is None:
            best_move = random.choice(legal_moves)

        return best_move

    def reset(self):
        pass
