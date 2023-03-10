import quarto
import numpy as np
from utility import *


class S309413(quarto.Player):
    """S309413 player"""

    def __init__(self, quarto: quarto.Quarto) -> None:
        super().__init__(quarto)
        self.previous_board = np.array([[-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1]])
        self.previous_move = None
        self.previous_piece = None
        self.minmax_piece = None
        self.memory = {}

    def choose_piece(self) -> int:
        """Choose a piece to play with:
        if minmax strategy is available, use it
        if not, use the blocking strategy
        if not, use the mirror strategy
        """
        if self.minmax_piece is not None and self.minmax_piece in get_available_pieces(
            self.get_game().get_board_status()
        ):
            piece = self.minmax_piece
            return piece

        pieces = block_strategy_piece(self)
        if len(pieces) < 3 and len(pieces) > 0:
            return list(pieces.keys())[0]
        piece = mirror_strategy_piece(self)
        return piece

    def place_piece(self) -> tuple[int, int]:
        """Place a piece on the board:
        if minmax strategy is available, use it
        if not, use the blocking strategy
        if not, use the mirror strategy
        """
        move = check_for_win(self.get_game())
        if move is not None:
            return move

        usable_pieces = get_available_pieces(self.get_game().get_board_status())
        if len(usable_pieces) < 8:
            game = deepcopy(self.get_game())
            value, move, piece = minmax(self, depth=5, alpha=-math.inf, beta=math.inf, isMaximizing=True, game=game)
            if piece != -1 and value > 0:
                self.minmax_piece = piece
                self.previous_board[move[0]][move[1]] = self.get_game().get_selected_piece()
                self.previous_piece = piece
                return move
        move = mirror_strategy_move(self)

        return move

    def evaluate_board(self, isMaximizing, game, last_move, last_piece):
        """Evaluate the board:
        parameters:
            isMaximizing: bool
            game: quarto.Quarto
            last_move: tuple[int, int]
            last_piece: int
        return:
            tuple[int, tuple[int, int], int] = (value, last_move, last_piece)
        """

        if game.check_winner() > -1:
            return (100, last_move, last_piece)

        usable_pieces = get_available_pieces(game.get_board_status())
        blocking_pieces = blocking_piece(usable_pieces, game)

        # the value is the percentage of blocking pieces if isMaximizing is True
        # the value is the percentage of non blocking pieces if isMaximizing is False

        if isMaximizing:
            v = len(blocking_pieces) * 100 / len(usable_pieces)
            return (v, last_move, last_piece)
        else:
            v = (len(usable_pieces) - len(blocking_pieces)) * 100 / len(usable_pieces)
            return (v, last_move, last_piece)
