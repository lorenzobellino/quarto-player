import random
import quarto

import pickle
import math
import numpy as np

from utility import *

RL_POLICY_FILE = "populations/policy-5000.pkl"


class GAPlayer(quarto.Player):
    """Genetic Algorithm agent"""

    def __init__(self, quarto: quarto.Quarto) -> None:
        super().__init__(quarto)

    def set_genome(self, genome):
        self.genome = genome

    def choose_piece(self) -> int:
        data = cook_data_pieces(self)
        res = (
            (a[0], abs(self.genome["alpha"] * a[1] + self.genome["beta"] * b[1]))
            for a, b in zip(data["alpha"], data["beta"])
        )
        choosen_piece = min(res, key=lambda x: x[1])[0]
        return choosen_piece

    def place_piece(self) -> tuple[int, int]:
        data = cook_data_moves(self, self.get_game().get_selected_piece())
        res = (
            (g[0], abs(self.genome["gamma"] * g[1] + self.genome["delta"] * h[1] + self.genome["epsilon"] * i[1]))
            for g, h, i in zip(data["gamma"], data["delta"], data["epsilon"])
        )
        choosen_move = min(res, key=lambda x: x[1])[0]
        return choosen_move


class TrainedGAPlayer(quarto.Player):
    """Genetic Algorithm agent"""

    def __init__(self, quarto: quarto.Quarto) -> None:
        super().__init__(quarto)
        # self.genome = {
        #     "alpha": -81.66675697723781,
        #     "beta": -113.69148590004534,
        #     "gamma": -29.32357457256496,
        #     "delta": -171.92595218897023,
        #     "epsilon": -10.935275664842214,
        # }
        self.genome = {
            "alpha": -94.03014146974122,
            "beta": -107.17350875193313,
            "gamma": 152.6577141347451,
            "delta": -29.856838596915765,
            "epsilon": -12.095960806170313,
        }

    def choose_piece(self) -> int:
        data = cook_data_pieces(self)
        res = (
            (a[0], abs(self.genome["alpha"] * a[1] + self.genome["beta"] * b[1]))
            for a, b in zip(data["alpha"], data["beta"])
        )
        choosen_piece = min(res, key=lambda x: x[1])[0]
        return choosen_piece

    def place_piece(self) -> tuple[int, int]:
        data = cook_data_moves(self, self.get_game().get_selected_piece())
        res = (
            (g[0], abs(self.genome["gamma"] * g[1] + self.genome["delta"] * h[1] + self.genome["epsilon"] * i[1]))
            for g, h, i in zip(data["gamma"], data["delta"], data["epsilon"])
        )
        choosen_move = min(res, key=lambda x: x[1])[0]
        return choosen_move


class RandomPlayer(quarto.Player):
    """Random player"""

    def __init__(self, quarto: quarto.Quarto) -> None:
        super().__init__(quarto)

    def choose_piece(self) -> int:
        return random.randint(0, 15)

    def place_piece(self) -> tuple[int, int]:
        return random.randint(0, 3), random.randint(0, 3)


class DumbPlayer(quarto.Player):
    """Dumb player"""

    def __init__(self, quarto: quarto.Quarto) -> None:
        super().__init__(quarto)

    def choose_piece(self) -> int:
        status = self.get_game().get_board_status()
        used_pieces = {c for x in status for c in x if c > -1}
        usable_pieces = [_ for _ in {x for x in range(16)} - used_pieces]
        return usable_pieces[0]

    def place_piece(self) -> tuple[int, int]:
        status = self.get_game().get_board_status()
        possible_moves = [(c, r) for r in range(4) for c in range(4) if status[r][c] == -1]
        return possible_moves[0]


class HumanPlayer(quarto.Player):
    """Random player"""

    def __init__(self, quarto: quarto.Quarto) -> None:
        super().__init__(quarto)

    def choose_piece(self) -> int:
        status = self.get_game().get_board_status()
        used_pieces = {c for x in status for c in x if c > -1}
        usable_pieces = [_ for _ in {x for x in range(16)} - used_pieces]
        print(self.get_game().print())
        print(f"Choose a piece: {usable_pieces}")
        piece = int(input())
        while piece not in usable_pieces:
            print("Invalid piece")
            print(f"Choose a piece in : {usable_pieces}")

        return piece

    def place_piece(self) -> tuple[int, int]:
        status = self.get_game().get_board_status()
        possible_moves = [(c, r) for r in range(4) for c in range(4) if status[r][c] == -1]
        print(self.get_game().print())
        print(f"Choose a move: {possible_moves}")
        move = tuple(map(int, input().split()))
        while move not in possible_moves:
            print("Invalid move")
            print(f"Choose a move in : {possible_moves}")
        return move


class RuleBasedPlayer(quarto.Player):
    def __init__(self, quarto: quarto.Quarto) -> None:
        super().__init__(quarto)
        self.previous_board = np.array([[-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1]])
        self.previous_move = None
        self.previous_piece = None
        self.playing_first = False
        # self.winning_pieces = {}

    def choose_piece(self) -> int:
        pieces = block_strategy_piece(self)
        if len(pieces) < 3 and len(pieces) > 0:
            return list(pieces.keys())[0]
        piece = mirror_strategy_piece(self)

        return piece

    def place_piece(self) -> tuple[int, int]:
        move = check_for_win(self.get_game())
        if move is not None:
            return move
        move = mirror_strategy_move(self)
        return move


class MinMaxPlayer(quarto.Player):
    def __init__(self, quarto: quarto.Quarto) -> None:
        super().__init__(quarto)
        self.first_move = True
        self.piece_choice = None
        self.move_choice = None
        self.memory = {}

    def choose_piece(self) -> int:
        piece = 0
        if self.first_move:
            piece = random.randint(0, 15)
        else:
            piece = self.piece_choice
        return piece

    def place_piece(self) -> tuple[int, int]:
        move = (0, 0)
        game = deepcopy(self.get_game())
        value, move, piece = minmax(self, depth=1, alpha=-math.inf, beta=math.inf, isMaximizing=True, game=game)
        self.piece_choice = piece
        self.move_choice = move
        return move

    def evaluate_board(self, isMaximizing, game, last_move, last_piece):

        if game.check_winner() > -1:
            return (100, last_move, last_piece)

        usable_pieces = get_available_pieces(game.get_board_status())
        blocking_pieces = blocking_piece(usable_pieces, game)

        if isMaximizing:
            v = len(blocking_pieces) * 100 / len(usable_pieces)
            return (v, last_move, last_piece)
        else:
            v = (len(usable_pieces) - len(blocking_pieces)) * 100 / len(usable_pieces)
            return (v, last_move, last_piece)


class MixedStrategyPlayer(quarto.Player):
    def __init__(self, quarto: quarto.Quarto) -> None:
        super().__init__(quarto)
        self.previous_board = np.array([[-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1]])
        self.previous_move = None
        self.previous_piece = None
        self.playing_first = False
        self.minmax_piece = None
        self.memory = {}

    def choose_piece(self) -> int:
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
        move = check_for_win(self.get_game())
        if move is not None:
            return move

        usable_pieces = get_available_pieces(self.get_game().get_board_status())
        if len(usable_pieces) < 6:
            game = deepcopy(self.get_game())
            value, move, piece = minmax(self, depth=3, alpha=-math.inf, beta=math.inf, isMaximizing=True, game=game)
            if piece != -1 and value > 0:
                self.minmax_piece = piece
                self.previous_board[move[0]][move[1]] = self.get_game().get_selected_piece()
                self.previous_piece = piece
                return move
        move = mirror_strategy_move(self)

        return move

    def evaluate_board(self, isMaximizing, game, last_move, last_piece):

        if game.check_winner() > -1:
            return (100, last_move, last_piece)

        usable_pieces = get_available_pieces(game.get_board_status())
        blocking_pieces = blocking_piece(usable_pieces, game)

        if isMaximizing:
            v = len(blocking_pieces) * 100 / len(usable_pieces)
            return (v, last_move, last_piece)
        else:
            v = (len(usable_pieces) - len(blocking_pieces)) * 100 / len(usable_pieces)
            return (v, last_move, last_piece)


class TrainedRLPlayer(quarto.Player):
    def __init__(self, quarto: quarto.Quarto) -> None:
        super().__init__(quarto)
        self.is_learning = False
        self.G = pickle.load(open(RL_POLICY_FILE, "rb"))
        self.current_state = dict()
        self.randomness = 0.7
        self.learning_rate = 10e-3

    def choose_piece(self) -> int:
        board = self.get_game().get_board_status()
        available_pieces = get_available_pieces(board)

        if hash(str(board)) in self.G:
            try:
                possible_chioce = self.G[hash(str(board))]["piece"]
                choice = max(possible_chioce, key=possible_chioce[1])
            except:
                choice = random.choice(available_pieces)
        else:
            choice = random.choice(available_pieces)

        return choice

    def place_piece(self) -> tuple[int, int]:
        board = self.get_game().get_board_status()
        available_moves = get_available_moves(board)
        if hash(str(board)) in self.G:
            try:
                possible_chioce = self.G[hash(str(board))]["move"]
                choice = max(possible_chioce, key=possible_chioce[1])
            except:
                choice = random.choice(available_moves)
        else:
            choice = random.choice(available_moves)

        return choice


class MixedStrategyRL(quarto.Player):
    def __init__(self, quarto: quarto.Quarto) -> None:
        super().__init__(quarto)
        self.previous_board = np.array([[-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1]])
        self.previous_move = None
        self.previous_piece = None
        self.playing_first = False
        self.minmax_piece = None
        self.memory = {}
        self.G = pickle.load(open(RL_POLICY_FILE, "rb"))
        self.current_state = dict()
        self.randomness = 0.7
        self.learning_rate = 10e-3

    def choose_piece(self) -> int:
        print("choosing piece")
        available_pieces = get_available_pieces(self.get_game().get_board_status())

        if self.minmax_piece is not None and self.minmax_piece != -1 and self.minmax_piece in available_pieces:
            minmax_piece = self.minmax_piece
            return minmax_piece

        rl_piece = self.choose_piece_rl()
        if rl_piece is not None:
            return rl_piece
        else:
            mirror_piece = mirror_strategy_piece(self)
            if mirror_piece is not None and mirror_piece in available_pieces:
                return mirror_piece
            else:
                return random.choice(available_pieces)

    def place_piece(self) -> tuple[int, int]:
        print("placing piece")
        winning_move = check_for_win(self.get_game())
        if winning_move is not None:
            return winning_move

        usable_pieces = get_available_pieces(self.get_game().get_board_status())

        rl_move = self.choose_move_rl()

        if len(usable_pieces) < 6:
            game = deepcopy(self.get_game())
            value, minmax_move, minmax_piece = minmax(
                self, depth=3, alpha=-math.inf, beta=math.inf, isMaximizing=True, game=game
            )
            if value > 60 and minmax_piece != -1 and minmax_move != (-1, -1):
                self.minmax_piece = minmax_piece
                self.previous_board[minmax_move[0]][minmax_move[1]] = self.get_game().get_selected_piece()
                self.previous_piece = minmax_piece
                return minmax_move
        if rl_move is not None:
            return rl_move
        else:
            mirror_move = mirror_strategy_move(self)
            if mirror_move is not None:
                return mirror_move
            else:
                return random.choice(get_available_moves(self.get_game().get_board_status()))

    def choose_piece_rl(self):
        board = self.get_game().get_board_status()
        if hash(str(board)) in self.G:
            try:
                possible_chioce = self.G[hash(str(board))]["piece"]
                choice = max(possible_chioce, key=possible_chioce[1])
            except:
                choice = None
        else:
            choice = None
        print(f"rl choiche piece{choice}")
        return choice

    def choose_move_rl(self):
        board = self.get_game().get_board_status()
        choice = None
        print(f"g : {self.G.keys()}")
        print(f"b : {hash(str(board))}")
        print(f"s : {str(board)}")
        print(f"len : {len(self.G)}")
        input()
        if hash(str(board)) in self.G:
            try:
                possible_chioce = self.G[hash(str(board))]["move"]
                choice = max(possible_chioce, key=possible_chioce[1])
            except:
                choice = None
        print(f"rl choiche move -> {choice}")
        return choice

    def evaluate_board(self, isMaximizing, game, last_move, last_piece):

        if game.check_winner() > -1:
            return (100, last_move, last_piece)

        usable_pieces = get_available_pieces(game.get_board_status())
        blocking_pieces = blocking_piece(usable_pieces, game)

        if isMaximizing:
            v = len(blocking_pieces) * 100 / len(usable_pieces)
            return (v, last_move, last_piece)
        else:
            v = (len(usable_pieces) - len(blocking_pieces)) * 100 / len(usable_pieces)
            return (v, last_move, last_piece)
