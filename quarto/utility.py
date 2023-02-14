import quarto
import operator
import random
import math
from copy import deepcopy
import logging
from itertools import accumulate
from functools import cache


def cook_data_pieces(state: quarto.Quarto) -> dict:
    data = {}
    status = state.get_game().get_board_status()
    # print(status)
    used_pieces = {c for x in status for c in x if c > -1}
    usable_pieces = {x for x in range(16)} - used_pieces
    alpha = list()
    beta = list()
    # print(usable_pieces)
    # print(used_pieces)
    if len(usable_pieces) == 1:
        return {"alpha": [(list(usable_pieces)[0], 1)], "beta": [(list(usable_pieces)[0], 1)]}
    for p in usable_pieces:
        usable_pieces.remove(p)
        *_, a = accumulate([_ for _ in usable_pieces], operator.add)
        usable_pieces.add(p)
        used_pieces.add(p)
        *_, b = accumulate(used_pieces, lambda x, y: ~(x & y))
        used_pieces.remove(p)
        alpha.append((p, a))
        beta.append((p, b))
    data["alpha"] = alpha
    data["beta"] = beta
    return data


def cook_data_moves(state: quarto.Quarto, piece: int) -> dict:
    data = {}
    status = state.get_game().get_board_status()
    possible_moves = [(c, r) for r in range(4) for c in range(4) if status[r][c] == -1]
    gamma = list()
    delta = list()
    epsilon = list()
    # print(possible_moves)
    if len(possible_moves) == 1:
        return {
            "gamma": [(possible_moves[0], 1)],
            "delta": [(possible_moves[0], 1)],
            "epsilon": [(possible_moves[0], 1)],
        }
    for p in possible_moves:
        possible_moves.remove(p)
        *_, g = accumulate([x[0] for x in possible_moves], operator.or_)
        *_, d = accumulate([x[1] for x in possible_moves], operator.or_)
        *_, e = accumulate([x[0] for x in possible_moves], operator.and_)
        possible_moves.append(p)
        gamma.append((p, g))
        delta.append((p, d))
        epsilon.append((p, e))
    data["gamma"] = gamma
    data["delta"] = delta
    data["epsilon"] = epsilon

    return data


def sum_operation(status) -> int:
    """
    Perform the sum operation on the pieces on the board
    param state: Nim
    return: result: int, the sub operation of the state
    """
    *_, r = accumulate(status, operator.add)
    try:
        *_, result = accumulate(r, operator.add)
    except TypeError:
        result = r
    return result


def sub_operation(status) -> int:
    """
    Perform the sub operation on the pieces on the board
    param state: Nim
    return: result: int, the sub operation of the state
    """
    *_, r = accumulate(status, operator.sub)
    try:
        *_, result = accumulate(r, operator.sub)
    except TypeError:
        result = r
    return result


def nand_operation(status) -> int:
    """
    Perform the nand operation on the pieces on the board
    param state: Nim
    return: result: int, the nand operation of the state
    """
    *_, r = accumulate(status, lambda x, y: ~(x & y))
    try:
        *_, result = accumulate(r, lambda x, y: ~(x & y))
    except TypeError:
        result = r
    return result


def and_operation(status) -> int:
    """
    Perform the and operation on the pieces on the board
    param state: Nim
    return: result: int, the and operation of the state
    """
    *_, r = accumulate(status, operator.and_)
    try:
        *_, result = accumulate(r, operator.and_)
    except TypeError:
        result = r
    return result


def or_operation(status) -> int:
    """
    Perform the or operation on the pieces on the board
    param state: Nim
    return: result: int, the or operation of the state
    """
    *_, r = accumulate(status, operator.or_)
    try:
        *_, result = accumulate(r, operator.or_)
    except TypeError:
        result = r
    return result


def mirror_strategy_piece(self):
    piece = random.randint(0, 15)

    current_board = self.get_game().get_board_status()
    if self.previous_piece == None:
        self.playing_first = True

    if self.playing_first and self.previous_piece == None:
        piece = random.randint(0, 15)
    elif self.playing_first and self.previous_piece != None:
        piece_info = self.get_game().get_piece_charachteristics(self.previous_piece).binary
        used_pieces = {c for x in current_board for c in x if c > -1}
        usable_pieces = [_ for _ in {x for x in range(16)} - used_pieces]
        pieces = list()
        for p in usable_pieces:
            p_info = [int(x) for x in format(p, "04b")]
            r = sum([abs(x - y) for x, y in zip(p_info, piece_info)])
            pieces.append((p, r))
            if r == 4:
                piece = p
                break
        piece = max(pieces, key=lambda x: x[1])[0]
    else:
        piece_info = self.get_game().get_piece_charachteristics(self.previous_piece).binary
        used_pieces = {c for x in current_board for c in x if c > -1}
        usable_pieces = [_ for _ in {x for x in range(16)} - used_pieces]
        pieces = list()
        for p in usable_pieces:
            p_info = [int(x) for x in format(p, "04b")]
            # print(f"p_info: {p_info} piece_info: {piece_info}")
            r = sum([abs(x - y) for x, y in zip(p_info, piece_info)])
            pieces.append((p, r))
            if r == 4:
                piece = p
                break
        piece = max(pieces, key=lambda x: x[1])[0]
    # print(f"selected piece : {piece}")
    # print(f"prev piece: {self.previous_piece}")
    # input()
    return piece


def mirror_strategy_move(self):
    self.previous_piece = self.get_game().get_selected_piece()
    # print("playing first: ", self.playing_first)
    if self.playing_first:
        self.previous_piece = self.get_game().get_selected_piece()
        current_board = self.get_game().get_board_status()
        for i, r in enumerate(zip(self.previous_board, current_board)):
            # print(f"r1: {r[0]} r2: {r[1]}")
            for j, c in enumerate(zip(r[0], r[1])):
                # print(f"c1: {c[0]} c2: {c[1]}")
                if c[0] != c[1]:
                    # print(f"r:{current_board.index(r2)} c={r2.index(c2)}")
                    self.previous_move = (i, j)
                    # self.previous_move = (current_board.where(x == r2), r2.where(x == c2))
                    # print(f"updating prev move: {self.previous_move}")
                    break
        possible_moves = [(c, r) for r in range(4) for c in range(4) if self.previous_board[r][c] == -1]
        move = (3 - self.previous_move[1], 3 - self.previous_move[0])
        if move not in possible_moves:
            move = random.choice(possible_moves)
        self.previous_board = deepcopy(current_board)
        self.previous_board[move[1]][move[0]] = self.previous_piece
    else:
        # print("going second place piece")
        move = (-1, -1)
        current_board = self.get_game().get_board_status()
        for i, r in enumerate(zip(self.previous_board, current_board)):
            # print(f"r1: {r[0]} r2: {r[1]}")
            for j, c in enumerate(zip(r[0], r[1])):
                # print(f"c1: {c[0]} c2: {c[1]}")
                if c[0] != c[1]:
                    # print(f"r:{current_board.index(r2)} c={r2.index(c2)}")
                    self.previous_move = (i, j)
                    # self.previous_move = (current_board.where(x == r2), r2.where(x == c2))
                    # print(f"updating prev move: {self.previous_move}")
                    break
        possible_moves = [(c, r) for r in range(4) for c in range(4) if self.previous_board[r][c] == -1]

        if move not in possible_moves or self.previous_move == None:
            try:
                move = random.choice(possible_moves)
            except IndexError:
                print("index error")
                print(possible_moves)
                print(self.previous_board)
        else:
            move = (3 - self.previous_move[1], 3 - self.previous_move[0])

        self.previous_board = deepcopy(current_board)
        self.previous_board[move[1]][move[0]] = self.previous_piece

        # input()
    return move


def block_strategy_piece(self):
    piece = -1
    winner = False
    current_board = self.get_game().get_board_status()
    used_pieces = {c for x in current_board for c in x if c > -1}
    usable_pieces = [_ for _ in {x for x in range(16)} - used_pieces]
    possible_moves = [(c, r) for r in range(4) for c in range(4) if current_board[r][c] == -1]
    pieces = {}
    for p in usable_pieces:
        winner = False
        for m in possible_moves:
            board = deepcopy(self.get_game())

            board.select(p)
            board.place(m[0], m[1])
            if board.check_winner() > -1:
                winner = True
                # print(current_board)
                # print(board.check_winner())
                # print(f"blocking piece: {p}")
                # print(board.get_board_status())
                # print(f"move: {m}")
                # input()
        if not winner:
            pieces[p] = m

    return pieces


def check_for_win(gameboard):
    move = None

    piece = gameboard.get_selected_piece()
    current_board = gameboard.get_board_status()
    possible_moves = [(c, r) for r in range(4) for c in range(4) if current_board[r][c] == -1]
    for m in possible_moves:
        game = deepcopy(gameboard)
        game.select(piece)
        game.place(m[0], m[1])
        if game.check_winner() > -1:
            # print("winning move found")
            # print(f"piece: {piece}")
            # print(f"move: {m}")
            # print(f"{game.get_board_status()}")
            move = m
            # input()
            break

    return move


def blocking_piece(usable_pieces, game):
    winner = False
    pieces = {}
    possible_moves = [(c, r) for r in range(4) for c in range(4) if game.get_board_status()[r][c] == -1]
    for p in usable_pieces:
        winner = False
        for m in possible_moves:
            board = deepcopy(game)
            board.select(p)
            board.place(m[0], m[1])
            if board.check_winner() > -1:
                winner = True
                # print(current_board)
                # print(board.check_winner())
                # print(f"blocking piece: {p}")
                # print(board.get_board_status())
                # print(f"move: {m}")
                # input()
        if not winner:
            pieces[p] = m

    return pieces


@cache
def minmax(self, depth, alpha, beta, isMaximizing, last_move=None, last_piece=None, game=None):
    """Minmax to choise the best move to play and piece to use"""

    # input()
    if (isMaximizing and game.check_winner() > -1) or depth == 0:
        # print("evaluationm")
        evaluation = self.evaluate_board(isMaximizing, game, last_move, last_piece)
        self.memory[(isMaximizing, hash(str(game)))] = evaluation
        return evaluation
    # if depth == 0:
    #     return (self.evaluate_board(isMaximizing, board), last_move, last_piece, game)

    # if isMaximizing and victory(board):
    #     return (self.evaulate_board(isMaximizing, board), last_move, last_piece)
    # if (self, isMaximizing) in self.memory:
    #     return self.memory[(self, isMaximizing)]

    if (isMaximizing, hash(str(game))) in self.memory:
        print("memory")
        return self.memory[(isMaximizing, hash(str(game)))]

    best_choice = None
    # board = game.get_board_status()
    selected_piece = game.get_selected_piece()
    board = game.get_board_status()
    avvailable_piece = get_available_pieces(board, selected_piece)
    available_moves = get_available_moves(board)
    if isMaximizing:
        # print(board)
        # value = -math.inf
        best_choice = (-math.inf, -1, -1)
        # board = deepcopy(self.get_game().get_board_status())
        # avvailable_piece = get_available_pieces(board)
        # available_moves = get_available_moves(board)
        # print(
        #     f"depth: {depth} last_move: {last_move} last_piece: {last_piece} isMaximizing: {isMaximizing} selected piece: {game.get_selected_piece()} \n {board}"
        # )
        # selected_piece = game.get_selected_piece()
        for m in available_moves:
            game_copy = deepcopy(game)
            game_copy.place(m[0], m[1])
            for p in avvailable_piece:
                if not game_copy.select(p):
                    logging.debug(f"piece {p} not available")
                    # print(f"pice: {p} move: {m} \n {game_copy.get_board_status()}")
                evaluation = minmax(self, depth - 1, alpha, beta, False, m, p, game_copy)
                # print(f"evaluation max: {evaluation}")
                best_choice = max(best_choice, evaluation, key=lambda x: x[0])
                alpha = max(alpha, best_choice[0])
                if beta <= alpha or best_choice[0] == 100:
                    break
            if best_choice[0] == 100:
                break
        return best_choice

    else:
        best_choice = (math.inf, -1, -1)
        # board = deepcopy(self.get_game().get_board_status())
        # avvailable_piece = get_available_pieces(board)
        # available_moves = get_available_moves(board)
        # print(
        #     f"depth: {depth} last_move: {last_move} last_piece: {last_piece} isMaximizing: {isMaximizing} selected piece: {game.get_selected_piece()} \n {board}"
        # )
        # # selected_piece = game.get_selected_piece()
        for m in available_moves:
            game_copy = deepcopy(game)
            game_copy.place(m[0], m[1])
            for p in avvailable_piece:
                if not game_copy.select(p):
                    logging.debug(f"piece {p} not available")
                    # print(f"pice: {p} move: {m} \n {game_copy.get_board_status()}")
                evaluation = minmax(self, depth - 1, alpha, beta, True, m, p, game_copy)
                # print(f"evaluation min: {evaluation}")
                best_choice = min(best_choice, evaluation, key=lambda x: x[0])
                beta = min(beta, best_choice[0])
                if beta <= alpha or best_choice[0] == 0:
                    break
            if best_choice[0] == 0:
                break
        return best_choice


'''
def minmax(self, depth, alpha, beta, isMaximizing, last_move=None, last_piece=None, board=None):
    """Minmax to choise the best move to play and piece to use"""
    if depth == 0:
        return (self.evaluate_board(isMaximizing, board), last_move, last_piece)
    # if isMaximizing and victory(board):
    #     return (self.evaulate_board(isMaximizing, board), last_move, last_piece)
    # if (self, isMaximizing) in self.memory:
    #     return self.memory[(self, isMaximizing)]
    best_choice = None
    if isMaximizing:
        # print(board)
        # value = -math.inf
        best_choice = (-math.inf, -1, -1)
        # board = deepcopy(self.get_game().get_board_status())
        avvailable_piece = get_available_pieces(board)
        available_moves = get_available_moves(board)
        for p in avvailable_piece:
            for m in available_moves:
                board[m[0]][m[1]] = p
                evaluation = minmax(self, depth - 1, alpha, beta, False, m, p, board)
                best_choice = max(best_choice, evaluation, key=lambda x: x[0])
                board[m[0]][m[1]] = -1
                alpha = max(alpha, best_choice[0])
                if beta <= alpha:
                    break
        return best_choice
    else:
        best_choice = (math.inf, -1, -1)
        # board = deepcopy(self.get_game().get_board_status())
        avvailable_piece = get_available_pieces(board)
        available_moves = get_available_moves(board)
        for p in avvailable_piece:
            for m in available_moves:
                board[m[0]][m[1]] = p
                evaluation = minmax(self, depth - 1, alpha, beta, True, m, p, board)
                # print(best_choice)
                # print(evaluation)
                best_choice = min(evaluation, best_choice, key=lambda x: x[0])
                board[m[0]][m[1]] = -1
                beta = min(beta, best_choice[0])
                if beta <= alpha:
                    break
        return best_choice
'''


def get_available_pieces(board, selected_piece=None):
    used_pieces = {c for x in board for c in x if c > -1}
    if selected_piece is not None:
        used_pieces.add(selected_piece)
    return [_ for _ in {x for x in range(16)} - used_pieces]


def get_available_moves(board):
    return [(c, r) for r in range(4) for c in range(4) if board[r][c] == -1]
