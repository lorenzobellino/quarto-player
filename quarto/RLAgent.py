import logging
import random
import numpy as np
import operator as op
import quarto
import sys
import pickle

from players import RuleBasedPlayer, DumbPlayer, RandomPlayer
from utility import *

NUM_MATCHES = 5000
SAVE_POLICY = True
LOG_FREQ = 1000


class ReinforcementLearningAgent(quarto.Player):
    """A reinforcement learning agent that learns to play Quarto."""

    def __init__(self, quarto: quarto.Quarto) -> None:
        super().__init__(quarto)
        self.is_learning = False
        self.G = dict()
        self.current_state = dict()
        self.randomness = 0.7
        self.learning_rate = 10e-3

    def get_policy(self) -> dict:
        """Returns the policy of the agent."""
        return self.G

    def set_learning(self, is_learning: bool) -> None:
        """Sets the learning mode of the agent."""
        self.is_learning = is_learning

    def choose_piece(self) -> int:
        board = self.get_game().get_board_status()
        available_pieces = get_available_pieces(board)
        available_moves = get_available_moves(board)

        if self.is_learning and random.random() < self.randomness:
            # if is learning choose a ranfdom piece and add it to the current state
            choice = random.choice(available_pieces)
            self.current_state[hash(str(board))] = {"piece": choice}
        else:
            # if not learning choose the best piece if it is in the policy
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
        # if is learning choose a random move and add it to the current state
        if self.is_learning and random.random() < self.randomness:
            choice = random.choice(available_moves)
            self.current_state[hash(str(board))] = {"move": choice}
        else:
            # if not learning choose the best move if it is in the policy
            if hash(str(board)) in self.G:
                try:
                    possible_chioce = self.G[hash(str(board))]["move"]
                    choice = max(possible_chioce, key=possible_chioce[1])
                except:
                    choice = random.choice(available_moves)
            else:
                choice = random.choice(available_moves)
        return choice

    def learn(self, win: bool) -> None:
        for board, value in self.current_state.items():
            # current_state = {board: {"piece": [[piece, score], ..], "move": [[move, score], ..]}}
            flag = False
            if board in self.G:  # if the board is in the policy
                # if the update is for a piece
                if "piece" in value and "piece" in self.G[board]:
                    for p in self.G[board]["piece"]:  # self.G[board]["piece"] = [[piece, score], ..]
                        if p[0] == value["piece"]:
                            if win:
                                p[1] += 1  # if win increase the score by 1
                            else:
                                p[1] -= 1  # if lose decrease the score by 1
                            flag = True
                            break
                    if not flag:  # if the piece is not in the policy add it
                        self.G[board]["piece"].append([value["piece"], 1 if win else -1])
                # if the update is for a move
                elif "move" in value and "move" in self.G[board]:
                    for p in self.G[board]["move"]:  # self.G[board]["move"] = [[move, score], ..]
                        if p[0] == value["move"]:
                            if win:
                                p[1] += 1  # if win increase the score by 1
                            else:
                                p[1] -= 1  # if lose decrease the score by 1
                            flag = True
                            break
                    if not flag:  # if the move is not in the policy add it
                        self.G[board]["move"].append([value["move"], 1 if win else -1])
            else:  # if the board is not in the policy
                if "piece" in value:
                    self.G[board] = {"piece": [[value["piece"], 1 if win else -1]]}  # add a new entry for the piece
                elif "move" in value:
                    self.G[board] = {"move": [[value["move"], 1 if win else -1]]}  # add a new entry for the move

        self.current_state = dict()  # reset the current state
        self.randomness -= self.learning_rate  # decrease the randomness


def reinforcement_training(
    game: quarto.Quarto, agent: ReinforcementLearningAgent, opponent: quarto.Player, num_matches: int
) -> None:
    win = 0
    print(f"training {0} / {NUM_MATCHES}", end="")
    for t in range(num_matches):
        game.reset()
        if t % 2 == 0:
            game.set_players((agent, opponent))
        else:
            game.set_players((opponent, agent))
        winner = game.run()
        if winner == t % 2:
            win += 1
            agent.learn(True)
        else:
            agent.learn(False)

        if SAVE_POLICY and t + 1 % LOG_FREQ == 0:
            print(f"\r\ntraining {t+1} / {NUM_MATCHES} - saving policy\n")
            policy = agent.get_policy()
            pickle.dump(policy, open(f"populations/policy-{t%LOG_FREQ}.pkl", "wb"))
        sys.stdout.flush()
        print(f"\rtraining {t+1} / {NUM_MATCHES}", end="")
    policy = agent.get_policy()
    pickle.dump(policy, open(f"populations/policy-{NUM_MATCHES}.pkl", "wb"))
    return win / num_matches


def test(game: quarto.Quarto, agent: ReinforcementLearningAgent, opponent: quarto.Player, num_matches: int) -> None:
    win = 0
    print(f"testing {0} / {NUM_MATCHES}", end="")
    for t in range(num_matches):
        game.reset()
        if t % 2 == 0:
            game.set_players((agent, opponent))
        else:
            game.set_players((opponent, agent))
        winner = game.run()
        if winner == t % 2:
            win += 1

        sys.stdout.flush()
        print(f"\rtesting {t+1} / {NUM_MATCHES}", end="")
    return win / num_matches


def train():
    game = quarto.Quarto()
    agent = ReinforcementLearningAgent(game)
    agent.set_learning(True)
    opponent = RuleBasedPlayer(game)
    winratio = reinforcement_training(game, agent, opponent, NUM_MATCHES)
    logging.info(f"\nwin ratio after traing: {winratio}")
    agent.set_learning(False)
    logging.info(f"tersting after training")
    winratio = test(game, agent, opponent, NUM_MATCHES)
    logging.info(f"\nwin ratio after testing: {winratio}")


if __name__ == "__main__":
    train()
