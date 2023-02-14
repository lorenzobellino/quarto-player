import logging
import random
import numpy as np
import operator as op
import quarto
import sys
import pickle

from players import RuleBasedPlayer, DumbPlayer, RandomPlayer
from utility import *

NUM_MATCHES = 1000
SAVE_POLICY = True
LOG_FREQ = 400


class ReinforcementLearningAgent(quarto.Player):
    def __init__(self, quarto: quarto.Quarto) -> None:
        super().__init__(quarto)
        self.is_learning = False
        self.G = dict()
        self.current_state = dict()
        self.randomness = 0.7
        self.learning_rate = 10e-3

    def get_policy(self) -> dict:
        return self.G

    def set_learning(self, is_learning: bool) -> None:
        self.is_learning = is_learning

    def choose_piece(self) -> int:
        board = self.get_game().get_board_status()
        available_pieces = get_available_pieces(board)
        available_moves = get_available_moves(board)

        if self.is_learning and random.random() < self.randomness:
            choice = random.choice(available_pieces)
            self.current_state[hash(str(board))] = {"piece": choice}
        else:
            if hash(str(board)) in self.G:

                try:
                    possible_chioce = self.G[hash(str(board))]["piece"]
                    choice = max(possible_chioce, key=possible_chioce[1])
                except:
                    choice = random.choice(available_pieces)
                # choice = max(self.G[hash(str(board))], key=self.G[hash(str(board))]["reward"])["move"]
            else:
                choice = random.choice(available_pieces)

        return choice

    def place_piece(self) -> tuple[int, int]:
        board = self.get_game().get_board_status()
        available_moves = get_available_moves(board)
        # available_pieces = get_available_pieces(board)
        # choice = random.choice(available_moves)

        if self.is_learning and random.random() < self.randomness:
            # self.current_state[hash(str(board))]["place_piece"] = choice
            choice = random.choice(available_moves)
            self.current_state[hash(str(board))] = {"move": choice}
            # return choice
        else:
            if hash(str(board)) in self.G:
                try:
                    possible_chioce = self.G[hash(str(board))]["move"]
                    choice = max(possible_chioce, key=possible_chioce[1])
                except:
                    choice = random.choice(available_moves)
                # choice = max(self.G[hash(str(board))], key=self.G[hash(str(board))]["piece"])["piece"]
            else:
                choice = random.choice(available_moves)
        return choice

    def learn(self, win: bool) -> None:
        # TODO learning phase
        for board, value in self.current_state.items():
            # print(f"board: {board} \nvalue: {value}")
            flag = False
            if board in self.G:
                if "piece" in value and "piece" in self.G[board]:
                    for p in self.G[board]["piece"]:
                        if p[0] == value["piece"]:
                            if win:
                                p[1] += 1
                            else:
                                p[1] -= 1
                            flag = True
                            break
                    if not flag:
                        self.G[board]["piece"].append([value["piece"], 1 if win else -1])
                elif "move" in value and "move" in self.G[board]:
                    for p in self.G[board]["move"]:
                        if p[0] == value["move"]:
                            if win:
                                p[1] += 1
                            else:
                                p[1] -= 1
                            flag = True
                            break
                    if not flag:
                        self.G[board]["move"].append([value["move"], 1 if win else -1])

            else:
                if "piece" in value:
                    self.G[board] = {"piece": [[value["piece"], 1 if win else -1]]}
                elif "move" in value:
                    self.G[board] = {"move": [[value["move"], 1 if win else -1]]}

        self.current_state = dict()
        self.randomness -= self.learning_rate


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
            # print(policy)
            # policy_str = {"".join(str(_) for _ in k): v for k, v in policy.items()}
            # json.dump(policy_str, open(f"populations/policy-{t%LOG_FREQ}.json", "w"))
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
    print(f"\nwin ratio: {winratio}")
    # print(f"policy: {agent.get_policy()}")
    agent.set_learning(False)
    winratio = test(game, agent, opponent, NUM_MATCHES)
    print(f"\nwin ratio: {winratio}")


if __name__ == "__main__":
    train()
