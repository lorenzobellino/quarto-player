# Free for personal or classroom use; see 'LICENSE.md' for details.
# https://github.com/squillero/computational-intelligence

import logging
import argparse
import quarto
import sys
import itertools

from players import (
    DumbPlayer,
    RandomPlayer,
    HumanPlayer,
    GAPlayer,
    RuleBasedPlayer,
    TrainedGAPlayer,
    MinMaxPlayer,
    MixedStrategyPlayer,
    TrainedRLPlayer,
    MixedStrategyRL,
)
from bestPlayer import S309413


def main():
    win = 0
    N_GAMES = 1000

    # players = [
    #     DumbPlayer,
    #     RandomPlayer,
    #     RuleBasedPlayer,
    #     TrainedGAPlayer,
    #     MixedStrategyPlayer,
    #     TrainedRLPlayer,
    #     MixedStrategyRL,
    # ]

    # players = [
    #     DumbPlayer,
    #     RandomPlayer,
    #     RuleBasedPlayer,
    #     TrainedGAPlayer,
    #     TrainedRLPlayer,
    # ]
    players = [S309413, TrainedGAPlayer, RandomPlayer, DumbPlayer, RuleBasedPlayer, TrainedRLPlayer]

    # for p1, p2 in itertools.product(players, repeat=2):
    for p1, p2 in itertools.combinations(players, 2):

        print(f"evaluating {p1.__name__} against {p2.__name__} for {N_GAMES} games")
        print(f"game {0} / {N_GAMES} -> win : {0}/{N_GAMES}", end="")
        game = quarto.Quarto()
        win = 0
        for i in range(N_GAMES):
            game.reset()
            if i % 2 == 0:
                game.set_players((p1(game), p2(game)))
            else:
                game.set_players((p2(game), p1(game)))
            winner = game.run()
            if i % 2 == winner:
                win += 1
            sys.stdout.flush()
            print(f"\rgame {i} / {N_GAMES} -> win : {win}/{N_GAMES}", end="")
        print("\n")
        logging.warning(f"winratio :\n{p1.__name__} -> {win/N_GAMES}\n{p2.__name__} -> {(N_GAMES-win)/N_GAMES}\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="count", default=0, help="increase log verbosity")
    parser.add_argument(
        "-d", "--debug", action="store_const", dest="verbose", const=2, help="log debug messages (same as -vv)"
    )
    args = parser.parse_args()

    if args.verbose == 0:
        logging.getLogger().setLevel(level=logging.WARNING)
    elif args.verbose == 1:
        logging.getLogger().setLevel(level=logging.INFO)
    elif args.verbose == 2:
        logging.getLogger().setLevel(level=logging.DEBUG)

    main()
