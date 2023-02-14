# Free for personal or classroom use; see 'LICENSE.md' for details.
# https://github.com/squillero/computational-intelligence

import logging
import argparse
import random
import quarto

from players import (
    DumbPlayer,
    RandomPlayer,
    HumanPlayer,
    GAPlayer,
    RuleBasedPlayer,
    TrainedGAPlayer,
    MinMaxPlayer,
    MixedStrategyPlayer,
)


class RandomPlayer(quarto.Player):
    """Random player"""

    def __init__(self, quarto: quarto.Quarto) -> None:
        super().__init__(quarto)

    def choose_piece(self) -> int:
        return random.randint(0, 15)

    def place_piece(self) -> tuple[int, int]:
        return random.randint(0, 3), random.randint(0, 3)


def main():
    win = 0
    N_GAMES = 100

    # players = (RandomPlayer, RuleBasedPlayer)
    players = (TrainedGAPlayer, RuleBasedPlayer)
    # players = (MinMaxPlayer, RuleBasedPlayer)
    # players = (TrainedGAPlayer)
    players = (RandomPlayer, MinMaxPlayer)
    players = (RandomPlayer, MixedStrategyPlayer)
    for i in range(N_GAMES):

        game = quarto.Quarto()
        if i % 2 == 0:
            game.set_players((players[0](game), players[1](game)))
            # game.set_players((RandomPlayer(game), RuleBasedPlayer(game)))
        else:
            game.set_players((players[1](game), players[0](game)))
            # game.set_players((RuleBasedPlayer(game), RandomPlayer(game)))
        winner = game.run()
        if i % 2 == 0 and winner == 1 or i % 2 == 1 and winner == 0:
            win += 1
            logging.warning(f"rulebased has won game {i}")
    logging.warning(f"winratio = {win/N_GAMES}")
    # game = quarto.Quarto()
    # game.set_players((RandomPlayer(game), RuleBasedPlayer(game)))
    # winner = game.run()
    # logging.warning(f"main: Winner: player {winner}")


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
