import logging
import argparse
import quarto
import sys

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


def main(player1, player2, N_GAMES):
    win = 0

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
        print(f"\rgame {i+1} / {N_GAMES} -> win : {win}/{N_GAMES}", end="")
    print("\n")
    logging.warning(f"winratio :\n{p1.__name__} -> {win/N_GAMES}\n{p2.__name__} -> {(N_GAMES-win)/N_GAMES}\n\n")


if __name__ == "__main__":
    players = {
        "S309413": S309413,
        "GA": TrainedGAPlayer,
        "Random": RandomPlayer,
        "Dumb": DumbPlayer,
        "RuleBased": RuleBasedPlayer,
        "RL": TrainedRLPlayer,
        "Human": HumanPlayer,
        "MixedStrategy": MixedStrategyPlayer,
        "MixedRL": MixedStrategyRL,
    }
    parser = argparse.ArgumentParser(description="Quarto player: choose two players and play a game")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="increase log verbosity")
    parser.add_argument(
        "-d", "--debug", action="store_const", dest="verbose", const=2, help="log debug messages (same as -vv)"
    )
    parser.add_argument(
        "player1",
        type=str,
        choices=[
            "S309413",
            "GA",
            "Random",
            "Dumb",
            "RuleBased",
            "RL",
            "Human",
            "MixedStrategy",
            "MixedRL",
        ],
        help="first player",
    )
    parser.add_argument(
        "player2",
        type=str,
        choices=[
            "S309413",
            "GA",
            "Random",
            "Dumb",
            "RuleBased",
            "RL",
            "Human",
            "MixedStrategy",
            "MixedRL",
        ],
        help="first player",
    )
    parser.add_argument("N", type=int, nargs="?", default=1, help="number of games to be played - default 1")
    args = parser.parse_args()
    if args.verbose == 0:
        logging.getLogger().setLevel(level=logging.WARNING)
    elif args.verbose == 1:
        logging.getLogger().setLevel(level=logging.INFO)
    elif args.verbose == 2:
        logging.getLogger().setLevel(level=logging.DEBUG)
    try:
        p1 = players[args.player1]
        p2 = players[args.player2]
    except KeyError:
        logging.info("Unknown player")
        logging.info("Available players are:")
        for p in players:
            print(f"{p}\n")
        sys.exit(1)
    main(player1=p1, player2=p2, N_GAMES=args.N)
