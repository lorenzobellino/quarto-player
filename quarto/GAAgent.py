import random
import logging
import json

import quarto
from players import DumbPlayer, RandomPlayer, GAPlayer, RuleBasedPlayer

logging.getLogger().setLevel(logging.DEBUG)

IMPORT_POPULATION = False
NUM_MATCHES = 25
POPULATION = 20
NUM_GENERATIONS = 50
OFFSPRING = 10
LOG_FREQ = 10
POPULATION_FILE = "populations/population_GA_v3.json"


def fitness(genome):
    game = quarto.Quarto()
    agent = GAPlayer(game)
    agent.set_genome(genome)

    opponent = RandomPlayer(game)
    random_eval = evaluate(game, agent, opponent, NUM_MATCHES)
    game.reset()

    opponent = DumbPlayer(game)
    dumb_eval = evaluate(game, agent, opponent, NUM_MATCHES)
    game.reset()

    opponent = RuleBasedPlayer(game)
    rule_eval = evaluate(game, agent, opponent, NUM_MATCHES)

    return (rule_eval, random_eval, dumb_eval)


def generate_population(dim: int) -> list:
    r = []
    for _ in range(dim):
        genome = {
            "alpha": random.uniform(-10, 10),
            "beta": random.uniform(-10, 10),
            "gamma": random.uniform(-10, 10),
            "delta": random.uniform(-10, 10),
            "epsilon": random.uniform(-10, 10),
        }
        fit = fitness(genome)
        r.append((fit, genome))
    return r


def tournament(population, tournament_size=5):
    return max(random.choices(population, k=tournament_size), key=lambda i: i[0])


def combine(population, offspring):
    population += offspring
    population = sorted(population, key=lambda i: i[0], reverse=True)[:POPULATION]
    return population


def generate_offspring(population: list, gen: int) -> list:
    offspring = list()
    for _ in range(OFFSPRING):
        p = tournament(population)

        p[1]["alpha"] += random.gauss(0, 20 / (gen + 1))
        p[1]["beta"] += random.gauss(0, 20 / (gen + 1))
        p[1]["gamma"] += random.gauss(0, 20 / (gen + 1))
        p[1]["delta"] += random.gauss(0, 20 / (gen + 1))
        p[1]["epsilon"] += random.gauss(0, 20 / (gen + 1))
        fit = fitness(p[1])
        offspring.append((fit, p[1]))

    return offspring


def GA():
    i = 0
    best_sol = None
    logging.info("Starting GA")
    if IMPORT_POPULATION:
        with open(POPULATION_FILE, "r") as f:
            pop = json.load(f)
            population = [(fitness(p["genome"]), p["genome"]) for p in pop.values()]
            logging.info(f"population imported")
    else:
        logging.info(f"generating population of {POPULATION} individuals")
        population = generate_population(POPULATION)
        logging.info(f"population generated")

    for _ in range(NUM_GENERATIONS):
        logging.info(f"Generation {_}")
        offspring = generate_offspring(population, _)
        population = combine(population, offspring)
        logging.debug(f"best genome: {population[0][1]}")
        logging.debug(f"best fitness: {population[0][0]}")
        if (_ + 1) % LOG_FREQ == 0:
            with open(f"populations/population_GA_v{i}.json", "w") as f:
                pop = {f"individual_{i:02}": {"fitness": p[0], "genome": p[1]} for i, p in enumerate(population)}
                json.dump(pop, f, indent=4)
                logging.info(f"saved population")
            i += 1

    best_sol = population[0][1]
    return best_sol


def evaluate(game: quarto.Quarto, player1: GAPlayer, player2: quarto.Player, n: int):
    """Evaluate the performance of a player against another player
    param:
        game: the game to be played
        player1: the first player
        player2: the second player
        n: the number of matches to be played
    return:
        the ratio of matches won by player1
    """
    win = 0
    for _ in range(n):
        game.reset()
        if _ % 2 == 0:
            game.set_players((player1, player2))
        else:
            game.set_players((player2, player1))
        winner = game.run()
        if _ % 2 == winner:
            win += 1
    return win / n


def training():
    best_genome = GA()
    game = quarto.Quarto()
    agentGen = GAPlayer(game)
    agentGen.set_genome(best_genome)
    result = evaluate(game, agentGen, RandomPlayer(game), NUM_MATCHES)
    logging.info(f"main: Winner ratio of GA: {result} -- RANDOM")
    game.reset()
    result = evaluate(game, agentGen, DumbPlayer(game), NUM_MATCHES)
    logging.info(f"main: Winner ratio of GA: {result} -- DUMB")
    game.reset()
    result = evaluate(game, agentGen, RuleBasedPlayer(game), NUM_MATCHES)
    logging.info(f"main: Winner ratio of GA: {result} -- RULE BASED")
    game.reset()


if __name__ == "__main__":
    training()
