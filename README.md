# Creating an Agent able to play the game of QUARTO 

## Introduction 

### What is QUARTO? 
The game of quarto consist of a board with **16 squares** and **16 pieces**. Each piece has **4 attributes** (height, color, shape, and filling) and each attribute can be either **big** or **small**, **light** or **dark** , **squared** or **round** , **full** or **hollow**. 
The goal is to establish a line of four pieces ith at least one common characteristic (attribute) on the board.
Quarto is a turn based. impartial game.
The game start with *player1* choosing a piece to place on the board, then *player2* chooses where to place it and choose an ather piece to be placed on the board. The game continues until one of the players has established a line of four pieces with at least one common attribute.
## Play it yorself
You can play the game by running the file [play.py](https://github.com/lorenzobellino/quarto-player/blob/main/quarto/play.py) with the command
```bash
python play.py "player1" "player2" "N"
```
where *player1* and *player2* are the names of the players you want to play against and *N* is the number of games that you want to play. If you don't specify a number of games, the game will be played only once. 
The available players are:
- *random* : a random player that choose a random piece and a random position on the board
- *dumb* : a player that choose the first piece that is not already on the board and place it on the first empty square on the board
- *human* : a player that ask the user to choose a piece and a position on the board
- *GA* : a player that use a genetic algorithm to choose the best piece and the best position on the board
- *RL* : a player that use a reinforcement learning algorithm to choose the best piece and the best position on the board
- *MixedStrategy* : a player that uses minmax and rule based strategies to choose the best piece and the best position on the board
- *MixedRL* : a player that uses minmax and reinforcement learning strategies to choose the best piece and the best position on the board
- *S309413* : my best player
- *RuleBased* : a player that uses rule based strategies to choose the best piece and the best position on the board

## create your own player
You can create your own player by creating a class that extends the class *quarto.Player* and implementing the methods *choose_piece* and *place_piece*.
boylerlate code for player:
```python
class MyPlayer(quarto.Player):
    """My player"""

    def __init__(self, quarto: quarto.Quarto) -> None:
        super().__init__(quarto)

    def choose_piece(self) -> int:
        # choose a piece
        return piece

    def place_piece(self) -> tuple[int, int]:
        # choose a position
        return position
```


## Analyzing the problem 
The first step that i took to search for a solution was to create a playable version of the player in order to sense the problem and to be able to test possible strategies by playing aganinst the random player already created.
The code for this player simply take in input a value from **0** to **15** representing the choosen piece, and **2** values from **0** to **2** representing the row and the column where the piece will be placed.
```python
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
```

After playng for some times i started to come up with different options for a strategy.

# Proposed Solutions <a name="Proposed-solutions"></a>

All of my attempt at creating a wiining player can be found in [platyers.py](https://github.com/lorenzobellino/quarto-player/blob/main/quarto/players.py) but now I will try to briefly explain the main idea behind each of them.

## Dumb Player 
This player is really simple but it was really usefull especially in the beggining phases of the progect because it allowed me to have an easbly beatable player to test my strategies againsT, particolarly when i was trying to implement a GA approach.
```python
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
```
When prompted to choose a piece, the player simply choose the first piece that is not already on the board. When prompted to place a piece, the player simply choose the first empty square on the board.

## GA approach 
The first real attempt at finding a winning strategy was to implement a Genetic Algorithm. Since i had great results with the GA implemented for the third Lab, i decided to try to use it to solve this problem.
The genome of the individuals consists of 5 values :
```python
genome = {
            "alpha": -12.228556866040943,
            "beta": 77.44867497850886,
            "gamma": 7.190937752438881,
            "delta": 10.629790618213386,
            "epsilon": 34.46747782969669
        }
```

This parameters are Then used to decide which piece to choose and where to place it on the board. *alpha* and *beta* are used to decide which piece to choose, while *gamma*, *delta* and *epsilon* are used to decide where to place it. 
To calculate the piece: 
```python
def choose_piece(self) -> int:
        data = cook_data_pieces(self)
        res = (
            (a[0], abs(self.genome["alpha"] * a[1] + self.genome["beta"] * b[1]))
            for a, b in zip(data["alpha"], data["beta"])
        )
        choosen_piece = min(res, key=lambda x: x[1])[0]
        return choosen_piece
```
where ```data``` is a dictionary calculated in the function ```cook_data_pieces``` where each **key** represents the result of a bitwise operation calculated on the available pieces. For each available piece i remove it from the available piece as if the player decided to choose it, then i calculate the sum of the remaining pieces and the bitwise nand of the remaining pieces. The result of this operation is then used as a **key** in the dictionary ```data```. The value associated with each key is a list of tuples where each tuple contains the piece and the result of the operation. The piece is then chosen by taking the piece with the lowest value of the operation:
```python
def cook_data_pieces(state: quarto.Quarto) -> dict:
    """provide usefull data for the genetic algorithm whe need to choose a piece"""
    data = {}
    status = state.get_game().get_board_status()
    used_pieces = {c for x in status for c in x if c > -1}
    usable_pieces = {x for x in range(16)} - used_pieces
    alpha = list()
    beta = list()
    if len(usable_pieces) == 1:
        # if there is only one piece left, it is the best choice
        return {"alpha": [(list(usable_pieces)[0], 1)], "beta": [(list(usable_pieces)[0], 1)]}
    for p in usable_pieces:
        usable_pieces.remove(p)
        # remove the selected piece from the usable pieces
        # calculate the sum of the remaining pieces
        *_, a = accumulate([_ for _ in usable_pieces], operator.add)
        usable_pieces.add(p)
        used_pieces.add(p)
        # add the selected piece to the used pieces
        # calculate the nand of the used pieces
        *_, b = accumulate(used_pieces, lambda x, y: ~(x & y))
        used_pieces.remove(p)
        alpha.append((p, a))
        beta.append((p, b))
    data["alpha"] = alpha
    data["beta"] = beta
    return data
```

To calculate the move:
```python
def place_piece(self) -> tuple[int, int]:
        data = cook_data_moves(self, self.get_game().get_selected_piece())
        res = (
            (g[0], abs(self.genome["gamma"] * g[1] + self.genome["delta"] * h[1] + self.genome["epsilon"] * i[1]))
            for g, h, i in zip(data["gamma"], data["delta"], data["epsilon"])
        )
        choosen_move = min(res, key=lambda x: x[1])[0]
        return choosen_move
```
In the same way as before ```data``` is a dictionary calculated in the function ```cook_data_moves``` where each **key** represents the result of a bitwise operation calculated on the available moves on the board. For each available move i remove it from the available moves as if the player decided to place the piece there, then i calculate the sum of the remaining moves, the bitwise and of the remaining moves and the bitwise or of the remaining moves. The result of this operation is then used as a **key** in the dictionary ```data```. The value associated with each key is a list of tuples where each tuple contains the move and the result of the operation. The move is then chosen by taking the move with the lowest value of the operation :
```python
def cook_data_moves(state: quarto.Quarto, piece: int) -> dict:
    """provide usefull data for the genetic algorithm whe need to choose a move"""
    data = {}
    status = state.get_game().get_board_status()
    possible_moves = [(c, r) for r in range(4) for c in range(4) if status[r][c] == -1]
    gamma = list()
    delta = list()
    epsilon = list()
    if len(possible_moves) == 1:
        # if there is only one move left, it is the best choice
        return {
            "gamma": [(possible_moves[0], 1)],
            "delta": [(possible_moves[0], 1)],
            "epsilon": [(possible_moves[0], 1)],
        }
    for p in possible_moves:
        # remove the selected move from the possible moves
        possible_moves.remove(p)
        # calculate the sum of the remaining moves
        # calculate the or of the remaining moves
        # calculate the and of the remaining moves
        *_, g = accumulate([x[0] for x in possible_moves], operator.add)
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
```

In order to evolve and explore different individuals the algorith is based on the following steps:
1. Generate a population of ```POPULATION``` individuals
2. Evaluate the fitness of each individual
3. select with a tournament of size ```k = 5 ``` an individual from the population
4. mutate the individual's genome
5. calculate the fitness of the mutated individual
6. repeat step 3 to 5 until ```OFFSPRING``` individuals are generated
7. select the best ```POPULATION``` individuals from the population and the offspring
8. repeat step 2 to 7 until the maximum number of generations is reached

All the code for the GA can be found in the file [GAAgent.py](https://github.com/lorenzobellino/quarto-player/tree/main/quarto/GAAgent.py) but here is a summary of the main functions:


Step 1:
```python
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
```
the genome is generate based on a random uniform distribution between -10 and 10

Step 2:
```python
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
```
The fitness is calculated by playing ```NUM_MATCHES``` against three different opponents: a random player, a dumb player and a rule based player. The fitness is then calculated as the average of the number of wins against each opponent. The fitness is then a tuple of three values where the first value is the fitness against the rule based player, the second value is the fitness against the random player and the third value is the fitness against the dumb player.
For each evaluation against an opponent half of the games are played as *player1* and the other half as *player2* in order to avoid bias.

Step 3:
```python
def tournament(population, tournament_size=5):
    return max(random.choices(population, k=tournament_size), key=lambda i: i[0])
```

Step 4 & 5 & 6:
```python
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
```
The genome is mutated by adding a random value sampled from a gaussian distribution with mean 0 and standard deviation 20/(gen+1) where gen is the current generation. The mutation is applied to each gene of the genome.

Step 7
```python
def combine(population, offspring):
    population += offspring
    population = sorted(population, key=lambda i: i[0], reverse=True)[:POPULATION]
    return population
```
Step 8:
```python
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
```
This is the core of the GA. In order to save particularly instresting genome and their fitness the population is saved every ```LOG_FREQ``` generations in a json file.

### Results 
This attempt unfortunatly was not successful. The best genome found was:
```json
"genome": {
            "alpha": -94.03014146974122,
            "beta": -107.17350875193313,
            "gamma": 152.6577141347451,
            "delta": -29.856838596915765,
            "epsilon": -12.095960806170313
        }
```
But It yelded pretty poor results against every player except the most simple one (the dumb player). Against the ```RandomPlayer``` the agent had a winrate between $50%$ and $60%$ and against the ```RuleBasedPlayer``` the winrate was between $20%$ and $30%$. The agent was able to beat the ```DumbPlayer``` with a winrate between $90%$ and $100%$.
This are the result of **1000** matches against each player:

|                 	| RuleBasedPlayer 	| RandomPlayer 	| DumbPlayer 	|
|-----------------	|-----------------	|--------------	|------------	|
| **TrainedGAPlayer** | 282              	| 580           	| 1000        	|

## Rule Based Player 

The player uses a set of hardcoded rules to decide which piece to play and where to play it.
```python
class RuleBasedPlayer(quarto.Player):
    def __init__(self, quarto: quarto.Quarto) -> None:
        super().__init__(quarto)
        self.previous_board = np.array([[-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1]])
        self.previous_move = None
        self.previous_piece = None

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
```
The deciosion of which piece to play is based on the following rules:

1. ```block_strategy_piece``` : the player can recognize when a piece is needed for the opponent to win and if possible will not hand it to player 2, effectively blocking the opponent from winning.
```python
def block_strategy_piece(self):
    """check if the are moves that can make the opponent win
    if so, return the piece that can block the move
    param:
        usable_pieces: list of pieces that can be used
        game: current game
    return:
        pieces: list of pieces that will not make the opponent win
    """
    winner = False
    current_board = self.get_game().get_board_status()
    used_pieces = {c for x in current_board for c in x if c > -1}
    usable_pieces = [_ for _ in {x for x in range(16)} - used_pieces]
    possible_moves = [(c, r) for r in range(4) for c in range(4) if current_board[r][c] == -1]
    pieces = {}
    for p in usable_pieces:
        winner = False
        for m in possible_moves:
            # if i choose a piece and the opponent can not make a winning move with that piece add it to the list
            board = deepcopy(self.get_game())

            board.select(p)
            board.place(m[0], m[1])
            if board.check_winner() > -1:
                winner = True
        if not winner:
            pieces[p] = m

    return pieces
```

2. ```mirror_strategy_piece```: based on the binary representation of the last piece placed on the board, the player will try to choose a piece that is as much different as possible from the last one. This is done by calculating the **Hamming distance** between the binary representation of the last piece and the binary representation of the possible pieces. The piece with the highest Hamming distance will be chosen.
```python
def mirror_strategy_piece(self):
    """mirror the choice of the opponent
    if the opponent choose a piece, i choose the most different one
    """
    piece = random.randint(0, 15)

    current_board = self.get_game().get_board_status()

    if self.previous_piece == None:
        piece = random.randint(0, 15)
        # if i am plain first, i choose a random piece for the first move
    else:
        piece_info = self.get_game().get_piece_charachteristics(self.previous_piece).binary
        used_pieces = {c for x in current_board for c in x if c > -1}
        usable_pieces = [_ for _ in {x for x in range(16)} - used_pieces]
        pieces = list()
        for p in usable_pieces:
            # for each usable pieces find the most different from the previous piece
            p_info = [int(x) for x in format(p, "04b")]
            r = sum([abs(x - y) for x, y in zip(p_info, piece_info)])
            pieces.append((p, r))
            if r == 4:
                piece = p
                break
        piece = max(pieces, key=lambda x: x[1])[0]
    return piece
```
The deciosion of where to place the piece is based on the following rules:
1. ```check_for_win```: the player can recognize when a move is needed for the player to win and if possible will play it, effectively winning the game.
```python
def check_for_win(gameboard):
    """Check if the gameboard has a winning move. If so, return the move"""
    move = None

    piece = gameboard.get_selected_piece()
    current_board = gameboard.get_board_status()
    possible_moves = [(c, r) for r in range(4) for c in range(4) if current_board[r][c] == -1]
    for m in possible_moves:
        game = deepcopy(gameboard)
        game.select(piece)
        game.place(m[0], m[1])
        if game.check_winner() > -1:
            move = m
            break
    # return the list that can not make the opponent win
    return move
```
2. ```mirror_strategy_move```: based on the last move played the player will try to place the piece in the mirroring position on the board.
```python

def mirror_strategy_move(self):
    """mirror the choice of the opponent
    if the opponent choose a move, i choose the opposite one if i can
    """
    self.previous_piece = self.get_game().get_selected_piece()
    current_board = self.get_game().get_board_status()
    # find the previous move by vomparing the current board with the previous one and find the first difference
    for i, r in enumerate(zip(self.previous_board, current_board)):
        for j, c in enumerate(zip(r[0], r[1])):
            if c[0] != c[1]:
                self.previous_move = (i, j)
                break
    if self.previous_move != None:
        # if there is a previous move, i choose the opposite one if i can
        possible_moves = [(c, r) for r in range(4) for c in range(4) if self.previous_board[r][c] == -1]
        move = (3 - self.previous_move[1], 3 - self.previous_move[0])
        if move not in possible_moves:
            # if the opposite move is not possible, i try to find a move that is as far as possible from the previous one
            # the distance is calculated usiong the manhattan distance
            manhattan = list()
            for m in possible_moves:
                v = sum(abs(x - y) for x, y in zip(move, self.previous_move))
                manhattan.append((m, v))
            move = max(manhattan, key=lambda x: x[1])[0]
    else:
        move = (random.randint(0, 3), random.randint(0, 3))
    self.previous_board = deepcopy(current_board)  # update the previous board
    self.previous_board[move[1]][move[0]] = self.previous_piece  # update the previous board with the selected move
    return move
```

### Results 
This player is really effective against the other player tested up to this point, this are the results after **1000** matches:

|                 	| TrainedGAPlayer 	| RandomPlayer 	| DumbPlayer 	|
|-----------------	|-----------------	|--------------	|------------	|
| **RuleBasedPlayer** | 718             | 958           	| 693       	    |


## MinMaxPlayer 
As the name suggest the next tecnique that i tried to implement was a minmax player with alpha beta pruning and a memory of previous states.
The Player class is defined as follows:
```python
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
```
The ```minmax``` function is defined as follows:
```python
@cache
def minmax(self, depth, alpha, beta, isMaximizing, last_move=None, last_piece=None, game=None):
    """Minmax to choise the best move to play and piece to use"""
    if (isMaximizing and game.check_winner() > -1) or depth == 0:
        # if winning position or max depth reached
        # evaluate the position and return the value
        evaluation = self.evaluate_board(isMaximizing, game, last_move, last_piece)
        self.memory[(isMaximizing, hash(str(game)))] = evaluation
        return evaluation
    if (isMaximizing, hash(str(game))) in self.memory:
        # if the state is already solved in the memory return the value
        return self.memory[(isMaximizing, hash(str(game)))]

    best_choice = None
    selected_piece = game.get_selected_piece()
    board = game.get_board_status()
    avvailable_piece = get_available_pieces(board, selected_piece)
    available_moves = get_available_moves(board)
    if isMaximizing:
        best_choice = (-math.inf, -1, -1)
        for m in available_moves:
            game_copy = deepcopy(game)
            game_copy.place(m[0], m[1])
            for p in avvailable_piece:
                # for each move and each possible piece minimize the opponent and maximize my score
                if not game_copy.select(p):
                    logging.debug(f"piece {p} not available")
                evaluation = minmax(self, depth - 1, alpha, beta, False, m, p, game_copy)
                best_choice = max(best_choice, evaluation, key=lambda x: x[0])
                alpha = max(alpha, best_choice[0])
                if beta <= alpha or best_choice[0] == 100:  # alpha beta pruning or winning position
                    break
            if best_choice[0] == 100:
                break
        return best_choice

    else:
        best_choice = (math.inf, -1, -1)
        for m in available_moves:
            game_copy = deepcopy(game)
            game_copy.place(m[0], m[1])
            for p in avvailable_piece:
                # for each move and each possible piece minimize the opponent and maximize my score
                if not game_copy.select(p):
                    logging.debug(f"piece {p} not available")
                evaluation = minmax(self, depth - 1, alpha, beta, True, m, p, game_copy)
                best_choice = min(best_choice, evaluation, key=lambda x: x[0])
                beta = min(beta, best_choice[0])
                if beta <= alpha or best_choice[0] == 0:  # alpha beta pruning or losing position for minimizer
                    break
            if best_choice[0] == 0:
                break
        return best_choice
```
The drawback of this player is that it is really slow and can only work with a limited depth.
The evaluation function is defined as follows:
```python
def evaluate_board(self, isMaximizing, game, last_move, last_piece):
        """Evaluate the board and return a value
        params:
            isMaximizing: True if the player is maximizing, False otherwise
            game: the game to evaluate
            last_move: the last move made
            last_piece: the last piece used
        return:
            a tuple containing the value of the board, the last move made and the last piece used
        """
        if game.check_winner() > -1:
            return (100, last_move, last_piece)

        usable_pieces = get_available_pieces(game.get_board_status())
        blocking_pieces = blocking_piece(usable_pieces, game)
        # the value is the percentage of blocking pieces if the player is maximizing
        # the value is the percentage of non blocking pieces if the player is minimizing
        if isMaximizing:
            v = len(blocking_pieces) * 100 / len(usable_pieces)
            return (v, last_move, last_piece)
        else:
            v = (len(usable_pieces) - len(blocking_pieces)) * 100 / len(usable_pieces)
            return (v, last_move, last_piece)
```
if the player is maximizin and the evaluating move is winnig the evaluation is 100, otherwise the evaluation is the number of blocking pieces divided by the number of available pieces. where blocking pieces are the pieces that can be used to block the opponent from winning.
```python
def blocking_piece(usable_pieces, game):
    """check if the are moves that can make the opponent win
    if so, return the piece that can block the move
    param:
        usable_pieces: list of pieces that can be used
        game: current game
    return:
        pieces: list of pieces that will not make the opponent win
    """
    winner = False
    pieces = {}
    possible_moves = [(c, r) for r in range(4) for c in range(4) if game.get_board_status()[r][c] == -1]
    for p in usable_pieces:
        winner = False
        for m in possible_moves:
            # if i choose a piece and the opponent can not make a winning move with that piece add it to the list
            board = deepcopy(game)
            board.select(p)
            board.place(m[0], m[1])
            if board.check_winner() > -1:
                winner = True
        if not winner:
            pieces[p] = m
    # return the list that can not make the opponent win
    return pieces
```
## RL Agent 
Since I wasn't able to implement a good minmax player I decided to try to implement a reinforcement learning agent. 
The core of the algorithm is expressed in this function:
```python
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
```
The agent is trained against an opponent for a large number of matches and after every match the agent will update his policy table based on the outcome of the match.
The Agent's class is defined as follows:
```python
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
        choiche = ...
        ...

        return choice

    def place_piece(self) -> tuple[int, int]:
        choiche = ...
        ...

        return choice

    def learn(self, win: bool) -> None:
        for board, value in self.current_state.items():
            
            update_policy_table 
            ...


        self.current_state = dict()
        self.randomness -= self.learning_rate
```

The policy table is defined as a dictionary where the key is the board status and the value is an othe dicvtionary with **2** keys `piece` and `move` and for each key the value is a list of tuple where the first element is the probability of choosing that piece or move and the second element is the value of the move or piece.
In order to update the policy table an history of the choosen pieces and moves is kept in the `current_state` dictionary which is updated in the `choose_piece` and `place_piece` methods as follows:

```python
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
            else:
                choice = random.choice(available_pieces)

        return choice
```
```python
def place_piece(self) -> tuple[int, int]:
        board = self.get_game().get_board_status()
        available_moves = get_available_moves(board)
        if self.is_learning and random.random() < self.randomness:
            choice = random.choice(available_moves)
            self.current_state[hash(str(board))] = {"move": choice}
        else:
            if hash(str(board)) in self.G:
                try:
                    possible_chioce = self.G[hash(str(board))]["move"]
                    choice = max(possible_chioce, key=possible_chioce[1])
                except:
                    choice = random.choice(available_moves)
            else:
                choice = random.choice(available_moves)
        return choice
```
The agent was trained for **5000** matches against different opponents and the obtained policy saved in a file to be used with a trained player.

### Results 
Herre we can see the results for **1000** matches between every opponents. 

|           vs  	    | TrainedGAPlayer 	| RandomPlayer 	| DumbPlayer 	| RuleBasedPlayer 	| TrainedRLPlayer 	|
|-----------------	    |-----------------	|--------------	|---------------|-----------------  |------------------ |
| **TrainedGAPlayer**   |                   | 557           | 1000       	| 292               | 576               |
| **RandomPlayer**      | 443               |               | 386       	| 53                | 473               |
| **DumbPlayer**        | 0                 | 614           |           	| 393               | 576               |
| **RuleBasedPlayer**   | 718               | 947           | 607       	|                   | 958               |
| **TrainedRLPlayer**   | 424               | 527           | 424       	| 42                |                   |

## Mixing strategies
In order to improve the performance of the agent I decided to mix the strategies of the different players. The first Player that i created is based on the ```RuleBasedPlayer``` and the ```MinimaxPlayer```. Since minmax is really slow, especially for the first moves, I decided to implement a player that at first used a fixed set of rules but towards the end of the game switched to search for optimal solution using minmax. In the latest imlementation the minmax functionality is activated only when the number of available pieces is halfed. 
The two main functions of the player are as folows:
```python 
def choose_piece(self) -> int:
        if self.minmax_piece is not None:
            piece = self.minmax_piece
            self.minmax_piece = None
            return piece

        pieces = block_strategy_piece(self)
        if len(pieces) < 3 and len(pieces) > 0:
            return list(pieces.keys())[0]
        piece = mirror_strategy_piece(self)
        return piece
```
In this function if the minmax is already been activated the function returns the piece choosen by minmax. Otherwise it checks if there is a piece that can block the opponent from winning and if there is it returns that piece. Finally if there is no such piece it checks if there is a piece that can be used to mirror the opponent's piece.

```python
    
def place_piece(self) -> tuple[int, int]:
    move = check_for_win(self.get_game())
    if move is not None:
        return move

    usable_pieces = get_available_pieces(self.get_game().get_board_status())
    if len(usable_pieces) < 8:
        game = deepcopy(self.get_game())
        value, move, piece = minmax(self, depth=4, alpha=-math.inf, beta=math.inf, isMaximizing=True, game=game)
        if piece != -1 and value > 0:
            self.minmax_piece = piece
            self.previous_board[move[0]][move[1]] = self.get_game().get_selected_piece()
            self.previous_piece = piece
            return move
    move = mirror_strategy_move(self)

    return move
```
In this function the first thing that the player will do is to checkl wether there is a move that can be used to win the game. If there is it returns that move. Otherwise it checks if the number of available pieces is less than 8 and if it is it activates the minmax functionality. If the minmax functionality is activated it returns the move choosen by minmax. Finally if the minmax functionality is not activated it returns the move choosen by the mirror strategy.

### Results
todo

## Mixed strategies with RL policy
Similarly to what I did with the other mixed strategy player I tried to incorporate the policy learned by the RL agent in order to make a better choiche in the first stages of the game.
The two main functions are as follows:
```python
    def choose_piece(self) -> int:
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
```
```python
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
    return choice
```
In this function the player first checks if the minmax functionality is activated and if it is it returns the piece choosen by minmax. Otherwise it checks if the RL agent has a policy for the current state and if it does it returns the piece choosen by the RL agent : ```choose_piece_rl(self)```. If the RL agent does not have a policy for the current state it checks if there is a piece that can be used to mirror the opponent's piece and if there is it returns that piece. Finally if there is no such piece it returns a random piece.

```python
    def place_piece(self) -> tuple[int, int]:
        winning_move = check_for_win(self.get_game())
        if winning_move is not None:
            return winning_move

        usable_pieces = get_available_pieces(self.get_game().get_board_status())

        if len(usable_pieces) < 7:
            game = deepcopy(self.get_game())
            value, minmax_move, minmax_piece = minmax(
                self, depth=4, alpha=-math.inf, beta=math.inf, isMaximizing=True, game=game
            )
            if value > 60 and minmax_piece != -1 and minmax_move != (-1, -1):
                self.minmax_piece = minmax_piece
                self.previous_board[minmax_move[0]][minmax_move[1]] = self.get_game().get_selected_piece()
                self.previous_piece = minmax_piece
                return minmax_move
        rl_move = self.choose_move_rl()
        if rl_move is not None:
            return rl_move
        else:
            mirror_move = mirror_strategy_move(self)
            if mirror_move is not None:
                return mirror_move
            else:
                return random.choice(get_available_moves(self.get_game().get_board_status()))
```
Similarly the second function at first check if there is a move that can be used to win the game. If there is it returns that move. Otherwise it checks if the number of available pieces is less than 7 and if it is it activates the minmax functionality. If the minmax functionality is activated it returns the move choosen by minmax. Otherwise it checks if the RL agent has a policy for the current state and if it does it returns the move choosen by the RL agent. If the RL agent does not have a policy for the current state it checks if there is a move that can be used to mirror the opponent's piece and if there is it returns that move. Finally if there is no such move it returns a random move.
```python
def choose_move_rl(self):
    board = self.get_game().get_board_status()
    choice = None
    if hash(str(board)) in self.G:
        try:
            possible_chioce = self.G[hash(str(board))]["move"]
            choice = max(possible_chioce, key=possible_chioce[1])
        except:
            choice = None
    return choice
``` 

this function simply search in the RLpolicy if there is a policy for the current state and if there is it returns the piece choosen by the RL agent.

The resuls for this player are not improving much if at all compared to the other mixed strategy player. This is probably due to the fact that the RL agent is not learning much in the first stages of the game and therefore the policy is not very good.

# My best player (S309413)

In the best player that i could find and implement is the first mixed strategy player that uses minmax, mirroring strategy and blocking strategy in order to choose the piece and the move.
The code for this player can be found in the file [bestPlayer.py](https://github.com/lorenzobellino/quarto-player/blob/main/quarto/bestPlayer.py)
but her is also a copy :
```python
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
```

### Results
This are the result for y best player compare to every other opponent tested for **1000** games, playing first for 500 and second for 500

|           vs  	    | TrainedGAPlayer 	| RandomPlayer 	| DumbPlayer 	| RuleBasedPlayer 	| TrainedRLPlayer 	| 
|-----------------	    |-----------------	|--------------	|---------------|-----------------  |------------------ |
| **S309413**           |       692         |      999      |      645      |        593        |       958         |
