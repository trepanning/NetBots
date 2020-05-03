'''
NetBots.py

2D bots learn how to navigate towards food without 
being explicitly told how to do so. They use basic
fully-connected feed forward neural networks trained
with a simple evolution algorithm. Bots that capture
more food have a higher chance to pass on their traits.

The networks are given these values as inputs: 
- the coordinates of the closest food
- their own position in the world

The networks are trained with a basic evolution algorithm:
- roulette wheel selection of parents
- random crossover of parent genes
- random mutation

At the end of training, a window will open showing
the final generation. Sometimes more than one run
is needed for decent results.
'''

import math, random
import tkinter as tk

import NeuralNet

WIDTH = 800
HEIGHT = 600

BACKGROUND_COLOR = '#000000'
BOT_COLOR = '#ffffff'
BOT_SIZE = 8
BOT_SIZE_HALF = round(BOT_SIZE / 2)
FOOD_COLOR = '#00ff00'
FOOD_SIZE = 6
FOOD_SIZE_HALF = round(FOOD_SIZE / 2)
CLOSEST_FOOD_COLOR = '#0099ff'
DRAW_TARGET_LINES = True

# target FPS when running simulation with graphics
SIMULATION_RATE = 60
# used to set timer for next call to update
RATE_IN_MS = round((1 / SIMULATION_RATE) * 1000)

# how many generations of bots to evolve
TRAINING_EPOCH = 30
# how many frames to run the simulation for each generation
EPOCH_LENGTH = 1500
# how many bots are in each generation
POPULATION = 30
# how many active food pickups are available at a time
FOOD_SUPPLY = 10
# used to give each food pellet a unique id
FOOD_COUNTER = 0
# used to give each bot a unique id
BOT_COUNTER = 0
MIN_DISTANCE_TO_CAPTURE = BOT_SIZE_HALF + FOOD_SIZE_HALF
# number of values going in to a network
NETWORK_INPUTS = 4
# number of values coming out of a network
NETWORK_OUTPUTS = 2

# the number of hidden layers and neurons in each layer.
# does not include output layer.
# [6, 4] would translate to two hidden layers of 6 and 4 neurons each
# default is [6], one hidden layer with six neurons
# [0] would exclude the hidden layer completely (only output layer)
NETWORK_LAYERS = [6]

'''
available activation functions:

logistic
tanh
relu
leakyrelu
relu6
leakyrelu6
'''
NETWORK_HIDDEN_LAYER_AF = 'leakyrelu6'
NETWORK_OUTPUT_LAYER_AF = 'tanh'
EVOLUTION_MUTATION_RATE = 0.05

def distance(x1, y1, x2, y2):
    return math.sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2))

class Entity():
    def __init__(self):
        self.x = random.randint(100, WIDTH - 100)
        self.y = random.randint(100, HEIGHT - 100)
        self.img = None

class Bot(Entity):
    def __init__(self, n = NeuralNet.NeuralNetwork(\
        NETWORK_INPUTS, \
        NETWORK_OUTPUTS, \
        NETWORK_LAYERS, \
        NETWORK_HIDDEN_LAYER_AF, \
        NETWORK_OUTPUT_LAYER_AF)):
        Entity.__init__(self)
        self.nn = n
        self.score = 0
        self.target = None
        global BOT_COUNTER
        self.id = str(BOT_COUNTER)
        BOT_COUNTER += 1
        self.food_line = None
        self.dx = 0
        self.dy = 0

    def update(self, food):
        v = distance(food.x, food.y, self.x, self.y)
        x1 = (self.x - food.x) / v
        y1 = (self.y - food.y) / v
        x2 = self.x / WIDTH
        y2 = self.y / HEIGHT
        self.dx, self.dy = self.nn.feed_forward([x1, y1, x2, y2])
        self.x += self.dx
        self.y += self.dy

class Food(Entity):
    def __init__(self):
        global FOOD_COUNTER
        Entity.__init__(self)
        self.id = str(FOOD_COUNTER)
        FOOD_COUNTER += 1

class BotWorld():
    def __init__(self, bots):
        self.bots = bots
        self.food = {}
        for _ in range(FOOD_SUPPLY):
            food = Food()
            self.food[food.id] = food
        self.update_food_img = None
        self.update_bot_img = None

    # when a food pellet is captured, recycle its image,
    #   create a new food pellet somewhere else,
    #   and update the image
    def replace_food(self, f):
        food = Food()
        food.img = self.food[f].img
        self.food[food.id] = food
        if self.update_food_img:
            self.update_food_img(food)
        self.food.pop(f)

    # run a single frame of the simulation
    def update(self, length = 1):
        n = 0
        while n < length:
            for b in self.bots:
                bot = self.bots[b]
                mindist = None
                best_target = None
                for food in self.food:
                    f = self.food[food]
                    dist = distance(f.x, f.y, bot.x, bot.y)
                    if mindist == None or dist < mindist:
                        mindist = dist
                        best_target = food
                bot.target = best_target
                if distance(self.food[bot.target].x, \
                    self.food[bot.target].y, bot.x, bot.y) \
                    < MIN_DISTANCE_TO_CAPTURE:
                    self.replace_food(bot.target)
                    bot.score += 1
                else:
                    bot.update(self.food[bot.target])
                    if self.update_bot_img:
                        self.update_bot_img(bot)
            n += 1

# encapsulates bot training
class EvolutionAlgorithm():
    def __init__(self):
        self.bots = {}
        self.avg_score = 0
        self.high_score = 0
        self.mutations = 0
        self.elite_bot = {}
        for _ in range(POPULATION):
            bot = Bot()
            self.bots[bot.id] = bot

    def train(self):
        print('Training some bots...')
        for x in range(TRAINING_EPOCH):
            print('Generation: ' \
                + str(x + 1) + '/' \
                + str(TRAINING_EPOCH) \
                + ', Mutations: ' + str(self.mutations) \
                + ', Avg. Score: ' \
                + str(self.avg_score) \
                + ', High Score: ' \
                + str(self.high_score), \
                end = '          \r')
            self.avg_score = 0
            world = BotWorld(self.bots)
            # run the simulation for EPOCH_LENGTH time steps
            world.update(EPOCH_LENGTH)

            # create a roulette wheel for bot pairing
            roulette = []
            for bot in self.bots:
                # each bot has a chance, even if they captured no food
                roulette += [bot] * (self.bots[bot].score + 1)
                self.avg_score += self.bots[bot].score
                if self.bots[bot].score > self.high_score:
                    self.high_score = self.bots[bot].score
                    self.elite_bot = {bot : self.bots[bot]}

            # calculate the average score for this generation
            self.avg_score = round(self.avg_score / len(self.bots), 2)
            newbots = {}
            for _ in range(POPULATION):
                # choose two random parents from the roulette wheel
                parent_a = random.choice(roulette)
                parent_b = random.choice(roulette)
                # get the first parent's weights as a list
                parent_a_weights = self.bots[parent_a].nn.encoded()
                x = 0
                # make sure parents are different
                while parent_a == parent_b and x < POPULATION:
                    parent_b = random.choice(roulette)
                    x += 1
                parent_b_weights = self.bots[parent_b].nn.encoded()
                # pick a random point to cross over parent a and b
                crossover = random.randint(1, len(parent_a_weights) - 1)
                # randomly choose a parent for first segment of genes
                if random.randint(0, 1):
                    crossed_weights = parent_a_weights[:crossover]
                    crossed_weights += parent_b_weights[crossover:]
                else:
                    crossed_weights = parent_b_weights[:crossover]
                    crossed_weights += parent_a_weights[crossover:]
                n = NeuralNet.NeuralNetwork(\
                    NETWORK_INPUTS, \
                    NETWORK_OUTPUTS, \
                    NETWORK_LAYERS, \
                    NETWORK_HIDDEN_LAYER_AF, \
                    NETWORK_OUTPUT_LAYER_AF)
                # randomly mutate a weight
                if random.random() < EVOLUTION_MUTATION_RATE:
                    crossed_weights[random.randint( \
                        0, len(crossed_weights) - 1)] = \
                            random.uniform(-1, 1)
                    self.mutations += 1
                # create new bot, add it to the next generation's population
                n.decode(crossed_weights)
                bot = Bot(n)
                newbots[bot.id] = bot
            self.bots = newbots
        print()

# create window to show a running simulation
class BotsWindow():
    def __init__(self, evo):
        self._tk = tk.Tk()
        self._tk.title('NetBots')
        self._tk.resizable(False, False)
        self._tk.geometry('%dx%d+%d+%d' % (WIDTH, HEIGHT, WIDTH, HEIGHT))
        self.canvas = tk.Canvas(\
            self._tk, \
            width = WIDTH, \
            height = HEIGHT, \
            bg = BACKGROUND_COLOR, \
            bd = 0, highlightthickness = 0, relief = 'ridge')
        self.canvas.grid()
        self.world = BotWorld(evo.elite_bot)
        self.world.update_food_img = self.update_food_img
        self.world.update_bot_img = self.update_bot_img

        #create graphics for bots and target lines
        for b in self.world.bots:
            bot = self.world.bots[b]
            if DRAW_TARGET_LINES:
                bot.food_line = self.canvas.create_line( \
                    0, 0, 0, 0, fill = CLOSEST_FOOD_COLOR)
            bot.img = self.canvas.create_rectangle(\
                bot.x - BOT_SIZE_HALF, bot.y - BOT_SIZE_HALF, \
                bot.x + BOT_SIZE_HALF, bot.y + BOT_SIZE_HALF, \
                fill = BOT_COLOR)

        # create and position graphics for food
        for f in self.world.food:
            food = self.world.food[f]
            food.img = self.canvas.create_rectangle(\
                food.x - FOOD_SIZE_HALF, food.y - FOOD_SIZE_HALF, \
                food.x + FOOD_SIZE_HALF, food.y + FOOD_SIZE_HALF, \
                fill = FOOD_COLOR)

    # move food graphics
    def update_food_img(self, food):
        self.canvas.coords(food.img, \
            food.x - FOOD_SIZE_HALF, food.y - FOOD_SIZE_HALF, \
            food.x + FOOD_SIZE_HALF, food.y + FOOD_SIZE_HALF)

    # update bot graphics and their target lines
    def update_bot_img(self, bot):
        self.canvas.move(bot.img, bot.dx, bot.dy)
        if DRAW_TARGET_LINES:
            self.canvas.coords(bot.food_line, \
                bot.x, bot.y, \
                self.world.food[bot.target].x, \
                self.world.food[bot.target].y)

    def update(self):
        self.world.update()
        self._tk.after(RATE_IN_MS, self.update)


evo = EvolutionAlgorithm()
evo.train()
w = BotsWindow(evo)
w.update()
w._tk.mainloop()