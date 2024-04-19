MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.01

PROGRESS_REWARD = 100 # The agent fills an empty location with a valid number
NO_PROGRESS_REWARD = -1 # The agent removes a number from a location that was previously filled but with a valid number
VIOLATION_REWARD = -1000 # The agent fills an empty location with an invalid number
LOSS_REWARD = -2000 # After a certain number of moves, the agent has not solved the puzzle
WIN_REWARD = 1000 # The agent has solved the puzzle
DUMB_MOVE_REWARD = -10 # The agent makes a move that is stupid

TRAIN_DATA_PATH = "../data/train.txt"

# Set SIZE of game and other constants
CELL_SIZE = 50
MINOR_GRID_SIZE = 1
MAJOR_GRID_SIZE = 3
BUFFER = 5
BUTTON_HEIGHT = 50
BUTTON_WIDTH = 125
BUTTON_BORDER = 2
WIDTH = CELL_SIZE*9 + MINOR_GRID_SIZE*6 + MAJOR_GRID_SIZE*4 + BUFFER*2
HEIGHT = CELL_SIZE*9 + MINOR_GRID_SIZE*6 + \
    MAJOR_GRID_SIZE*4 + BUTTON_HEIGHT + BUFFER*3 + BUTTON_BORDER*2
SIZE = WIDTH, HEIGHT
WHITE_COLOR = 255, 255, 255
BLACK_COLOR = 0, 0, 0
GRAY_COLOR = 200, 200, 200
GREEN_COLOR = 0, 175, 0
RED_COLOR = 200, 0, 0
INACTIVE_BTN = 51, 255, 255
ACTIVE_BTN = 51, 153, 255
