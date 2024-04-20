# Work in progress implementation of a sudoku game solver using RL.
# Copyright (C) 2024 Cristian Cerasuolo

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

import pygame
from solver import Sudoku
import time
import math
import pandas as pd
from ast import literal_eval as listify
import random

from sudokuUI import *

pygame.init()

class Game:

    def __init__(self):
        self.initial_board = self._load_sudoku()
        self.game = Sudoku(self.initial_board)
        self.cells = create_cells()
        self.active_cell = None
        self.solve_rect = pygame.Rect(        
            BUFFER,
            HEIGHT-BUTTON_HEIGHT - BUTTON_BORDER*2 - BUFFER,
            BUTTON_WIDTH + BUTTON_BORDER*2,
            BUTTON_HEIGHT + BUTTON_BORDER*2
        )
        self.moves = 0
        self.threshold = MAX_MOVES

    def _load_sudoku(self):
        data = pd.read_csv(TRAIN_DATA_PATH)
        data = listify(data["data"][random.randint(0, len(data)-1)])
        sudoku = [[] for _ in range(9)]
        for i in range(9):
            for j in range(9):
                sudoku[i].append(data[i*9+j])
        return sudoku
                
    def _get_cell(self, x, y):
        '''Returns the cell from the cells list given x and y.'''
        return self.cells[x][y]

    def play_step(self, action):
        self.moves += 1
        reward = 0, ""
        game_over = (False, -1)

        # 1. Perform action
        x, y, num = action
        num = None if num == 0 else num
        active_cell = self._get_cell(x, y)
        old_val = self.game.board[active_cell.row][active_cell.col].value
        self.game.board[active_cell.row][active_cell.col].value = num

        if not self.game.is_editable(self.game.board[active_cell.row][active_cell.col]):
            # Here if the agent tries to fill a non-editable cell, aka an input cell
            print("VIOLATION")
            reward = VIOLATION_REWARD, "Violation"
            self.game.board[active_cell.row][active_cell.col].value = old_val
        else:
            if num is not None:
                if not self.game.check_move(active_cell, num):
                    # Here if the agent fills a location with an invalid number
                    print("VIOLATION")
                    reward = VIOLATION_REWARD, "Violation"
                    self.game.board[active_cell.row][active_cell.col].value = old_val
                else:
                    # Here if the cell is editable and the agents wants to fill with a number (!= None)
                    if old_val is None:
                        # Here if the cell was empty and the agent fills it with a valid number
                        print("PROGRESS")
                        # reward = PROGRESS_REWARD*math.log2(len(self.game.get_empty_cells())), "Progress" # Reward is logarithmic to the number of empty cells
                        reward = PROGRESS_REWARD, "Progress"
                    else:
                        # Here if the cell was filled and the agent replaces the number with a valid number
                        print("DUMB MOVE")
                        reward = CHANGE_REWARD, "Change"
            else:
                # Here if the agent removes a number from a cell that is editable
                if old_val is not None:
                    # Here if the cell was filled and the agent removes the number
                    print("NO PROGRESS")
                    # reward = NO_PROGRESS_REWARD*math.log2(len(self.game.get_empty_cells())*2), "No progress"
                    reward = NO_PROGRESS_REWARD, "No progress"
                else:
                    # Here if the cell was empty and the agent removes the number (nothing to remove)
                    print("DUMB MOVE")
                    reward = DUMB_MOVE_REWARD, "Dumb move"

        # 2. Update UI
        screen.fill(WHITE_COLOR)

        draw_board(self.active_cell, self.cells, self.game)

        # 3. Check if game is complete
        if not self.game.get_empty_cell():
            if self.game.check_sudoku():
                # Set the text
                font = pygame.font.Font(None, 36)
                text = font.render('Solved!', 1, GREEN_COLOR)
                textbox = text.get_rect(center=(self.solve_rect.center))
                screen.blit(text, textbox)

                reward = WIN_REWARD, "Win"
                game_over = (True, 1)

        # 4. Update screen
        pygame.display.flip()

        # 5. Check if game is lost
        if self.moves > self.threshold:
            print("Game over, you lose")
            reward = LOSS_REWARD, "Loss"
            game_over = (True, 0)

        return reward, game_over
    
    def get_state(self):
        board = self.game.get_board()
        board = [list(map(lambda x: 0 if x is None else x, col)) for col in board]
        return self.initial_board + board


# def visual_solve(game, cells):
#     '''Solves the game while giving a visual representation of what is being done.'''
#     # Get first empty cell
#     cell = game.get_empty_cell()

#     # Solve is complete if cell is False
#     if not cell:
#         return True

#     # Check each possible move
#     for val in range(1, 10):
#         # Allow game to quit when being solved
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 sys.exit()

#         # Place value in board
#         cell.value = val

#         # Outline cell being changed in RED_COLOR
#         screen.fill(WHITE_COLOR)
#         draw_board(None, cells, game)
#         cell_rect = cells[cell.row][cell.col]
#         pygame.draw.rect(screen, RED_COLOR, cell_rect, 5)
#         pygame.display.update([cell_rect])
#         time.sleep(0.05)

#         # Check if the value is a valid move
#         if not game.check_move(cell, val):
#             cell.value = None
#             continue

#         # If all recursive calls return True then board is solved
#         screen.fill(WHITE_COLOR)
#         pygame.draw.rect(screen, GREEN_COLOR, cell_rect, 5)
#         draw_board(None, cells, game)
#         pygame.display.update([cell_rect])
#         if visual_solve(game, cells):
#             return True

#         # Undo move is solve was unsuccessful
#         cell.value = None

#     # No moves were successful
#     screen.fill(WHITE_COLOR)
#     pygame.draw.rect(screen, WHITE_COLOR, cell_rect, 5)
#     draw_board(None, cells, game)
#     pygame.display.update([cell_rect])
#     return False

# def play():
#     '''Contains all the functionality for playing a game of Sudoku.'''

#     game = Sudoku(easy)
#     cells = create_cells()
#     active_cell = None
#     solve_rect = pygame.Rect(
#         BUFFER,
#         HEIGHT-BUTTON_HEIGHT - BUTTON_BORDER*2 - BUFFER,
#         BUTTON_WIDTH + BUTTON_BORDER*2,
#         BUTTON_HEIGHT + BUTTON_BORDER*2
#     )

#     while True:
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 sys.exit()

#             # Handle mouse click
#             if event.type == pygame.MOUSEBUTTONUP:
#                 mouse_pos = pygame.mouse.get_pos() #(x,y)

#                 # Reset button is pressed
#                 if reset_btn.collidepoint(mouse_pos):
#                     game.reset()

#                 # Solve button is pressed
#                 if solve_btn.collidepoint(mouse_pos):
#                     screen.fill(WHITE_COLOR)
#                     active_cell = None
#                     draw_board(active_cell, cells, game)
#                     reset_btn = draw_button(
#                         WIDTH - BUFFER - BUTTON_BORDER*2 - BUTTON_WIDTH,
#                         HEIGHT - BUTTON_HEIGHT - BUTTON_BORDER*2 - BUFFER,
#                         BUTTON_WIDTH,
#                         BUTTON_HEIGHT,
#                         BUTTON_BORDER,
#                         INACTIVE_BTN,
#                         BLACK_COLOR,
#                         'Reset'
#                     )
#                     solve_btn = draw_button(
#                         WIDTH - BUFFER*2 - BUTTON_BORDER*4 - BUTTON_WIDTH*2,
#                         HEIGHT - BUTTON_HEIGHT - BUTTON_BORDER*2 - BUFFER,
#                         BUTTON_WIDTH,
#                         BUTTON_HEIGHT,
#                         BUTTON_BORDER,
#                         INACTIVE_BTN,
#                         BLACK_COLOR,
#                         'Visual Solve'
#                     )
#                     pygame.display.flip()
#                     visual_solve(game, cells)

#                 # Test if point in any cell
#                 active_cell = None
#                 for row in cells:
#                     for cell in row:
#                         if cell.collidepoint(mouse_pos):
#                             active_cell = cell

#                 # Test if active cell is empty
#                 if active_cell and not game.board[active_cell.row][active_cell.col].editable:
#                     active_cell = None

#             # Handle key press
#             if event.type == pygame.KEYUP:
#                 if active_cell is not None:

#                     # Input number based on key press
#                     if event.key == pygame.K_0 or event.key == pygame.K_KP0:
#                         game.board[active_cell.row][active_cell.col].value = 0
#                     if event.key == pygame.K_1 or event.key == pygame.K_KP1:
#                         game.board[active_cell.row][active_cell.col].value = 1
#                     if event.key == pygame.K_2 or event.key == pygame.K_KP2:
#                         game.board[active_cell.row][active_cell.col].value = 2
#                     if event.key == pygame.K_3 or event.key == pygame.K_KP3:
#                         game.board[active_cell.row][active_cell.col].value = 3
#                     if event.key == pygame.K_4 or event.key == pygame.K_KP4:
#                         game.board[active_cell.row][active_cell.col].value = 4
#                     if event.key == pygame.K_5 or event.key == pygame.K_KP5:
#                         game.board[active_cell.row][active_cell.col].value = 5
#                     if event.key == pygame.K_6 or event.key == pygame.K_KP6:
#                         game.board[active_cell.row][active_cell.col].value = 6
#                     if event.key == pygame.K_7 or event.key == pygame.K_KP7:
#                         game.board[active_cell.row][active_cell.col].value = 7
#                     if event.key == pygame.K_8 or event.key == pygame.K_KP8:
#                         game.board[active_cell.row][active_cell.col].value = 8
#                     if event.key == pygame.K_9 or event.key == pygame.K_KP9:
#                         game.board[active_cell.row][active_cell.col].value = 9
#                     if event.key == pygame.K_BACKSPACE or event.key == pygame.K_DELETE:
#                         game.board[active_cell.row][active_cell.col].value = None

#         screen.fill(WHITE_COLOR)

#         # Draw board
#         draw_board(active_cell, cells, game)

#         # Create buttons
#         reset_btn = draw_button(
#             WIDTH - BUFFER - BUTTON_BORDER*2 - BUTTON_WIDTH,
#             HEIGHT - BUTTON_HEIGHT - BUTTON_BORDER*2 - BUFFER,
#             BUTTON_WIDTH,
#             BUTTON_HEIGHT,
#             BUTTON_BORDER,
#             INACTIVE_BTN,
#             BLACK_COLOR,
#             'Reset'
#         )
#         solve_btn = draw_button(
#             WIDTH - BUFFER*2 - BUTTON_BORDER*4 - BUTTON_WIDTH*2,
#             HEIGHT - BUTTON_HEIGHT - BUTTON_BORDER*2 - BUFFER,
#             BUTTON_WIDTH,
#             BUTTON_HEIGHT,
#             BUTTON_BORDER,
#             INACTIVE_BTN,
#             BLACK_COLOR,
#             'Visual Solve'
#         )

#         # Check if mouse over either button
#         if reset_btn.collidepoint(pygame.mouse.get_pos()):
#             reset_btn = draw_button(
#                 WIDTH - BUFFER - BUTTON_BORDER*2 - BUTTON_WIDTH,
#                 HEIGHT - BUTTON_HEIGHT - BUTTON_BORDER*2 - BUFFER,
#                 BUTTON_WIDTH,
#                 BUTTON_HEIGHT,
#                 BUTTON_BORDER,
#                 ACTIVE_BTN,
#                 BLACK_COLOR,
#                 'Reset'
#             )
#         if solve_btn.collidepoint(pygame.mouse.get_pos()):
#             solve_btn = draw_button(
#                 WIDTH - BUFFER*2 - BUTTON_BORDER*4 - BUTTON_WIDTH*2,
#                 HEIGHT - BUTTON_HEIGHT - BUTTON_BORDER*2 - BUFFER,
#                 BUTTON_WIDTH,
#                 BUTTON_HEIGHT,
#                 BUTTON_BORDER,
#                 ACTIVE_BTN,
#                 BLACK_COLOR,
#                 'Visual Solve'
#             )

#         # Check if game is complete
#         if not game.get_empty_cell():
#             if check_sudoku(game):
#                 # Set the text
#                 font = pygame.font.Font(None, 36)
#                 text = font.render('Solved!', 1, GREEN_COLOR)
#                 textbox = text.get_rect(center=(solve_rect.center))
#                 screen.blit(text, textbox)

#         # Update screen
#         pygame.display.flip()


if __name__ == '__main__':
    
    game = Game()
    print(game.play_step((0, 3, 1)))
    import time
    time.sleep(1000)
