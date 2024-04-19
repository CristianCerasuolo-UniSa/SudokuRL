import pygame
from constants import *

screen = pygame.display.set_mode(SIZE)
pygame.display.set_caption('Sudoku')

class Cell:
    '''Represents a cell within a game of Sudoku.'''

    def __init__(self, row, col, value, editable):
        '''Initializes an instance of a Sudoku cell.'''
        self.row = row
        self.col = col
        self.value = value
        self._editable = editable

    @property
    def row(self):
        '''Getter method for row.'''
        return self._row

    @row.setter
    def row(self, row):
        '''Setter method for row.'''
        if row < 0 or row > 8:
            raise AttributeError('Row must be between 0 and 8.')
        else:
            self._row = row

    @property
    def col(self):
        '''Getter method for col.'''
        return self._col

    @col.setter
    def col(self, col):
        '''Setter method for col.'''
        if col < 0 or col > 8:
            raise AttributeError('Col must be between 0 and 8.')
        else:
            self._col = col

    @property
    def value(self):
        '''Getter method for value.'''
        return self._value

    @property
    def editable(self):
        '''Getter method for editable.'''
        return self._editable

    def __repr__(self):
        return f'{self.__class__.__name__}({self.value})'

    @value.setter
    def value(self, value):
        '''Setter method for value.'''
        if value is not None and (value < 1 or value > 9):
            raise AttributeError('Value must be between 1 and 9.')
        else:
            self._value = value


class RectCell(pygame.Rect):
    '''
    A class built upon the pygame Rect class used to represent individual cells in the game.
    This class has a few extra attributes not contained within the base Rect class.
    '''

    def __init__(self, left, top, row, col):
        super().__init__(left, top, CELL_SIZE, CELL_SIZE)
        self.row = row
        self.col = col


def create_cells():
    '''Creates all 81 cells with RectCell class.'''
    cells = [[] for _ in range(9)]

    # Set attributes for for first RectCell
    row = 0
    col = 0
    left = BUFFER + MAJOR_GRID_SIZE
    top = BUFFER + MAJOR_GRID_SIZE

    while row < 9:
        while col < 9:
            cells[row].append(RectCell(left, top, row, col))

            # Update attributes for next RectCell
            left += CELL_SIZE + MINOR_GRID_SIZE
            if col != 0 and (col + 1) % 3 == 0:
                left = left + MAJOR_GRID_SIZE - MINOR_GRID_SIZE
            col += 1

        # Update attributes for next RectCell
        top += CELL_SIZE + MINOR_GRID_SIZE
        if row != 0 and (row + 1) % 3 == 0:
            top = top + MAJOR_GRID_SIZE - MINOR_GRID_SIZE
        left = BUFFER + MAJOR_GRID_SIZE
        col = 0
        row += 1

    return cells


def draw_grid():
    '''Draws the major and minor grid lines for Sudoku.'''
    # Draw minor grid lines
    lines_drawn = 0
    pos = BUFFER + MAJOR_GRID_SIZE + CELL_SIZE
    while lines_drawn < 6:
        pygame.draw.line(screen, BLACK_COLOR, (pos, BUFFER),
                         (pos, WIDTH-BUFFER-1), MINOR_GRID_SIZE)
        pygame.draw.line(screen, BLACK_COLOR, (BUFFER, pos),
                         (WIDTH-BUFFER-1, pos), MINOR_GRID_SIZE)

        # Update number of lines drawn
        lines_drawn += 1

        # Update pos for next lines
        pos += CELL_SIZE + MINOR_GRID_SIZE
        if lines_drawn % 2 == 0:
            pos += CELL_SIZE + MAJOR_GRID_SIZE

    # Draw major grid lines
    for pos in range(BUFFER+MAJOR_GRID_SIZE//2, WIDTH, CELL_SIZE*3 + MINOR_GRID_SIZE*2 + MAJOR_GRID_SIZE):
        pygame.draw.line(screen, BLACK_COLOR, (pos, BUFFER),
                         (pos, WIDTH-BUFFER-1), MAJOR_GRID_SIZE)
        pygame.draw.line(screen, BLACK_COLOR, (BUFFER, pos),
                         (WIDTH-BUFFER-1, pos), MAJOR_GRID_SIZE)


def fill_cells(cells, board):
    '''Fills in all the numbers for the game.'''
    font = pygame.font.Font(None, 36)

    # Fill in all cells with correct value
    for row in range(9):
        for col in range(9):
            if board.board[row][col].value is None:
                continue

            # Fill in given values
            if not board.board[row][col].editable:
                font.bold = True
                text = font.render(f'{board.board[row][col].value}', 1, BLACK_COLOR)

            # Fill in values enteRED_COLOR by user
            else:
                font.bold = False
                if board.check_move(board.board[row][col], board.board[row][col].value):
                    text = font.render(
                        f'{board.board[row][col].value}', 1, GREEN_COLOR)
                else:
                    text = font.render(
                        f'{board.board[row][col].value}', 1, RED_COLOR)

            # Center text in cell
            xpos, ypos = cells[row][col].center
            textbox = text.get_rect(center=(xpos, ypos))
            screen.blit(text, textbox)


def draw_button(left, top, WIDTH, HEIGHT, border, color, border_color, text):
    '''Creates a button with a border.'''
    # Draw the border as outer rect
    pygame.draw.rect(
        screen,
        border_color,
        (left, top, WIDTH+border*2, HEIGHT+border*2),
    )

    # Draw the inner button
    button = pygame.Rect(
        left+border,
        top+border,
        WIDTH,
        HEIGHT
    )
    pygame.draw.rect(screen, color, button)

    # Set the text
    font = pygame.font.Font(None, 26)
    text = font.render(text, 1, BLACK_COLOR)
    xpos, ypos = button.center
    textbox = text.get_rect(center=(xpos, ypos))
    screen.blit(text, textbox)

    return button


def draw_board(active_cell, cells, game):
    '''Draws all elements making up the board.'''
    # Draw grid and cells
    draw_grid()
    if active_cell is not None:
        pygame.draw.rect(screen, GRAY_COLOR, active_cell)

    # Fill in cell values
    fill_cells(cells, game)