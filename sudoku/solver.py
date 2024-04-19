from sudokuUI import *

class Sudoku:
    '''Represents a game/board of Sudoku.'''

    def __init__(self, board):
        '''Initializes an instance of a Sudoku game.'''
        self.initial_board = board
        self.board = []
        for row in range(9):
            self.board.append([])
            for col in range(9):
                if board[row][col] == 0:
                    val = None
                    editable = True
                else:
                    val = board[row][col]
                    editable = False
                self.board[row].append(Cell(row, col, val, editable))

    def is_editable(self, cell):
        return cell.editable


    def check_move(self, cell, num):
        '''Returns whether a number is a valid move for a cell.'''
        # Check if the number is valid for the row
        for col in range(9):
            if self.board[cell.row][col].value == num and col != cell.col:
                return False

        # Check if the number is valid for the column
        for row in range(9):
            if self.board[row][cell.col].value == num and row != cell.row:
                return False

        # Check if the number is valid in its box
        for row in range(cell.row // 3 * 3, cell.row // 3 * 3 + 3):
            for col in range(cell.col // 3 * 3, cell.col // 3 * 3 + 3):
                if (
                    self.board[row][col].value == num
                    and row != cell.row
                    and col != cell.col
                ):
                    return False

        # Move is valid
        return True

    def get_possible_moves(self, cell):
        '''Returns a list of the valid moves for a cell.'''
        possible_moves = [num for num in range(1, 10)]

        # Check numbers in the row
        for col in range(9):
            if self.board[cell.row][col].value in possible_moves:
                possible_moves.remove(self.board[cell.row][col].value)

        # Check numbers in the column
        for row in range(9):
            if self.board[row][cell.col].value in possible_moves:
                possible_moves.remove(self.board[row][cell.col].value)

        # Check numbers in the box
        for row in range(cell.row // 3 * 3, cell.row // 3 * 3 + 3):
            for col in range(cell.col // 3 * 3, cell.col // 3 * 3 + 3):
                if self.board[row][col].value in possible_moves:
                    possible_moves.remove(self.board[row][col].value)

        return possible_moves

    def get_empty_cell(self):
        '''Returns an empty cell. Returns False if all cells are filled in.'''
        for row in range(9):
            for col in range(9):
                if self.board[row][col].value is None:
                    return self.board[row][col]

        return False
    
    def get_empty_cells(self):
        '''Returns a list of empty cells.'''
        empty_cells = []
        for row in range(9):
            for col in range(9):
                if self.board[row][col].value is None:
                    empty_cells.append(self.board[row][col])

        return empty_cells

    def solve(self):
        '''
        Solves the game from it's current state with a backtracking algorithm.
        Returns True if successful and False if not solvable.
        '''
        cell = self.get_empty_cell()

        # Board is complete if cell is False
        if not cell:
            return True

        # Check each possible value in cell
        for val in range(1, 10):

            # Check if the value is a valid move
            if not self.check_move(cell, val):
                continue

            # Place value in board
            cell.value = val

            # If all recursive calls return True then board is solved
            if self.solve():
                return True

            # Undo move is solve was unsuccessful
            cell.value = None

        # No moves were successful
        return False

    def get_board(self):
        '''Returns a list of values that are in the Sudoku board.'''
        return [[self.board[row][col].value for col in range(9)] for row in range(9)]

    def test_solve(self):
        '''Checks if the current configuration is solvable.'''
        current_board = self.get_board()
        solvable = self.solve()

        # Reset board to state before solve check
        for row in range(9):
            for col in range(9):
                self.board[row][col].value = current_board[row][col]

        return solvable

    def reset(self):
        '''Resets the game to its starting state.'''
        for row in self.board:
            for cell in row:
                if cell.editable:
                    cell.value = None

    def __str__(self):
        '''Returns a string representing the board.'''
        board = ' -----------------------\n'
        for row, line in enumerate(self.board):
            board += '|'
            for col, cell in enumerate(line):
                if cell.value is None:
                    val = '-'
                else:
                    val = cell.value
                if col < 8:
                    board += f' {val}'
                    if (col + 1) % 3 == 0:
                        board += ' |'
                else:
                    board += f' {val} |\n'
            if row < 8 and (row + 1) % 3 == 0:
                board += '|-------|-------|-------|\n'
        board += ' -----------------------\n'
        return board

    def check_sudoku(self):
        '''
        Takes a complete instance of Soduku and 
        returns whether or not the solution is valid.
        '''
        # Ensure all cells are filled
        if self.get_empty_cell():
            raise ValueError('Game is not complete')

        # Will hold values for each row, column, and box
        row_sets = [set() for _ in range(9)]
        col_sets = [set() for _ in range(9)]
        box_sets = [set() for _ in range(9)]

        # Check all rows, columns, and boxes contain no duplicates
        for row in range(9):
            for col in range(9):
                box = (row // 3) * 3 + col // 3
                value = self.board[row][col].value

                # Check if number already encounteRED_COLOR in row, column, or box
                if value in row_sets[row] or value in col_sets[col] or value in box_sets[box]:
                    return False

                # Add value to corresponding set
                row_sets[row].add(value)
                col_sets[col].add(value)
                box_sets[box].add(value)

        # All rows, columns, and boxes are valid
        return True

