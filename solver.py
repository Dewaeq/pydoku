from typing import List, Tuple
import random


class Board:
    key: int
    squares: List[int]

    def __init__(self, squares: List[int]) -> None:
        self.key = 0
        self.squares = squares

    def is_valid(self, can_contain_zero=True) -> bool:
        valid = True
        for i in range(0, 81):
            value = self.squares[i]
            if value == 0 and can_contain_zero:
                continue
            self.squares[i] = -1
            if not self.valid_square_value(i, value):
                valid = False
                self.squares[i] = value
                break
            self.squares[i] = value

        return valid

    def get_key(self) -> int:
        for square, value in enumerate(self.squares):
            self.key ^= SquaresArray[square][value]
        return self.key

    def valid_square_value(self, square, value) -> bool:
        column, row = self.get_square_coord(square)
        return self.value_fits_row(row, value) and self.value_fits_column(column, value) and self.value_fits_cell(square, value)

    def value_fits_row(self, row, value) -> bool:
        start = row * 9
        stop = start + 9
        return value not in self.squares[start:stop]

    def value_fits_column(self, column, value) -> bool:
        for i in range(0, 9):
            if self.squares[i * 9 + column] == value:
                return False
        return True

    def value_fits_cell(self, square, value) -> bool:
        x, y = self.get_square_coord(square)
        cell_square = self.square_from_coord(x - (x % 3), y - (y % 3))
        for i in range(0, 3):
            for j in range(0, 3):
                if self.squares[cell_square + j] == value:
                    return False
            cell_square += 9
        return True

    def get_square_coord(self, square: int) -> Tuple[int, int]:
        y = square // 9
        x = square % 9
        return (x, y)

    def square_from_coord(self, x, y) -> int:
        return 9 * y + x

    def print_board(self) -> None:
        print("###########################")
        for y in range(0, 9):
            for x in range(0, 9):
                i = self.square_from_coord(x, y)
                value = self.squares[i]
                if x == 8:
                    print(f" {value} ")
                else:
                    print(f" {value} ", end="")
        print("###########################")


class Search:
    def __init__(self, board: Board) -> None:
        self.board = board

    def get_empty_square(self, square) -> int:
        for i in range(square, 81):
            if self.board.squares[i] == 0:
                return i
        return 255

    def search_board(self, square) -> bool:
        square = self.get_empty_square(square)

        # board is completed
        if square == 255:
            return True

        for i in range(1, 10):
            if self.board.valid_square_value(square, i):
                self.board.squares[square] = i
                if self.search_board(square):
                    return True

            self.board.squares[square] = 0
        return False


SquaresArray = [[0] * 10] * 81

for i in range(0, 81):
    for j in range(0, 10):
        SquaresArray[i][j] = random.randint(0, 2 ** 10)
