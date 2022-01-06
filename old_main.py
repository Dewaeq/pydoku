import cv2
import numpy as np
import tensorflow.keras as keras
from solver import Board, Search

BOARD_SIZE = 900
CELL_SIZE = BOARD_SIZE // 9

game_board = Board([])
search = Search(game_board)

cap = cv2.VideoCapture(0)
model = keras.models.load_model("./model/f.h5")


def main():
    img = cv2.imread("./f.jpeg")
    # img = cv2.imread("./sudoku.png")
    while True:
        # success, img = cap.read()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 3)
        canny = cv2.Canny(blur, 50, 50)
        cv2.imshow("canny", canny)

        cnt = get_countours(canny)
        if cnt is not None:
            corners = get_corners(cnt)
            corners = sort_corners(corners)
            img_copy = img.copy()
            for crn in corners:
                cv2.circle(img_copy, crn, 15, (189, 70, 189), -1)
            cv2.imshow("input", img_copy)

            board = get_board(img, corners)
            cells = get_cells(board)
            empty_cells, cells = get_empty_cells(cells)
            cell_values = get_cell_values(cells, empty_cells)
            game_board.squares = cell_values
            game_board.print_board()
        else:
            cv2.imshow("input", img)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break


def get_countours(img):
    # find the outer contours
    cnts, hier = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    max_area = 0
    result = None
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.015 * peri, True)

        # sksip if the area is to small or not not connected
        if area < img.size / 3 or not cv2.isContourConvex(approx):
            continue

        if area > max_area and len(approx) == 4:
            result = cnt
            max_area = area

    return result


def get_corners(contour):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.015 * peri, True)
    return approx


def sort_corners(corners):
    """
    order is upper-left, upper-right, lower-left, lower-right
    """
    crns = [(c[0][0], c[0][1]) for c in corners]
    # tuple is sorted by its first parameter
    crns.sort()

    def sort_crn(crn):
        return crn[1]

    left_crns = crns[0:2]
    right_crns = crns[2:4]

    left_crns.sort(key=sort_crn)
    right_crns.sort(key=sort_crn)

    return [left_crns[0], right_crns[0], left_crns[1], right_crns[1]]


def get_board(img, corners):
    """
    returns board in grayscale
    """
    pst1 = np.float32(corners)
    width = BOARD_SIZE
    height = BOARD_SIZE
    pst2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    matrix = cv2.getPerspectiveTransform(pst1, pst2)
    img_perspective = cv2.warpPerspective(img, matrix, (width, height))
    board = cv2.cvtColor(img_perspective, cv2.COLOR_BGR2GRAY)

    return board


def get_cells(board):
    cells = []
    offset = int(CELL_SIZE * 0.1)
    for x in range(0, 9):
        for y in range(0, 9):
            cell_x = x * CELL_SIZE
            cell_y = y * CELL_SIZE

            cell = board[cell_x + offset:cell_x +
                         CELL_SIZE - offset, cell_y + offset:cell_y + CELL_SIZE - offset]
            cells.append(cell)

    return cells


def get_empty_cells(cells):
    """
    empty cells are `True`, cells containing a digit are `False`
    """
    values = []

    for i in range(0, len(cells)):
        # cell = cv2.GaussianBlur(cells[i], (5, 5), 3)
        _, cell = cv2.threshold(cells[i], 100, 255, cv2.THRESH_BINARY_INV)
        cells[i] = cell

    for cell in cells:
        num_black_px = np.sum(cell == 255)
        values.append(num_black_px)

    m = max(values) // 4
    for i, value in enumerate(values):
        if value > m:
            values[i] = False
        else:
            values[i] = True

    return values, cells


def get_cell_value(cell):
    cell = cv2.resize(cell, (28, 28))
    cell = np.array([cell])
    predection = model.predict(cell)

    return np.argmax(predection)


def get_cell_values(cells, empty_cells):
    cv2.imshow("f", cells[59])
    cv2.imshow("u", cells[77])
    values = [None] * 81
    for i, empty in enumerate(empty_cells):
        if empty:
            values[i] = 0
            continue
        values[i] = get_cell_value(cells[i])
    return values


main()


cap.release()
cv2.destroyAllWindows()
