import time
import cv2
import numpy as np
import tensorflow.keras as keras
from solver import Board, Search

model = keras.models.load_model("./model/best.h5")
board = Board([])
search = Search(board)
solutions: dict[int, list[int]] = {}


def preprocess(img: cv2.Mat) -> cv2.Mat:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 3)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    return thresh


def get_contours(img: cv2.Mat):
    """
    returns a tuple containing the contour, its area and its corners
    """
    cnts, hier = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    result = (None, None)

    for cnt in cnts:
        area = cv2.contourArea(cnt)

        # skip if the area is not large enough
        if area < img.size / 3:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        # skip if the shape is not connected
        if not cv2.isContourConvex(approx):
            continue

        if area > max_area and len(approx) == 4:
            max_area = area
            result = (cnt, approx)

    return result


def sort_corners(corners: list) -> list[tuple]:
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


def get_board(img: cv2.Mat, corners: list) -> cv2.Mat:
    """
    returns the board between given corners in grayscale.
    board is 450x450 pixels
    """
    pst1 = np.float32(corners)
    pst2 = np.float32([[0, 0], [450, 0], [0, 450], [450, 450]])
    matrix = cv2.getPerspectiveTransform(pst1, pst2)
    img_warp = cv2.warpPerspective(img, matrix, (450, 450))

    return img_warp


def split_cells(img: cv2.Mat) -> list:
    cells = []
    rows = np.vsplit(img, 9)
    for row in rows:
        cols = np.hsplit(row, 9)
        for cell in cols:
            cells.append(cell)

    return cells


def crop_cells(cells: list) -> list:
    """
    this function expects that each cell is 50x50 pixels
    """
    cropped_cells = []

    for cell in cells:
        cell = np.asarray(cell)
        cell = cell[5:45, 4:45]
        cropped_cells.append(cell)

    return cropped_cells


def get_filled_cells(cells: list) -> list[bool]:
    """
    each cell must be 50x50 pixels
    """
    result = []

    for cell in cells:
        thresh = cv2.threshold(cell, 128, 255, cv2.THRESH_BINARY_INV)[1]
        num_white_px = np.count_nonzero(thresh[17:33, 17:33])

        if num_white_px > 5:
            result.append(True)
        else:
            result.append(False)

    return result


def get_cell_values(cells: list, filled_cells: list[bool]) -> list[int]:
    result = []

    for i, cell in enumerate(cells):
        if not filled_cells[i]:
            result.append(0)
            continue
        cell = cv2.GaussianBlur(cell, (3, 3), 1)
        cell = cv2.threshold(cell, 128, 255, cv2.THRESH_BINARY)[1]

        cell = cv2.resize(cell, (32, 32))
        cell = np.asarray(cell)
        cell = cell / 255
        cell = cell.reshape(1, 32, 32, 1)

        # ~10x faster than model.predict
        predictions = model(cell, training=False)

        result.append(np.argmax(predictions))

    return result


def blend_non_transparent(background_img, overlay_img):
    """
    wonderful stackoverflow answer
    https://stackoverflow.com/a/37198079/12292576
    """
    # Let's find a mask covering all the non-black (foreground) pixels
    # NB: We need to do this on grayscale version of the image
    gray_overlay = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2GRAY)
    overlay_mask = cv2.threshold(gray_overlay, 1, 255, cv2.THRESH_BINARY)[1]
    # overlay_mask = cv2.threshold(gray_overlay, 80, 255, cv2.THRESH_BINARY)[1]

    # Let's shrink and blur it a little to make the transitions smoother...
    # overlay_mask = cv2.erode(
    #     overlay_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    # overlay_mask = cv2.blur(overlay_mask, (3, 3))

    # And the inverse mask, that covers all the black (background) pixels
    background_mask = 255 - overlay_mask

    # Turn the masks into three channel, so we can use them as weights
    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

    # Create a masked out face image, and masked out overlay
    # We convert the images to floating point in range 0.0 - 1.0
    face_part = (background_img * (1 / 255.0)) * \
        (background_mask * (1 / 255.0))
    overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))

    # And finally just add them together, and rescale it back to an 8bit integer image
    return np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))


def overlay_cells(img: cv2.Mat, corners: list, values: list[int]) -> cv2.Mat:
    digits_img = np.zeros((450, 450, 3), np.uint8)
    for i, value in enumerate(values):
        if value is None:
            continue

        x = (i % 9) * 50
        y = (i // 9) * 50

        cv2.putText(digits_img, str(value),
                    (x + 15, y + 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    3)

    height, width = img.shape[:2]
    pst1 = np.float32(
        [[0, 0], [450, 0], [0, 450], [450, 450]])
    pst2 = np.float32(corners)

    matrix = cv2.getPerspectiveTransform(pst1, pst2)
    img_warp = cv2.warpPerspective(digits_img, matrix, (width, height))

    result = blend_non_transparent(img, img_warp)

    return result


def main(img: cv2.Mat):
    thresh = preprocess(img)
    contour, corners = get_contours(thresh)
    if contour is not None:
        corners = sort_corners(corners)
        cnt_img = img.copy()

        for crn in corners:
            cv2.circle(cnt_img, crn, 15, (189, 70, 189), -1)
        cv2.imshow("board", cnt_img)

        board_img = get_board(img, corners)
        board_gray = cv2.cvtColor(board_img, cv2.COLOR_BGR2GRAY)

        cells = split_cells(board_gray)
        cells = crop_cells(cells)
        filled_cells = get_filled_cells(cells)
        values = get_cell_values(cells, filled_cells)

        board.squares = values

        if not board.is_valid():
            return None

        key = board.get_key()
        solution = solutions.get(key)

        if solution == None:
            succes = search.search_board(0)
            if not succes or not board.is_valid(False):
                return None
            solutions[key] = board.squares
        else:
            board.squares = solutions.get(board.key)

        input_cells = map(
            lambda x: x[1] if filled_cells[x[0]] else None, enumerate(board.squares))
        input_img = overlay_cells(cnt_img, corners, input_cells)
        cv2.imshow("board", input_img)

        solution_cells = map(
            lambda x: x[1] if not filled_cells[x[0]] else None, enumerate(board.squares))
        solution_img = overlay_cells(img, corners, solution_cells)
        cv2.imshow("solution", solution_img)


fps = 0
total_frames = 0
fps_wait = time.time()
cap = cv2.VideoCapture(0)

while True:
    succes, img = cap.read()

    main(img)
    cv2.putText(img,
                "{:.0f} fps".format(fps),
                (15, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (100, 255, 0),
                1,
                cv2.LINE_AA)

    cv2.imshow("input", img)

    total_frames += 1

    cur_time = time.time()
    time_diff = cur_time - fps_wait

    if time_diff > 0.5:
        fps = total_frames / (time_diff)
        total_frames = 0
        fps_wait = cur_time

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
