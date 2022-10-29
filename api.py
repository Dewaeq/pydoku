import cv2
import numpy as np
import base64
from fastapi import File, FastAPI
from fastapi.responses import HTMLResponse
from io import BytesIO
import processing

app = FastAPI()


@app.get("/", response_class=HTMLResponse)
def main():
    return HTMLResponse("".join(open("./index.html").readlines()))


@app.post("/solve")
def solve(file: bytes = File(...)):
    image_stream = BytesIO(file)
    image_stream.seek(0)
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    height, width = frame.shape[:2]

    max_size = 1000

    if width > max_size:
        s = width / max_size
        frame = cv2.resize(frame, (int(width / s), int(height / s)))
    elif height > max_size:
        s = height / max_size
        frame = cv2.resize(frame, (int(width / s), int(height / s)))

    result = processing.process(frame)

    if result.output_image is None:
        return HTMLResponse(content='<h3>Failed to read sudoku</h3>')

    _, png_result = cv2.imencode(".png", result.output_image)
    base64_str = base64.b64encode(png_result.tobytes()).decode('utf-8')

    if not result.succes:
        return HTMLResponse(content=f'''
            <h3>Failed to read board, this is what I see:</h3>
            <img src="data:image/png;base64,{base64_str}">
        ''')

    return HTMLResponse(content=f'''
        <h3>Output:</h3>
        <img src="data:image/png;base64,{base64_str}" style="height: 100%; width: 100%; object-fit: contain">
    ''')
