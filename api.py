import cv2
import numpy as np
import uvicorn
from fastapi import File, FastAPI, UploadFile
from fastapi.responses import StreamingResponse, HTMLResponse
from PIL import Image
from io import BytesIO

from processing import process

app = FastAPI()


@app.get("/", response_class=HTMLResponse)
def main():
    return HTMLResponse("".join(open("./index.html").readlines()))


@app.post("/solve")
def solve(file: UploadFile = File(...)):
    img = Image.open(file.file)
    max_size = 1000

    if img.width > max_size:
        s = img.width / max_size
        img = img.resize((int(img.width / s), int(img.height / s)))
    elif img.height > max_size:
        s = img.height / max_size
        img = img.resize((int(img.width / s), int(img.height / s)))

    cv2_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    result = process(cv2_img)


    if result is None:
        return HTMLResponse(content="<h3>Failed to read sudoku</h3>")

    _, png_result = cv2.imencode(".png", result)

    return StreamingResponse(BytesIO(png_result.tobytes()), media_type="image/png")

