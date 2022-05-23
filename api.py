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
async def solve(file: UploadFile = File(...)):
    img = Image.open(file.file)
    cv2_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    result = process(cv2_img)

    if result is None:
        return HTMLResponse(content="h3>Failed to read sudoku</h3>")

    _, png_result = cv2.imencode(".png", result)

    return StreamingResponse(BytesIO(png_result.tobytes()), media_type="image/png")

if __name__ == "__main__":
    uvicorn.run(app)
