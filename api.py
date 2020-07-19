import io, cv2
import numpy as np
from fastapi.datastructures import UploadFile
from fastapi.param_functions import File, Body
from fastapi.applications import FastAPI
from utils.tf_serving_helper import YoloHelper

yolohelper = YoloHelper()

app = FastAPI(title="Yolov2 Deployment")


@app.get("/ping", status_code=200, summary=" Liveliness Check ")
async def ping():
    return {"ping": "pong"}


@app.post("/v1/predict", status_code=200)
async def predict(image: UploadFile = File(...)):
    img_read = image.file._file.read()
    img = cv2.imdecode(np.fromstring(img_read, np.uint8), cv2.COLOR_RGB2BGR)
    pre_img = yolohelper.preprocess(img)
    predict_img = yolohelper.predict(pre_img,threshold=0.6)
    return predict_img


