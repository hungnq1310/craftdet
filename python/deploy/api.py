import numpy as np
import os
import io
import pathlib
import time
import sys

from functools import partial
from PIL import Image
import pdf2image
from fastapi import *
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from typing import List
from pydantic import BaseModel

from dotenv import load_dotenv
from tritonclient import grpc as grpcclient
from tritonclient import http as httpclient
from tritonclient.utils import InferenceServerException
from PIL import ImageFont, ImageDraw, Image

from python.craftdet.utils import client

from trism import TritonModel


#############
# Initialize
#############
load_dotenv()
#
MODEL_NAME    = os.getenv("MODEL_NAME")
MODEL_VERSION = os.getenv("MODEL_VERSION", "")
BATCH_SIZE    = int(os.getenv("BATCH_SIZE", 1))
#
TRITON_URL    = os.getenv("TRITON_URL", "localhost:8000")
PROTOCAL      = os.getenv("PROTOCOL", "HTTP")
VERBOSE       = os.getenv("VERBOSE", "False").lower() in ("true", "1", "t")
ASYNC_SET     = os.getenv("ASYNC_SET", "False").lower() in ("true", "1", "t")
#
grpc = PROTOCAL.lower() == "grpc"
# -----------------------------------------------------------------------------
model = TritonModel(
    model=MODEL_NAME,
    version=MODEL_VERSION,
    url=TRITON_URL,
    grpc=grpc
)
# View metadata.
for inp in model.inputs:
  print(f"name: {inp.name}, shape: {inp.shape}, datatype: {inp.dtype}\n")
for out in model.outputs:
  print(f"name: {out.name}, shape: {out.shape}, datatype: {out.dtype}\n")

# -------------------------------------------------------------------------------


##################################
"""Define images app to store image after process"""
OUTPUT_DIR = os.getenv('OUTPUT_DIR', default='prediction')
os.makedirs(OUTPUT_DIR, exist_ok=True)

ORIGIN_IMAGE_PATH = os.getenv('ORIGIN_IMAGE_PATH', default='origin_images')
ORIGIN_IMAGE_PATH = pathlib.Path(OUTPUT_DIR + "/" + ORIGIN_IMAGE_PATH)
ORIGIN_IMAGE_PATH.mkdir(exist_ok=True)

OCR_IMAGE_PATH = os.getenv('OCR_IMAGE_PATH', default='ocr_images')
OCR_IMAGE_PATH = pathlib.Path(OUTPUT_DIR + "/" + OCR_IMAGE_PATH)
OCR_IMAGE_PATH.mkdir(exist_ok=True)

OCR_TEXT_PATH = os.getenv('OCR_TEXT_PATH', default='ocr_text')
OCR_TEXT_PATH = pathlib.Path(OUTPUT_DIR + "/" + OCR_TEXT_PATH)
OCR_TEXT_PATH.mkdir(exist_ok=True)


############
# Config
############


class ImageBatchRequest(BaseModel):
    images: List[np.ndarray]


#
image_api = FastAPI()

# 
@image_api.get("/")
def read_app():
    return {"Hello": "Image Apps"}

image_api.mount('/ori_imgs', StaticFiles(directory=str(ORIGIN_IMAGE_PATH)), name='origin_images')
image_api.mount('/ocr_imgs', StaticFiles(directory=str(OCR_IMAGE_PATH)), name='ocr_images')
image_api.mount('/ocr_texts', StaticFiles(directory=str(OCR_TEXT_PATH)), name='oct_text')

##################################
'''Main App'''

#
app = FastAPI()
app.mount("/imageapi", image_api)
1
#
@app.get("/")
def root():
    return {"Hello": "Main Apps"}
#
@app.post("/upload/")
async def upload(files: List[UploadFile] = File(...)) -> JSONResponse:
    for file in files:
        request_object_content = await file.read()
        extension = file.filename.split(".")[-1]

        images = []
        if extension in ["jpg", "jpeg", "png"]:
            images = [Image.open(io.BytesIO(request_object_content))]
        elif extension in ["pdf"]:
            images = pdf2image.convert_from_bytes(request_object_content)
        else:
            raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Unsupported file type")
        
        assert len(images) > 0, "No image found after processing"
        for idx, image in enumerate(images):
            image.save(f"{ORIGIN_IMAGE_PATH}/{file.filename}_image{idx}.jpg")

    return JSONResponse(content={
        "message": [file.file.name for file in files]
    }, status_code=status.HTTP_200_OK)

    
    # for i, img in enumerate(images):
        
    #     img_rectify = np.ascontiguousarray(img)
        
    #     # predict OCR
    #     img_ocr, texts = ocr_predict(img_rectify, detector, ocr_model, i, len(images)) 

    #     # save to image api
    #     cv2.imwrite(f"{ORIGIN_IMAGE_PATH}/{file.filename}_{i}.jpg", np.asarray(img))
    #     cv2.imwrite(f"{OCR_IMAGE_PATH}/{file.filename}_{i}.jpg", img_ocr)
    #     with open(f"{OCR_TEXT_PATH}/{file.filename}_content{i}.txt", 'w') as f:
    #         for line in texts:
    #             f.write("%s\n" % line)


@app.post("/detect")
async def detect(request: ImageBatchRequest):
    #TODO: pipeline from upload file to image ocr return
    # 1. get file from upload
    # 2. call request to upload api
    # create folder for each file to save image
    
    images = request.images    
    assert len(images) > 0, "No image found after processing"
    
    # images one 1 file
    craftdet_response = craftdet(request)
    bboxes = craftdet_response.body["boxes"]

    for idx, img in enumerate(images):
        img_rectify = Image.fromarray(img)
        batch_img_rectify_crop = [
            np.array(img_rectify.crop(bboxes[j])) for j in range(len(bboxes))
        ]
        batch_texts = vietocr(ImageBatchRequest(images=batch_img_rectify_crop))
        for j, text in enumerate(batch_texts):
            cv2drawboxtext(img, text, bboxes[j], "page", j)
    return JSONResponse(content={"message": "success"}, status_code=status.HTTP_200_OK)
            

    # call request rieng cho tung module
    
@app.post("/detect/vietocr")
def vietocr(request: ImageBatchRequest):

    images = request.images
    assert len(images) > 0, "No image found after processing"

    try:
        start_time = time.time()
        outputs = model.run(data = [images])
        end_time = time.time()
        print("Process time: ", end_time - start_time)
        return JSONResponse(outputs)
    except InferenceServerException as e:
        return {"Error": "Inference failed with error: " + str(e)}
    except Exception as e:
        return {"Error": "An unexpected error occurred: " + str(e)}


    
@app.post("/detect/craftdet")
def craftdet(request: ImageBatchRequest):
    images = request.images
    assert len(images) > 0, "No image found after processing"
    try:
        start_time = time.time()
        outputs = model.run(data = [images])
        end_time = time.time()
        print("Process time: ", end_time - start_time)
        return JSONResponse(outputs)
    except InferenceServerException as e:
        return {"Error": "Inference failed with error: " + str(e)}
    except Exception as e:
        return {"Error": "An unexpected error occurred: " + str(e)}


@app.get("/ocr")
def get_ocr():
    ocr_imgs = [f'/imageapi/ocr_imgs/{k}' for k in os.listdir(OCR_IMAGE_PATH)]
    return {"ocr": ocr_imgs}

@app.get("/texts")
def get_texts():
    ocr_texts = [f'/imageapi/ocr_texts/{k}' for k in os.listdir(OCR_TEXT_PATH)]
    return {"texts": ocr_texts}

@app.get("/origin")
def get_origin():
    origin_imgs = [f'/imageapi/ori_imgs/{k}' for k in os.listdir(ORIGIN_IMAGE_PATH)]
    return {"retify": origin_imgs}

##################################

def requestGenerator(batched_image_data, input_name, output_names, dtype):
    
    if PROTOCAL == "grpc":
        client = grpcclient
    else:
        client = httpclient

    # Set the input data
    inputs = [client.InferInput(input_name, batched_image_data.shape, dtype)]
    inputs[0].set_data_from_numpy(batched_image_data)

    outputs = [
        client.InferRequestedOutput(output_names[0], binary_data=True), 
        client.InferRequestedOutput(output_names[1], binary_data=True)
    ]

    return inputs, outputs

def cv2drawboxtext(img: np.ndarray, text, a, filename, idx):
    font = ImageFont.truetype("font-times-new-roman/SVN-Times New Roman 2.ttf", 20)
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    # https://www.blog.pythonlibrary.org/2021/02/02/drawing-text-on-images-with-pillow-and-python/
    bbox = draw.textbbox(a, text, font=font, anchor='ls')

    draw.rectangle(a, fill="yellow", width=2) # draw bbox detection 
    draw.rectangle(bbox, fill="yellow") # draw text detection
    draw.text(a, text, font=font, anchor='ls', fill="black")
    img_pil.save(f"{OCR_IMAGE_PATH}/{filename}_{idx}.jpg")
    return img