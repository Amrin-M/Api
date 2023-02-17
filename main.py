import string
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

# origins = [
#     "http://localhost",
#     "http://localhost:8000",
#     "http://localhost:3000",
#     "exp://10.10.32.171:19000"
# ]
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

MODEL_INSECT = tf.keras.models.load_model("C:/MyBananaProject/saved_models/3")
CLASS_NAMES_INSECT = ["Earwigs", "Larva", "Weevils"]

MODEL_STEM = tf.keras.models.load_model("C:/MyBananaProject/saved_models_stem/1")
CLASS_NAMES_STEM = ["Fusarium wilt", "Healthy stem"]

MODEL_LEAF = tf.keras.models.load_model("C:/MyBananaProject/saved_models_leaves/1")
CLASS_NAMES_LEAF = ["Bacterial wilt", "Cordana", "Healthy","Pestalotiopsis","Sigatoka"]


@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
   
    img = Image.open(BytesIO(data))
    new_img = img.resize((256,256))
    image = np.array(new_img)
    
    return image

@app.post("/predict")
async def predict(
    type,
    file: UploadFile = File(...)     
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    
    predictions=""
    predicted_class=""
    confidence=0
    
    if(type=="insect"):
        predictions = MODEL_INSECT.predict(img_batch)
        predicted_class = CLASS_NAMES_INSECT[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])*100
    elif(type=="stem"):
        predictions = MODEL_STEM.predict(img_batch)
        predicted_class = CLASS_NAMES_STEM[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])*100
    elif(type=="leaf"):
        predictions = MODEL_LEAF.predict(img_batch)
        predicted_class = CLASS_NAMES_LEAF[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])*100  

    
    return {
        'class': predicted_class,
        'confidence': float(confidence),
        'TYPE': type

    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)