
import os
import numpy as np
import cv2
import tensorflow as tf
import scipy.io
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import google.generativeai as genai

# Initialize the FastAPI app
app = FastAPI()

# Handle CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Suppress TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"


GEMINI_API_KEY = "API_KEY"
genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel('gemini-1.5-flash')


# Global parameters
IMG_H = 256
IMG_W = 256
NUM_CLASSES = 11

# Utility functions (unchanged from Flask)
def create_dir(path):
    """Create a directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)

def read_image(image_file: UploadFile):
    """Read and preprocess the uploaded image."""
    npimg = np.frombuffer(image_file.file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (IMG_W, IMG_H))
    img = img.astype(np.float32)
    return img

def get_colormap():
    """Load colormap and class names."""
    colormap = scipy.io.loadmat('utils/ultimate.mat')["color"]
    classes = [
        "Background", "Spleen", "Right kidney", "Left kidney", "Liver",
        "Gallbladder", "Stomach", "Aorta", "Inferior vena cava", "Portal vein",
        "Pancreas"
    ]
    return classes, colormap

def grayscale_to_rgb(pred, classes, colormap):
    """Convert grayscale prediction to RGB using colormap."""
    h, w = IMG_H, IMG_W
    pred = np.squeeze(pred, axis=-1)
    pred = pred.astype(np.int32)
    output = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            pixel = pred[i, j]
            output[i, j, :] = colormap[pixel]
    return output

def save_results(image, pred, classes, colormap, save_image_path):
    """Save the blended image with prediction overlay."""
    if os.path.exists(save_image_path):
        os.remove(save_image_path)  # Delete existing file

    pred = np.expand_dims(pred, axis=-1)
    pred = grayscale_to_rgb(pred, classes, colormap)

    alpha = 0.5
    blended_image = alpha * image + (1 - alpha) * pred

    cv2.imwrite(save_image_path, blended_image)

# Define the POST endpoint
@app.post('/process_image')
async def process_image(image: UploadFile = File(...)):
    """Process the uploaded image and return the result."""
    # Directory for storing results
    saving_path = "results"
    create_dir(saving_path)

    # Load the model
    model = tf.keras.models.load_model('utils/model.h5')

    # Read and preprocess the image
    image_x = read_image(image)
    image_array = np.expand_dims(image_x, axis=0)

    # Make prediction
    pred = model.predict(image_array, verbose=0)[0]
    pred = np.argmax(pred, axis=-1)
    pred = pred.astype(np.float32)

    # Get colormap and classes
    classes, colormap = get_colormap()

    # Save the result
    save_image_path = f"{saving_path}/unet_prediction1.png"
    save_results(image_x, pred, classes, colormap, save_image_path)

    # Return the processed image
    return FileResponse(save_image_path, media_type='image/png')

@app.post('/get_diagnosis_gemini')
async def get_diagnosis_gemini(segmented_image: UploadFile = File(...)):
    """
    Takes a segmented PNG image, sends it to the Gemini API with a prompt,
    and returns Gemini's diagnosis.
    """
    try:
        
        image_bytes = await segmented_image.read()

       
        image_part = {
            'mime_type': 'image/png',
            'data': image_bytes
        }

        
        prompt_parts = [
            "Analyze this medical image segmentation. It highlights various organs. ",
            "Based on the segmentation, identify the organs present and provide a concise medical diagnosis or observation. ",
            "Highlight any abnormalities or noteworthy findings if visible in the segmentation. ",
            "For example, if you see an enlarged organ, mention it. If a normal organ is missing, mention it. ",
            "Focus purely on anatomical observations from the segmentation. Do not guess about conditions not visible.",
            image_part,
            "Detailed diagnosis:"
        ]

        response = model.generate_content(prompt_parts)
        print(response)
        diagnosis_text = response.text

        return JSONResponse(content={"diagnosis": diagnosis_text})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing diagnosis: {str(e)}")

# Run the app with Uvicorn
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)