import gradio as gr
from PIL import Image
import requests
import io
import numpy as np

def recognize_digit(image):
    # Convert the NumPy array (from Gradio sketchpad) to a PIL Image
    image = Image.fromarray(image.astype('uint8'))
    print(image)

    # Convert the PIL Image to bytes
    img_binary = io.BytesIO()
    image.save(img_binary, format="PNG")
    img_binary.seek(0)  # Reset stream position

    # Send the image to the API for prediction
    response = requests.post("http://127.0.0.1:5000/predict", data=img_binary.getvalue())
    print("API Response:", response.json())


    # Check if the request was successful
    if response.status_code == 200:
        prediction = response.json().get("prediction", "Error: No prediction returned")
        return f"Predicted Digit: {prediction}"
    else:
        return f"Error: {response.status_code} - {response.text}"

if __name__ == '__main__':
    # Create a Gradio interface
    gr.Interface(
        fn=recognize_digit, 
        inputs="sketchpad",  # Gradio sketchpad input
        outputs='label',     # Output as a label
        live=True,           # Update predictions in real-time
        description="Draw a number on the sketchpad to see the model's prediction.",
    ).launch(debug=True, share=True)  # Launch the interface