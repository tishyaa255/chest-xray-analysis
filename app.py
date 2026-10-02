from flask import Flask, render_template, request
from ai_edge_litert.interpreter import Interpreter
import cv2
import numpy as np
import os

app = Flask(__name__)

IMAGE_SIZE = 180
MODEL_PATH = "xray_model.tflite"

CLASS_NAMES = [
    "Cardiomegaly",
    "Normal",
    "Other Abnormality",
    "Pneumonia",
    "Pneumothorax",
    "Tuberculosis (TB)"
]

# Load LiteRT model
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


@app.route("/", methods=["GET", "POST"])
def home():

    results = None
    prediction = None
    score = None

    if request.method == "POST":

        file = request.files["xray"]

        if file:

            image_path = "uploaded_xray.png"
            file.save(image_path)

            image = cv2.imread(
                image_path,
                cv2.IMREAD_GRAYSCALE
            )

            if image is not None:

                image = cv2.resize(
                    image,
                    (IMAGE_SIZE, IMAGE_SIZE)
                )

                # IMPORTANT:
                # Do NOT divide by 255.
                # The model already contains Rescaling.
                image = image.astype("float32")

                image = image.reshape(
                    1,
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                    1
                )

                # Send image to LiteRT model
                interpreter.set_tensor(
                    input_details[0]["index"],
                    image
                )

                interpreter.invoke()

                prediction_values = interpreter.get_tensor(
                    output_details[0]["index"]
                )[0]

                predicted_index = np.argmax(
                    prediction_values
                )

                prediction = CLASS_NAMES[
                    predicted_index
                ]

                score = (
                    prediction_values[
                        predicted_index
                    ] * 100
                )

                results = []

                for i in range(6):

                    results.append({
                        "name": CLASS_NAMES[i],
                        "percentage": round(
                            prediction_values[i] * 100,
                            2
                        )
                    })

            if os.path.exists(image_path):
                os.remove(image_path)

    return render_template(
        "index.html",
        results=results,
        prediction=prediction,
        score=score
    )


if __name__ == "__main__":
    app.run(debug=True)
