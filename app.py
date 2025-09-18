import numpy as np
import gradio as gr
import tensorflow as tf
import joblib


model = tf.keras.models.load_model("iris_ann_model.h5")
scaler = joblib.load("iris_scaler.pkl")


class_labels = ['Iris-virginica', 'Iris-versicolor', 'Iris-setosa']



def predict_species(sepal_length, sepal_width, petal_length, petal_width):
   
    raw_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    
    scaled_data = scaler.transform(raw_data)
    
    prediction = model.predict(scaled_data, verbose=0)
    pred_class = np.argmax(prediction, axis=1)[0]
    
    return f"🌸 Predicted Species: {class_labels[pred_class]}"

inputs = [
    gr.Number(label="Sepal Length (cm)", value=5.1, precision=1, minimum=4.0, maximum=8.0),
    gr.Number(label="Sepal Width (cm)", value=3.5, precision=1, minimum=2.0, maximum=4.5),
    gr.Number(label="Petal Length (cm)", value=1.4, precision=1, minimum=1.0, maximum=7.0),
    gr.Number(label="Petal Width (cm)", value=0.2, precision=1, minimum=0.1, maximum=2.5),
]


outputs = gr.Textbox(label="Prediction", interactive=False)

demo = gr.Interface(
    fn=predict_species,
    inputs=inputs,
    outputs=outputs,
    title="🌸 Iris Flower Classification (ANN)",
    description="Predict the species of Iris flower using a trained ANN model.",
    theme=gr.themes.Soft(),
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch()
