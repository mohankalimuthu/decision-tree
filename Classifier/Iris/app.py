from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("C:\\Users\\hp\\PyCharmMiscProject\\git\\decision-tree-1\\Classifier\\Iris\\iris_model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        sepal_length = float(request.form["sepal_length"])
        sepal_width = float(request.form["sepal_width"])
        petal_length = float(request.form["petal_length"])
        petal_width = float(request.form["petal_width"])

        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        pred = model.predict(input_data)[0]

        classes = ["Setosa", "Versicolor", "Virginica"]
        prediction = classes[pred]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
