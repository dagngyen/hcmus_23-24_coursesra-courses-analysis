import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    num_enrollment = int(request.form['enrollment'])
    duration = float(request.form['duration'])
    instructor_rate = float(request.form['ins_rate'])
    general_topic = request.form['general']
    specific_topic = request.form['specific']
    language = request.form['language']
    level = request.form['level']
    offered_by = request.form['offered']

    data = {
        'enrollment': [num_enrollment],
        'duration': [duration],
        'instructor_rate': [instructor_rate],
        'general': [general_topic],
        'specify': [specific_topic],
        'language': [language],
        'level': [level],
        'offered by': [offered_by]
    }

    final_features = pd.DataFrame(data)
    prediction = model.predict(final_features)
    if prediction[0][0] > 5.0:
        prediction = 5
    else:
        prediction = prediction[0][0]

    return render_template('index.html', prediction_text='Rating is {:.3f}'.format(prediction))


@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    prediction = model.predict(pd.DataFrame(data))
    if prediction[0][0] > 5.0:
        output = 5
    else:
        output = prediction[0][0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
