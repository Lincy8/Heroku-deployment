#importing essentiaal libraries

from flask import Flask, render_template, request
import pickle


app = Flask(__name__)
model = pickle.load(open('Loyalty-type.pkl','rb'))
@app.route("/")
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    Recency = int(request.form['Recency'])
    frequency = int(request.form['frequency'])
    Monetory = int(request.form['Monetory'])
    df_RFM = [[Recency, frequency,Monetory]]
    if (Recency == 0 and frequency ==0 and Monetory == 0):
        my_prediction = [0]
    else:
        my_prediction = model.predict(df_RFM)

    return render_template('index.html', prediction_text='Customer type {}'.format(round(my_prediction[0],2)))

if __name__ == "__main__":
    app.run(debug=True)
