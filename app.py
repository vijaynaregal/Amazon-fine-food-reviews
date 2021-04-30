from flask import Flask, request, jsonify, render_template
import pickle


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
vector = pickle.load(open('vector.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    reviews =  request.form.get('Statement')
    review_vector = vector.transform([reviews]) 
    prediction = model.predict(review_vector)
    if prediction == 1:
        senti = 'Positive'
    elif prediction == 0:
        senti = 'Negative'
    return render_template('index.html', prediction_text='The sentiment of this food review is : {}'.format(senti))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port = 8080)

    
