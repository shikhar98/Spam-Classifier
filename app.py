from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from collections import Counter
import os
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)
data = {}
def build_data(root):
    all_words = []
    files = [os.path.join(root, file) for file in os.listdir(root)]

    global data

    for file in files:
        with open(file) as f:
            for line in f:
                words = line.split()
                all_words += words

    frequent = Counter(all_words)

    all_keys = list(frequent)

    for key in all_keys:
        if key.isalpha() == False:
            del frequent[key]

    frequent = frequent.most_common(2500)

    count = 0
    for word in frequent:
        data[word[0]] = count
        count += 1
    print(all_words)
    print(all_keys)

def feature_extraction(root):
    files = [os.path.join(root, file) for file in os.listdir(root)]
    matrix = np.zeros((len(files), 2500))
    labels = np.zeros(len(files))
    file_count = 0

    for file in files:
        with open(file) as file_obj:
            for index, line in enumerate(file_obj):
                if index == 2:
                    line = line.split()
                    for word in line:
                        if word in data:
                            matrix[file_count, data[word]] = line.count(word)

        labels[file_count] = 0
        if 'spmsg' in file:
            labels[file_count] = 1
        file_count += 1
    return matrix, labels

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    training_data = './training-data'
    testing_data = './testing-data'
    # Building word data
    build_data(training_data)

    print('Extracting features')
    training_feature, training_labels = feature_extraction(training_data)
    testing_features, testing_labels = feature_extraction(testing_data)
    model = MultinomialNB()
    model.fit(training_feature, training_labels)
    # Predicting
    predicted_labels = model.predict(testing_features)
    #print('Accuracy:', accuracy_score(testing_labels, predicted_labels) * 100)
    if request.method == 'POST':
        comment = request.form['comment']
        import os.path
        save_path = './RunningData/'
        # Predicting
        predicted_labels = model.predict(testing_features)
        file1 = open(save_path + "comment.txt", "w")
        file1.write(comment)
        file1.close()
        data11 = feature_extraction(save_path)
        my_prediction = model.predict(data11[0])
    return render_template('result.html',prediction = my_prediction[0])



if __name__ == '__main__':
	app.run(debug=True)