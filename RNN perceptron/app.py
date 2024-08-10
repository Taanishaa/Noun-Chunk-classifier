from flask import Flask, render_template, request, jsonify
from rnn import RNN
import numpy as np
import pandas as pd

import spacy

def convert_to_pos_tags(sentence):
    # Load the English language model in spaCy
    nlp = spacy.load("en_core_web_sm")
    
    # Process the input sentence using spaCy
    doc = nlp(sentence)
    
    # Map POS tags to the given numeric values
    converted_tags = []
    for token in doc:
        if token.tag_.startswith('NN'):  # If the tag starts with 'NN', it's a noun
            converted_tags.append(1)
        elif token.tag_.startswith('DT'):  # If the tag starts with 'DT', it's a determiner
            converted_tags.append(2)
        elif token.tag_.startswith('JJ'):  # If the tag starts with 'JJ', it's an adjective
            converted_tags.append(3)
        else:
            converted_tags.append(4)  # Assign 4 for other POS tags
    
    return converted_tags

# Example usage
sentence = "Boys play joyfully with brown balls in the green big field"
pos_tags = convert_to_pos_tags(sentence)
print(pos_tags)



app = Flask(__name__)
def preprocess(df):
    replacement_map = {
        1: [1, 0, 0, 0],
        2: [0, 1, 0, 0],
        3: [0, 0, 1, 0],
        4: [0, 0, 0, 1]
        # Add more mappings as needed
    }
    k=[[replacement_map.get(x, [0, 0, 0, 0]) for x in sublist] for sublist in df['pos_tags']]
    insert_list = [1, 1,1,1]
    updated_list = [[insert_list]+sublist for sublist in k]  
    df['pos_tags_hot_encoded']=updated_list
    X=np.array(df['pos_tags_hot_encoded'])
    X=X.reshape(X.shape[0],1)
    return X

Wax=np.array([[ 1.5234745 ],
 [-0.03025682],
 [ 0.53068704],
 [ 2.52354128],
 [-1.41983243],
 [ 1.33864204],
 [-0.6169723 ],
 [ 0.99901114]])
Wya=np.array([[0.38309115]])
b=np.array([[-1.34831732]])
net=RNN(Wax,Wya,b)


def new_x(X):
    k={}
    k["pos_tags"]=[X]
    df = pd.DataFrame(k)
    df=preprocess(df)
    return df

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check_palindrome', methods=['POST'])
def check_palindrome():
    input_string = request.form.get('inputString')
    X=convert_to_pos_tags(input_string)
    flag=False
    for i in X:
        if i!=2 and i!=1 and i!=3 and i!=4:
            k="Please enter valid sequence."
            flag=True
            break
    if flag:
        return jsonify({'result': k})
    else:
        X=new_x(X)
        out =net.prediction(X,Wax,Wya,b)    
        result = out[0]
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
