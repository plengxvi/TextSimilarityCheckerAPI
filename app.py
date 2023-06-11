from flask import Flask, jsonify, request
# from flask_restful import Api, Resource
from gensim.models.doc2vec import Doc2Vec
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk import word_tokenize, pos_tag
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import re

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "Hello, World!"

stop_words = stopwords.words('english')
lem = WordNetLemmatizer()

tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

def custom_lemmatize(t):
    lemmas = []
    for token, tag in pos_tag(t):
        lemma = lem.lemmatize(token, tag_map[tag[0]])
        lemmas.append(lemma)
    return lemmas

def preprocess(w):
    words = []
    for word in w:
        if word not in stop_words:
            word = re.sub(f'[^a-zA-Z]',' ', word).lower()
            words.append(word)
    words = custom_lemmatize(words)
    return words

@app.route('/checkSimilarity', methods=['GET'])
def checkSimilarity():
    if request.method == 'GET':

        postedData = request.get_json()
        text1 = postedData['text1']
        text2 = postedData['text2']
        loaded_model = Doc2Vec.load("textsimilaritychecker.model")

        pars = [text1, text2]

        par_tokens = [preprocess(word_tokenize(sentence)) for sentence in pars]
        par_vectors = [loaded_model.infer_vector([word for word in sent]).reshape(1, -1) for sent in par_tokens]
        similarity = round(cosine_similarity(par_vectors[0], par_vectors[1])[0][0] * 100, 2)
        similarity_str = str(similarity)+'%'
        print('*************************************************')
        print('similarity ', similarity)
        print('*************************************************')
        retJson = {
            "status": 200,
            "similarity_score": similarity_str,
            "msg": "Similarity score calculated successfully"
             }
        return jsonify(retJson)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app.debug = True
    app.run()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
