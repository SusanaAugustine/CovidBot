"""
    @script-author: Susana Niketa A
    @script-description: A flask based CovidBot using the chatbotAI library.
"""
from flask import Flask, render_template, request, url_for

import torch
from newspaper import Article
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import warnings
warnings.filterwarnings('ignore')
nltk.download('punkt', quiet=True)
import feedparser
from bs4 import BeautifulSoup
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from chatbot import Chat, register_call
from transformers import BertForQuestionAnswering
model_bert = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
url = 'https://tfhub.dev/google/universal-sentence-encoder/4'
model_faq = hub.load(url) 


app = Flask(__name__)
app.static_folder = 'static'


dataset = pd.read_excel("C:/Users/susan/Downloads/myproject/myproject/app/faqs1 (2).xlsx")
state_data = pd.read_csv('C:/Users/susan/Downloads/myproject/myproject/app/state.csv')
country_data = pd.read_csv('C:/Users/susan/Downloads/myproject/myproject/app/country.csv')
def answer_question(question, answer_text):
    input_ids = tokenizer.encode(question, answer_text)
    print('Query has {:,} tokens.\n'.format(len(input_ids)))
    sep_index = input_ids.index(tokenizer.sep_token_id)
    num_seg_a = sep_index + 1
    num_seg_b = len(input_ids) - num_seg_a
    segment_ids = [0]*num_seg_a + [1]*num_seg_b
    assert len(segment_ids) == len(input_ids)
    start_scores, end_scores = model_bert(torch.tensor([input_ids]),token_type_ids=torch.tensor([segment_ids]))
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    answer = tokens[answer_start]
    for i in range(answer_start + 1, answer_end + 1):
        if tokens[i][0:2] == '##':
            answer += tokens[i][2:]
        else:
            answer += ' ' + tokens[i]

    return answer
def news_bert(A):
    b = '%20'
    Aa = A+' covid-19'
    for word in Aa:
        if word == ' ':
            Aa = Aa.replace(word,b)
    url = "http://news.google.com/news?q="+Aa+"&hl=en-US&sort=date&gl=IN&num=500&output=rss"
    feed_url = url
    soup = BeautifulSoup()
    text = soup.get_text()
    text = text.replace('\xa0', ' ')
    c=''
    feeds = feedparser.parse(feed_url).entries
    b = feeds[0]['link']
    article = Article(b)
    article.download()
    article.parse()
    article.nlp()
    corpus = article.text
    if len(corpus) > 512:
        corpus = corpus[:512]
    return corpus
	

	
	
def nlp(corpus):
    from textblob.classifiers import NaiveBayesClassifier as NBC
    from textblob import TextBlob
    training_corpus = [
                       ('What is the status of covid in kerela?', 'Class_B'),
                       ("Who is the latest celebrith to be tested positive?", 'Class_B'),
                       ('When will vaccines be found?', 'Class_B'),
                       ('When will colleges open in karnataka', 'Class_B'),
                       ('What is the active case count in Goa?', 'Class_A'),
                       ('Total deaths in bangalore', 'Class_A'),
                       ('number of tested positive cases in the world', 'Class_A')]
    test_corpus = [
                    ("Is joe tested positive?", 'Class_B'), 
                    ("increasing cases in kerela", 'Class_A'), 
                    ("chennais active cases", 'Class_A'), 
                    ("when will the vaccine be found", 'Class_B')]

    model = NBC(training_corpus) 
    op = model.classify(corpus)
    #print(model.accuracy(test_corpus))
    return op
def embed(input):
    return model_faq([input])
dataset['Question_Vector'] = dataset.Question.map(embed)
dataset['Question_Vector'] = dataset.Question_Vector.map(np.array)
questions = dataset.Question
QUESTION_VECTORS = np.array(dataset.Question_Vector)
COSINE_THRESHOLD = 0.5
def cosine_similarity(v1, v2):
        mag1 = np.linalg.norm(v1)
        mag2 = np.linalg.norm(v2)
        if (not mag1) or (not mag2):
            return 0
        return np.dot(v1, v2) / (mag1 * mag2)
def semantic_search(query, data, vectors):        
        query_vec = np.array(embed(query))
        res = []
        for i, d in enumerate(data):
            qvec = vectors[i].ravel()
            sim = cosine_similarity(query_vec, qvec)
            res.append((sim, d[:100], i))
        return sorted(res, key=lambda x : x[0], reverse=True)      
@register_call("news_faqs")
def news_faqs(query,session_id="general"):
        most_relevant_row = semantic_search(query, questions, QUESTION_VECTORS)[0]
        if most_relevant_row[0][0]>=COSINE_THRESHOLD:
            answer = dataset.Answer[most_relevant_row[2]]
        else:
            answer = answer_question(query, news_bert(query))
        return answer


unimportant=['and']
def capitalise(word):
		resp=""
		v = word.split()
		for x in v:
			if x not in unimportant:
				resp += (" " + x.title())
			else:
				resp += (" " + x)

		return resp
import urllib.request, json
with urllib.request.urlopen("https://api.covid19india.org/state_district_wise.json") as url:
    data_dist = json.loads(url.read().decode())
with urllib.request.urlopen("https://api.covid19api.com/summary") as url:
    data_country = json.loads(url.read().decode())
with urllib.request.urlopen("https://api.covid19india.org/data.json") as url:
    data_state = json.loads(url.read().decode())
distr = ['Andaman and Nicobar Islands','Andhra Pradesh','Arunachal Pradesh','Assam','Bihar','Chandigarh','Chhattisgarh','Delhi','Dadra and Nagar Haveli and Daman and Diu','Goa','Gujarat','Himachal Pradesh','Haryana','Jharkhand','Jammu and Kashmir','Karnataka','Kerala','Ladakh','Lakshadweep','Maharashtra','Meghalaya','Manipur','Madhya Pradesh','Mizoram','Nagaland','Odisha','Punjab','Puducherry','Rajasthan','Sikkim','Telangana','Tamil Nadu','Tripura','Uttar Pradesh','Uttarakhand','West Bengal']
@register_call("country_names")
def country_names(query,session_id="general"):
		query = capitalise(query).strip()
		for i in data_country['Countries']:
			if i['Country']==query:
				output = 'TotalCases : '+str(i['TotalConfirmed'])+' NewCases : '+str(i['NewConfirmed'])+' TotalDeaths : '+str(i['TotalDeaths'])+' NewDeaths : '+str(i['NewDeaths'])+' TotalRecovered : '+str(i['TotalRecovered'])
				break
			else:
				output = 'Input valid country name'
		return(output)
@register_call("state_names")
def state_names(query,session_id="general"):
		query = capitalise(query).strip()
		for i in data_state['statewise']:
			if i['state']==query:
				output = ' Confirmed : '+str(i['confirmed'])+' Active : '+str(i['active'])+' Recovered : '+str(i['recovered'])+' Deaths : '+str(i['deaths'])
				break
			else:
				output = 'Input valid country name'
		return(output)
@register_call("district_names")
def district_names(query,session_id="general"):
		query = capitalise(query).strip()
		for i in data_dist:
			for j in data_dist[i]['districtData']:
				if j == query:
					output =  ' Active Cases : '+str(data_dist[i]['districtData'][j]['active'])+' Confirmed : '+str(data_dist[i]['districtData'][j]['confirmed'])+' Dead : '+str(data_dist[i]['districtData'][j]['deceased'])+' Recovered : '+str(data_dist[i]['districtData'][j]['recovered'])
		return(output)
			
def tempp(response1):
		if nlp(response1) == 'Class_B':
			temp = 'news.template'
		else:
			temp = 'demo.template'
		return(temp)



	
app = Flask(__name__)

@app.route("/", methods=['GET','POST'])
def home():
    return render_template("index.html")

@app.route("/get", methods=['GET','POST'])
def get_bot_response():
	
		response = request.args.get('msg')
		temp2 = tempp(response)
		return str(Chat(temp2).say(response))


if __name__ == "__main__":
    app.run(debug=True)
