from flask import Flask, render_template, request, url_for, redirect
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from filter import User

app = Flask(__name__)
users_data = pd.read_csv('users_data.csv').fillna(0).set_index("Unnamed: 0")
df = pd.read_csv('final_fixed_scrapped.csv')
Articles = []
starttime = False
Article = None
UserId = None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/fetch', methods=['POST', 'GET'])
def fetch():
    global UserId
    UserId = request.get_json()['IP'].split('.')
    UserId = int(''.join(UserId))
    return redirect(url_for('articles'))


@app.route('/articles')
def articles():
    global starttime, Article, UserId, users_data
    if starttime:
        time_spent = datetime.now()-starttime
        starttime = False
        Imp_rat = time_spent.total_seconds()/(df.iloc[Article]['totalwords']*60/200)
        try:
            users_data.loc[UserId] = users_data.loc[UserId] + np.array([Imp_rat if str(i) == str(Article) else 0 for i in users_data.columns])
        except:
            users_data.loc[UserId] = pd.Series([Imp_rat if str(i) == str(Article) else 0 for i in users_data.columns], index=users_data.columns)
        users_data.to_csv('users_data.csv')
    run_filter(UserId,users_data)
    return render_template('articles.html', articles=Articles)


@app.route('/article', methods=['POST', 'GET'])
def article():
    global Article, starttime
    if request.method == 'POST':
        Article = int(request.get_json()['article'])
        return '200'
    else:
        if Article:
            starttime = datetime.now()
            art = df.iloc[Article]
            return render_template('article.html', article=art.to_dict())



def run_filter(UserId,users_data):
    global Articles
    Articles=[]
    final_list=[]
    try:
        users_data.loc[UserId]
        usr = User(UserId)
        collab=usr.collab_filter()
        collab=collab[collab>0]
        if len(collab)>0:
            final_list=list(collab.index)
        content_based=usr.contentbased_lda()
        for i in content_based.index:
            final_list.append(i)
        for i in final_list:
            art=df.iloc[i].to_dict()
            art['Article']=art['Article'][:400]+'...'
            Articles.append(art)
    except KeyError:
        Ids = users_data.sum(axis=1).nlargest(10).index
        for i in Ids:
            art=df.iloc[i].to_dict()
            art['Article']=art['Article'][:400]+'...'
            Articles.append(art)