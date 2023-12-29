import json
from math import radians, sin, cos, sqrt, atan2
from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import time
from sqlalchemy.orm import Session
from collections import defaultdict

app = Flask(__name__)

# 读取城市数据
city_data = pd.read_csv('us-cities.csv')

review_data = pd.read_csv('newre.csv')

# 替换下面的数据库连接字符串为你的MySQL数据库连接信息
app.config['SQLALCHEMY_DATABASE_URI'] = f'mysql+pymysql://root:lym222@112.126.68.6:3307/Cloud?charset=utf8'
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # 设置为False以避免追踪修改导致性能问题

db = SQLAlchemy(app)
# 创建 Session
# db = pymysql.connect(host="112.126.68.6",
#                      user="root",
#                      password="lym222",
#                      database="Cloud",
#                      port=3307)

# Define your model class for the 'reviews' table
class Reviews(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    score = db.Column(db.Text)
    city = db.Column(db.Text)
    title = db.Column(db.Text)
    review = db.Column(db.Text)
    cluster = db.Column(db.Text)


class City(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    city = db.Column(db.Text)
    lat = db.Column(db.Text)
    lng = db.Column(db.Text)
    country = db.Column(db.Text)
    state = db.Column(db.Text)
    population = db.Column(db.Integer)

# 从数据库获取评论文本数据
# 每页城市数
cities_per_page = 50


@app.route('/')
def index():
    return render_template('index.html')

def haversine(lat1, lon1, lat2, lon2):
    # 将经纬度从度数转换为弧度
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine公式计算距离
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    radius_of_earth = 6371  # 地球半径（单位：公里）

    # 计算距离（单位：千米）
    distance = radius_of_earth * c

    return distance

def calculate_distances(target_city, target_state):
    # 查询城市距离并计算响应时间
    start_time = time.time()

    # 初始化结果列表
    city_distances = []

    # 获取目标城市的经纬度
    target_city_data = city_data[(city_data['city'] == target_city) & (city_data['state'] == target_state)]
    if not target_city_data.empty:
        target_lat = float(target_city_data['lat'])
        target_lng = float(target_city_data['lng'])

        # 计算目标城市与其他城市的距离并添加到列表中
        for _, city in city_data.iterrows():
            lat = float(city['lat'])
            lng = float(city['lng'])
            distance = haversine(target_lat, target_lng, lat, lng)
            city_distances.append(distance)

        # 按距离升序排序
        city_distances = sorted(city_distances)

    response_time = int((time.time() - start_time) * 1000)  # 计算响应时间（毫秒）
    return city_distances, response_time


def calculate_average_scores(target_city, target_state):
    # 计算平均评分和响应时间
    start_time = time.time()

    # 初始化结果列表
    city_scores = []

    # 获取目标城市的经纬度
    target_city_data = city_data[(city_data['city'] == target_city) & (city_data['state'] == target_state)]
    if not target_city_data.empty:
        target_lat = float(target_city_data['lat'])
        target_lng = float(target_city_data['lng'])

        # 计算目标城市与其他城市的距离和平均评分
        for _, city in city_data.iterrows():
            lat = float(city['lat'])
            lng = float(city['lng'])
            distance = haversine(target_lat, target_lng, lat, lng)

            # 过滤当前城市的评分数据
            city_reviews = review_data[review_data['city'] == city['city']]

            # 计算当前城市的平均评分
            average_score = city_reviews['score'].mean() if not city_reviews.empty else 0

            city_scores.append({'city': city['city'], 'distance': distance, 'average_score': average_score})

    # 按距离升序排序城市
    city_scores.sort(key=lambda x: x['distance'])

    response_time = int((time.time() - start_time) * 1000)  # 响应时间（毫秒）
    return city_scores[:cities_per_page], response_time

@app.route('/index_city', methods=['GET', 'POST'])
def index_city():
    if request.method == 'POST':
        city_name = request.form['city']
        state_name = request.form['state']

        # 计算城市距离
        distances, response_time = calculate_distances(city_name, state_name)

        # 将数据转换为JSON格式
        data = json.dumps({'distances': distances, 'response_time': response_time})
        return render_template('index_city.html', data=data)

    return render_template('index_city.html', data=None)


@app.route('/index_review', methods=['GET', 'POST'])
def index_review():
    if request.method == 'POST':
        city_name = request.form['city']
        state_name = request.form['state']

        # 计算平均评分
        scores, response_time = calculate_average_scores(city_name, state_name)

        # 将数据转换为JSON格式
        data = json.dumps({'scores': scores, 'response_time': response_time})
        return render_template('index_review.html', data=data)

    return render_template('index_review.html', data=None)


@app.route('/index_knn', methods=['GET', 'POST'])
def index_knn():
    return render_template('index_knn.html')

@app.route('/knn_reviews', methods=['POST'])
def knn_reviews():
    start_time = time.time()

    # 获取请求参数
    classes = int(request.form.get('classes', 3))
    k = int(request.form.get('k', 3))
    words = int(request.form.get('words', 5))

    # 从数据库获取评论文本数据
    with app.app_context():
        reviews = [review.review for review in Reviews.query.limit(300).all()]

    # 使用TF-IDF向量化文本数据
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(reviews)

    # 使用KMeans进行聚类
    kmeans = KMeans(n_clusters=classes, random_state=42)
    clusters = kmeans.fit_predict(X)

    # 更新数据库表中的cluster列
    for idx, cluster_id in enumerate(clusters):
        review = Reviews.query.get(idx + 1)  # Assuming the reviews are indexed starting from 1
        review.cluster = cluster_id
        db.session.commit()

    # 处理每个聚类并准备响应
    results = []
    for cluster_id in range(classes):
        cluster_reviews = Reviews.query.filter_by(cluster=cluster_id).limit(300).all()

        result = {
            'class_id': cluster_id,
            'cluster_reviews': [review.review for review in cluster_reviews],
            'popular_words': get_popular_words([review.review for review in cluster_reviews], words),
        }

        results.append(result)

    response_time = int((time.time() - start_time) * 1000)  # 计算响应时间（毫秒）

    # 创建一个字典用于存储每个 cluster 对应的 population 之和
    cluster_population = {}

    # 创建一个 Session
    session = Session(db)

    with app.app_context():
        temps = [(review.city, review.cluster) for review in Reviews.query.limit(300).all()]
    # print(reviews)

    with app.app_context():
        cities = [(city.city, city.population) for city in City.query.all()]
    # print(cities)

    # 创建一个字典，用于存储每个cluster的population总和
    cluster_population = defaultdict(int)

    # 遍历temps数组，将相同cluster的城市的population相加
    for city, cluster in temps:
        for c, population in cities:
            if city == c:
                cluster_population[cluster] += population

    # 将字典转换为列表形式
    result = [[cluster, population] for cluster, population in cluster_population.items()]

    print(result)

    # return render_template('knn_result.html', results=results, response_time=response_time)
    return render_template('show.html', data_from_backend=result)

def get_popular_words(reviews, num_words):
    # 使用简单的示例逻辑获取最受欢迎的单词
    # 在实际应用中，您可能需要使用更复杂的NLP技术
    words = ' '.join(reviews).split()
    # word_freq = pd.Series(words).value_counts()
    # popular_words = word_freq.head(num_words).index.tolist()
    stop_word = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
                 'yourselves',
                 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
                 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
                 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is',
                 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did',
                 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or''because', 'as', 'until',
                 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during',
                 'before', 'after', 'above', 'below', 'to', 'from', 'up',
                 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there',
                 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each',
                 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
                 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don',
                 'should', 'now']
    # 对评论进行分词，并统计单词出现的次数
    word_counts = {}
    for review in words:
        for word in review.split():
            if word not in stop_word:
                if word in word_counts:
                    word_counts[word] += 1
                else:
                    word_counts[word] = 1
    word_list = []
    top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:num_words]  # 获取最常见的十个单词及出现次数
    for word, count in top_words:
        word_list.append(word)
    return word_list




if __name__ == '__main__':
    app.run(debug=True)