from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import sqlite3
import pandas as pd
import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Conv1D, GlobalMaxPooling1D, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
app.secret_key = 'secret_key'

def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE email = ? AND password = ?', (email, password)).fetchone()
        conn.close()
        
        if user:
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['role'] = user['role']
            flash('Logged in successfully!', 'success')
            if user['role'] == 'admin':
                return redirect(url_for('admin_home'))
            return redirect(url_for('user_home'))
        else:
            flash('Invalid credentials', 'danger')
            
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        phone = request.form['phone']
        password = request.form['password']
        age = request.form['age']
        location = request.form['location']
        
        conn = get_db_connection()
        try:
            conn.execute('INSERT INTO users (username, email, phone, password, age, location) VALUES (?, ?, ?, ?, ?, ?)',
                         (username, email, phone, password, age, location))
            conn.commit()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Email already exists', 'danger')
        finally:
            conn.close()
            
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully!', 'info')
    return redirect(url_for('login'))

@app.route('/admin_home')
def admin_home():
    if session.get('role') != 'admin': return redirect(url_for('login'))
    return render_template('admin_home.html')

@app.route('/train')
def train():
    if session.get('role') != 'admin': return redirect(url_for('login'))
    return render_template('train.html')

@app.route('/admin_reports')
def admin_reports():
    if session.get('role') != 'admin': return redirect(url_for('login'))
    return render_template('admin_reports.html')

@app.route('/api/dataset_stats')
def dataset_stats():
    df = pd.read_csv('data/depression_tweets_2000.csv')
    stats = {
        'total': len(df),
        'counts': df['Label'].value_counts().to_dict(),
        'sample': df.head(50).to_dict(orient='records')
    }
    return jsonify(stats)

@app.route('/api/train', methods=['POST'])
def api_train():
    algorithm = request.json['algorithm']
    df = pd.read_csv('data/depression_tweets_2000.csv')
    X_raw = df['TweetText']
    y_raw = df['Label']
    
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    with open('model/label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    
    depressed_idx = list(le.classes_).index('Depressed')

    metrics = {}
    model_path = f'model/{algorithm}'

    if algorithm in ['SVM', 'RandomForest', 'LogisticRegression']:
        tfidf = TfidfVectorizer(stop_words='english')
        X = tfidf.fit_transform(X_raw)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if algorithm == 'SVM':
            model = LinearSVC()
        elif algorithm == 'RandomForest':
            model = RandomForestClassifier()
        else:
            model = LogisticRegression()
            
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, pos_label=depressed_idx),
            'recall': recall_score(y_test, y_pred, pos_label=depressed_idx),
            'f1_score': f1_score(y_test, y_pred, pos_label=depressed_idx)
        }
        
        model_path += '.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        with open('model/tfidf.pkl', 'wb') as f:
            pickle.dump(tfidf, f)

    elif algorithm in ['ANN', 'DNN', 'CNN']:
        if algorithm == 'CNN':
            max_words = 5000
            max_len = 100
            tokenizer = Tokenizer(num_words=max_words)
            tokenizer.fit_on_texts(X_raw)
            sequences = tokenizer.texts_to_sequences(X_raw)
            X = pad_sequences(sequences, maxlen=max_len)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = Sequential([
                Embedding(max_words, 64, input_length=max_len),
                Conv1D(128, 5, activation='relu'),
                GlobalMaxPooling1D(),
                Dense(64, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            
            with open('model/tokenizer.pkl', 'wb') as f:
                pickle.dump(tokenizer, f)
        else:
            tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
            X = tfidf.fit_transform(X_raw).toarray()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            if algorithm == 'ANN':
                model = Sequential([
                    Dense(64, activation='relu', input_dim=X.shape[1]),
                    Dense(32, activation='relu'),
                    Dense(1, activation='sigmoid')
                ])
            else: # DNN
                model = Sequential([
                    Dense(128, activation='relu', input_dim=X.shape[1]),
                    Dropout(0.3),
                    Dense(64, activation='relu'),
                    Dropout(0.3),
                    Dense(32, activation='relu'),
                    Dense(1, activation='sigmoid')
                ])
            
            with open('model/tfidf.pkl', 'wb') as f:
                pickle.dump(tfidf, f)

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
        
        y_pred_prob = model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, pos_label=depressed_idx)),
            'recall': float(recall_score(y_test, y_pred, pos_label=depressed_idx)),
            'f1_score': float(f1_score(y_test, y_pred, pos_label=depressed_idx))
        }
        
        model_path += '.h5'
        model.save(model_path)

    # Store in DB
    conn = get_db_connection()
    conn.execute('INSERT INTO performance (algorithm, accuracy, precision, recall, f1_score, model_path) VALUES (?, ?, ?, ?, ?, ?)',
                 (algorithm, metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1_score'], model_path))
    conn.commit()
    conn.close()
    
    return jsonify(metrics)

@app.route('/api/performance_logs')
def performance_logs():
    conn = get_db_connection()
    logs = conn.execute('SELECT * FROM performance ORDER BY id DESC').fetchall()
    conn.close()
    return jsonify([dict(log) for log in logs])

@app.route('/user_home')
def user_home():
    if not session.get('user_id'): return redirect(url_for('login'))
    return render_template('user_home.html')

@app.route('/test_data')
def test_data():
    if not session.get('user_id'): return redirect(url_for('login'))
    return render_template('test_data.html')

@app.route('/view_logs')
def view_logs():
    if not session.get('user_id'): return redirect(url_for('login'))
    return render_template('view_logs.html')

@app.route('/profile')
def profile():
    if not session.get('user_id'): return redirect(url_for('login'))
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (session['user_id'],)).fetchone()
    conn.close()
    return render_template('profile.html', user=user)

@app.route('/api/user_stats')
def user_stats():
    conn = get_db_connection()
    count = conn.execute('SELECT COUNT(*) FROM test_logs WHERE user_id = ?', (session['user_id'],)).fetchone()[0]
    history = conn.execute('SELECT * FROM test_logs WHERE user_id = ? ORDER BY timestamp DESC LIMIT 10', (session['user_id'],)).fetchall()
    conn.close()
    return jsonify({
        'count': count,
        'history': [dict(h) for h in history][::-1]
    })

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.json
    text = data['text']
    model_name = data['model']
    
    # Determine model file and type
    if model_name in ['ANN', 'DNN', 'CNN']:
        model_path = f'model/{model_name}.h5'
        is_keras = True
    else:
        model_path = f'model/{model_name}.pkl'
        is_keras = False

    if not os.path.exists(model_path) or not os.path.exists('model/label_encoder.pkl'):
        return jsonify({'prediction': 'Model not trained yet!'})
        
    with open('model/label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)

    if is_keras:
        model = load_model(model_path)
        if model_name == 'CNN':
            if not os.path.exists('model/tokenizer.pkl'):
                return jsonify({'prediction': 'Tokenizer not found!'})
            with open('model/tokenizer.pkl', 'rb') as f:
                tokenizer = pickle.load(f)
            sequences = tokenizer.texts_to_sequences([text])
            X = pad_sequences(sequences, maxlen=100)
        else:
            if not os.path.exists('model/tfidf.pkl'):
                return jsonify({'prediction': 'TFIDF not found!'})
            with open('model/tfidf.pkl', 'rb') as f:
                tfidf = pickle.load(f)
            X = tfidf.transform([text]).toarray()
        
        pred_prob = model.predict(X)
        prediction_idx = int(pred_prob[0][0] > 0.5)
    else:
        if not os.path.exists('model/tfidf.pkl'):
            return jsonify({'prediction': 'TFIDF not found!'})
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open('model/tfidf.pkl', 'rb') as f:
            tfidf = pickle.load(f)
            
        X = tfidf.transform([text])
        prediction_idx = model.predict(X)[0]
    
    prediction = le.inverse_transform([prediction_idx])[0]
    
    # Log result
    conn = get_db_connection()
    conn.execute('INSERT INTO test_logs (user_id, tweet_text, result) VALUES (?, ?, ?)',
                 (session['user_id'], text, prediction))
    conn.commit()
    conn.close()
    
    return jsonify({'prediction': prediction})

@app.route('/api/user_logs')
def user_logs():
    conn = get_db_connection()
    logs = conn.execute('SELECT * FROM test_logs WHERE user_id = ? ORDER BY timestamp DESC', (session['user_id'],)).fetchall()
    conn.close()
    return jsonify([dict(log) for log in logs])

if __name__ == '__main__':
    app.run(debug=True)
