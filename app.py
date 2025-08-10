import os, json, time, traceback
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from training.dataset import parse_history, make_dataset
from training.train import Trainer
import logging

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app)

IS_RENDER = os.environ.get('RENDER','false').lower() == 'true' or bool(os.environ.get('PORT'))

trainer = Trainer(model_dir=os.path.join(os.path.dirname(__file__), 'models'), is_render=IS_RENDER)

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'status':'ok','render_mode': IS_RENDER})

@app.route('/upload-history', methods=['POST'])
def upload_history():
    try:
        if request.is_json:
            payload = request.get_json()
            history = payload.get('history') or payload.get('data') or []
        elif 'file' in request.files:
            f = request.files['file']
            txt = f.read().decode('utf-8')
            history = parse_history(txt)
        else:
            return jsonify({'error':'No JSON or file provided'}), 400
        if not history or len(history) < 5:
            return jsonify({'error':'history too small (min 5)'}), 400
        sid = str(int(time.time()*1000))
        p = os.path.join(trainer.model_dir, f'history_{sid}.json')
        with open(p,'w') as fh:
            json.dump(history, fh)
        return jsonify({'session': sid, 'length': len(history)})
    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

@app.route('/train', methods=['POST'])
def train():
    try:
        body = request.get_json(force=True)
        history = body.get('history')
        if not history and body.get('session'):
            fp = os.path.join(trainer.model_dir, f'history_{body.get("session")}.json')
            if os.path.exists(fp):
                with open(fp,'r') as fh:
                    history = json.load(fh)
        if not history:
            return jsonify({'error':'No history provided'}), 400
        config = body.get('config', {})
        info = trainer.train_from_history(history, config)
        return jsonify({'status':'trained', 'model_name': info['name'], 'metrics': info['metrics']})
    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        body = request.get_json(force=True)
        history = body.get('history') or []
        model_name = body.get('model')
        seq_len = int(body.get('seq_len', 5))
        if not history or len(history) < seq_len:
            return jsonify({'error':'history too short for seq_len'}), 400
        pred = trainer.predict_from_history(history, seq_len=seq_len, model_name=model_name)
        return jsonify({'prediction': pred})
    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

@app.route('/models', methods=['GET'])
def list_models():
    try:
        models = trainer.list_models()
        return jsonify({'models': models})
    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

@app.route('/train-history', methods=['GET'])
def train_history():
    try:
        data = trainer.latest_history()
        if data is None:
            return jsonify({'error':'no history found'}), 404
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

if __name__ == '__main__':
    os.makedirs(os.path.join(os.path.dirname(__file__), 'models'), exist_ok=True)
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)
