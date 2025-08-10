import os, json, time, traceback
import numpy as np
import tensorflow as tf
from .dataset import make_dataset
from .model_builders import build_lstm, build_gru, build_dense, build_hybrid, build_transformer

class Trainer:
    def __init__(self, model_dir='models', is_render=False):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.is_render = is_render

    def _choose_builder(self, model_type):
        if model_type == 'lstm': return build_lstm
        if model_type == 'gru': return build_gru
        if model_type == 'dense': return build_dense
        if model_type == 'hybrid': return build_hybrid
        if model_type == 'transformer': return build_transformer
        return build_dense

    def train_from_history(self, history, config):
        seq_len = int(config.get('seq_len', 5))
        model_type = config.get('model_type', 'lstm')
        units = int(config.get('units', 64))
        layers = int(config.get('layers', 2))
        dropout = float(config.get('dropout', 0.15))
        epochs = int(config.get('epochs', 10))
        batch = int(config.get('batch', 32))
        lr = float(config.get('lr', 0.001))
        val_split = float(config.get('val_split', 0.15))

        # If running on Render, force lightweight defaults
        if self.is_render:
            epochs = min(epochs, 4)
            units = min(units, 48)
            batch = max(8, min(batch, 32))

        X, Y = make_dataset(history, seq_len=seq_len)
        if len(X) < 8:
            raise ValueError('Not enough samples to train (need > 8)')

        builder = self._choose_builder(model_type)
        if model_type in ('lstm','gru'):
            model = builder(seq_len, units=units, layers=layers, dropout=dropout)
        elif model_type == 'transformer':
            model = builder(seq_len, d_model=units, num_heads=int(config.get('num_heads',4)), ff_dim=int(config.get('ff_dim',128)), dropout=dropout, num_layers=layers)
        elif model_type == 'hybrid':
            model = builder(seq_len, units=units, lstm_layers=layers, dropout=dropout)
        else:
            model = builder(seq_len)

        opt = tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
        ]

        history_cb = model.fit(X, Y, epochs=epochs, batch_size=batch, validation_split=val_split, callbacks=callbacks, verbose=2)

        last = history_cb.history
        metrics = {'loss': last.get('loss')[-1], 'val_loss': last.get('val_loss')[-1] if 'val_loss' in last else None,
                   'accuracy': last.get('accuracy')[-1] if 'accuracy' in last else None,
                   'val_accuracy': last.get('val_accuracy')[-1] if 'val_accuracy' in last else None}

        model_name = f"{model_type}_{int(time.time())}"
        save_path = os.path.join(self.model_dir, model_name)
        model.save(save_path)

        meta = {'name': model_name, 'model_type': model_type, 'config': {'seq_len': seq_len, 'units': units, 'layers': layers, 'dropout': dropout}, 'metrics': metrics}
        with open(os.path.join(save_path, 'meta.json'),'w') as fh:
            json.dump(meta, fh)

        hist_out = {
            'epochs': list(range(1, len(last.get('loss', [])) + 1)),
            'loss': last.get('loss', []),
            'val_loss': last.get('val_loss', []),
            'accuracy': last.get('accuracy', []),
            'val_accuracy': last.get('val_accuracy', [])
        }
        with open(os.path.join(save_path, 'history.json'),'w') as fh:
            json.dump(hist_out, fh)

        return {'name': model_name, 'metrics': metrics, 'meta': meta}

    def list_models(self):
        out = []
        for fn in os.listdir(self.model_dir):
            p = os.path.join(self.model_dir, fn)
            if os.path.isdir(p):
                meta_file = os.path.join(p, 'meta.json')
                meta = {}
                if os.path.exists(meta_file):
                    try:
                        with open(meta_file,'r') as fh:
                            meta = json.load(fh)
                    except: meta = {}
                out.append({'name': fn, 'meta': meta})
        return out

    def predict_from_history(self, history, seq_len=5, model_name=None):
        import numpy as np, os, tensorflow as tf
        if model_name:
            model_path = os.path.join(self.model_dir, model_name)
            if not os.path.exists(model_path):
                raise FileNotFoundError('Model not found')
            model = tf.keras.models.load_model(model_path)
        else:
            all_models = sorted([d for d in os.listdir(self.model_dir) if os.path.isdir(os.path.join(self.model_dir,d))])
            if not all_models:
                raise FileNotFoundError('No models available')
            model = tf.keras.models.load_model(os.path.join(self.model_dir, all_models[-1]))
        seq = history[-seq_len:]
        from .dataset import token_map
        x = np.array([[token_map.get(s, [0,0,0]) for s in seq]], dtype='float32')
        probs = model.predict(x)[0].tolist()
        return {'P': probs[0], 'B': probs[1], 'T': probs[2]}

    def latest_history(self):
        # return history.json from latest model if exists
        models = sorted([d for d in os.listdir(self.model_dir) if os.path.isdir(os.path.join(self.model_dir,d))])
        if not models:
            return None
        latest = os.path.join(self.model_dir, models[-1], 'history.json')
        if not os.path.exists(latest):
            return None
        with open(latest,'r') as fh:
            return json.load(fh)
