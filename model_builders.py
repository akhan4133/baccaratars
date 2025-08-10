import tensorflow as tf

def build_lstm(seq_len, units=64, layers=2, dropout=0.15):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(units, return_sequences=(layers>1), input_shape=(seq_len,3)))
    for i in range(1, layers):
        model.add(tf.keras.layers.LSTM(units, return_sequences=(i < layers-1)))
        if dropout>0:
            model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    if dropout>0:
        model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(3, activation='softmax'))
    return model

def build_gru(seq_len, units=64, layers=2, dropout=0.15):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.GRU(units, return_sequences=(layers>1), input_shape=(seq_len,3)))
    for i in range(1, layers):
        model.add(tf.keras.layers.GRU(units, return_sequences=(i < layers-1)))
        if dropout>0:
            model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    if dropout>0:
        model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(3, activation='softmax'))
    return model

def build_dense(seq_len):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(seq_len,3)))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(3, activation='softmax'))
    return model

def build_hybrid(seq_len, units=64, lstm_layers=1, dropout=0.15):
    inp = tf.keras.Input(shape=(seq_len,3))
    x = inp
    for i in range(lstm_layers):
        x = tf.keras.layers.LSTM(units, return_sequences=True)(x)
        if dropout>0:
            x = tf.keras.layers.Dropout(dropout)(x)
    q = tf.keras.layers.Dense(units)(x)
    k = tf.keras.layers.Dense(units)(x)
    v = tf.keras.layers.Dense(units)(x)
    scores = tf.keras.layers.Dot(axes=(2,2))([q,k])
    attn = tf.keras.layers.Activation('softmax')(scores)
    attended = tf.keras.layers.Dot(axes=(2,1))([attn,v])
    out = tf.keras.layers.GlobalAveragePooling1D()(attended)
    out = tf.keras.layers.LayerNormalization()(out)
    out = tf.keras.layers.Dense(64, activation='relu')(out)
    out = tf.keras.layers.Dropout(dropout)(out)
    out = tf.keras.layers.Dense(3, activation='softmax')(out)
    model = tf.keras.Model(inputs=inp, outputs=out)
    return model

def build_transformer(seq_len, d_model=64, num_heads=4, ff_dim=128, dropout=0.15, num_layers=2):
    inp = tf.keras.Input(shape=(seq_len,3))
    x = tf.keras.layers.Dense(d_model, activation='relu')(inp)
    for _ in range(num_layers):
        x_norm = tf.keras.layers.LayerNormalization()(x)
        head_dim = max(1, d_model // num_heads)
        heads = []
        for _h in range(num_heads):
            q = tf.keras.layers.Dense(head_dim)(x_norm)
            k = tf.keras.layers.Dense(head_dim)(x_norm)
            v = tf.keras.layers.Dense(head_dim)(x_norm)
            scores = tf.keras.layers.Dot(axes=(2,2))([q,k])
            attn = tf.keras.layers.Activation('softmax')(scores)
            head = tf.keras.layers.Dot(axes=(2,1))([attn,v])
            heads.append(head)
        concat = tf.keras.layers.Concatenate(axis=-1)(heads)
        proj = tf.keras.layers.Dense(d_model)(concat)
        proj = tf.keras.layers.Dropout(dropout)(proj)
        x = tf.keras.layers.Add()([x, proj])
        x = tf.keras.layers.LayerNormalization()(x)
        ff = tf.keras.layers.Dense(ff_dim, activation='relu')(x)
        ff = tf.keras.layers.Dropout(dropout)(ff)
        ff = tf.keras.layers.Dense(d_model)(ff)
        x = tf.keras.layers.Add()([x, ff])
        x = tf.keras.layers.LayerNormalization()(x)
    out = tf.keras.layers.GlobalAveragePooling1D()(x)
    out = tf.keras.layers.Dropout(dropout)(out)
    out = tf.keras.layers.Dense(3, activation='softmax')(out)
    model = tf.keras.Model(inputs=inp, outputs=out)
    return model
