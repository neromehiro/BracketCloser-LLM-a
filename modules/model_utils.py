import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import regularizers
from modules.custom_layers import CustomMultiHeadAttention


'''
モデルの入力可能な変数はこちら
def define_gru_model(seq_length, output_dim, learning_rate, embedding_dim=64, gru_units=64, dropout_rate=0.2, recurrent_dropout_rate=0.2, regularizer_type='l2', regularizer_value=0.01):
def define_lstm_model(seq_length, output_dim, learning_rate, embedding_dim=64, lstm_units=64, dropout_rate=0.2, recurrent_dropout_rate=0.2, num_layers=2, regularizer_type='l2', regularizer_value=0.01):
def define_bert_model(seq_length, output_dim, learning_rate, embedding_dim=64, num_heads=4, ffn_units=128, num_layers=2, dropout_rate=0.1, use_attention_mask=False, regularizer_type='l2', regularizer_value=0.01):
def define_bert_model(seq_length, output_dim, learning_rate, embedding_dim=64, num_heads=4, ffn_units=128, num_layers=2, dropout_rate=0.1, use_attention_mask=False, regularizer_type='l2', regularizer_value=0.01):
def define_transformer_model(seq_length, output_dim, learning_rate, embedding_dim=64, num_heads=4, ffn_units=128, dropout_rate=0.1, regularizer_type='l2', regularizer_value=0.01):
def define_gpt_model(seq_length, vocab_size, learning_rate, embedding_dim=64, num_heads=8, ffn_units=2048, dropout_rate=0.1, regularizer_type='l2', regularizer_value=0.01):
'''
def define_gru_model(seq_length, output_dim, learning_rate, embedding_dim=64, gru_units=64, dropout_rate=0.2, recurrent_dropout_rate=0.2, regularizer_type='l2', regularizer_value=0.01):
    if regularizer_type == 'l1':
        regularizer = regularizers.l1(regularizer_value)
    elif regularizer_type == 'l2':
        regularizer = regularizers.l2(regularizer_value)
    else:
        regularizer = None

    inputs = layers.Input(shape=(seq_length,))
    x = layers.Embedding(input_dim=output_dim, output_dim=embedding_dim, mask_zero=True)(inputs)
    x = layers.GRU(gru_units, dropout=dropout_rate, recurrent_dropout=recurrent_dropout_rate, kernel_regularizer=regularizer)(x)
    outputs = layers.Dense(output_dim, activation="softmax", kernel_regularizer=regularizer)(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
        loss="sparse_categorical_crossentropy", 
        metrics=["accuracy"],
        weighted_metrics=["accuracy"]
    )
    return model


def define_lstm_model(seq_length, output_dim, learning_rate, embedding_dim=64, lstm_units=64, dropout_rate=0.2, recurrent_dropout_rate=0.2, num_layers=2, regularizer_type='l2', regularizer_value=0.01):
    if regularizer_type == 'l1':
        regularizer = regularizers.l1(regularizer_value)
    elif regularizer_type == 'l2':
        regularizer = regularizers.l2(regularizer_value)
    else:
        regularizer = None

    inputs = layers.Input(shape=(seq_length,))
    x = layers.Embedding(input_dim=output_dim, output_dim=embedding_dim, mask_zero=True)(inputs)
    for _ in range(num_layers - 1):
        x = layers.LSTM(lstm_units, return_sequences=True, dropout=dropout_rate, recurrent_dropout=recurrent_dropout_rate, kernel_regularizer=regularizer)(x)
    x = layers.LSTM(lstm_units, dropout=dropout_rate, recurrent_dropout=recurrent_dropout_rate, kernel_regularizer=regularizer)(x)
    outputs = layers.Dense(output_dim, activation="softmax", kernel_regularizer=regularizer)(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
        loss="sparse_categorical_crossentropy", 
        metrics=["accuracy"],
        weighted_metrics=["accuracy"]
    )
    return model


def define_bert_model(seq_length, output_dim, learning_rate, embedding_dim=64, num_heads=4, ffn_units=128, num_layers=2, dropout_rate=0.1, use_attention_mask=False, regularizer_type='l2', regularizer_value=0.01):
    if regularizer_type == 'l1':
        regularizer = regularizers.l1(regularizer_value)
    elif regularizer_type == 'l2':
        regularizer = regularizers.l2(regularizer_value)
    else:
        regularizer = None

    inputs = layers.Input(shape=(seq_length,), name='input_1')
    if use_attention_mask:
        attention_mask = layers.Input(shape=(seq_length,), dtype=tf.float32, name='attention_mask')
        input_list = [inputs, attention_mask]
    else:
        attention_mask = None
        input_list = inputs

    x = layers.Embedding(input_dim=output_dim, output_dim=embedding_dim, mask_zero=True, embeddings_regularizer=regularizer)(inputs)
    for _ in range(num_layers):
        if use_attention_mask:
            mask_expanded = tf.expand_dims(tf.expand_dims(attention_mask, axis=1), axis=1)  # (batch_size, 1, 1, seq_length)
        else:
            mask_expanded = None
        
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)(x, x, attention_mask=mask_expanded)
        attention_output = layers.Dropout(dropout_rate)(attention_output)
        attention_output = layers.LayerNormalization(epsilon=1e-6)(attention_output + x)
        ffn = layers.Dense(ffn_units, activation='relu', kernel_regularizer=regularizer)(attention_output)
        ffn_output = layers.Dense(embedding_dim, kernel_regularizer=regularizer)(ffn)
        ffn_output = layers.Dropout(dropout_rate)(ffn_output)
        x = layers.LayerNormalization(epsilon=1e-6)(ffn_output + attention_output)
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(output_dim, activation="softmax", kernel_regularizer=regularizer)(x)

    model = models.Model(inputs=input_list, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
        loss="sparse_categorical_crossentropy", 
        metrics=["accuracy"],
        weighted_metrics=["accuracy"]
    )
    return model


def define_transformer_model(seq_length, output_dim, learning_rate, embedding_dim=64, num_heads=4, ffn_units=128, dropout_rate=0.1, regularizer_type='l2', regularizer_value=0.01):
    if regularizer_type == 'l1':
        regularizer = regularizers.l1(regularizer_value)
    elif regularizer_type == 'l2':
        regularizer = regularizers.l2(regularizer_value)
    else:
        regularizer = None

    inputs = layers.Input(shape=(seq_length,), name='input_1')
    attention_mask = layers.Input(shape=(seq_length,), dtype=tf.float32, name='attention_mask')

    x = layers.Embedding(input_dim=output_dim, output_dim=embedding_dim, mask_zero=True, embeddings_regularizer=regularizer)(inputs)
    attention_output = CustomMultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)(x, x, attention_mask=attention_mask)
    attention_output = layers.Dropout(dropout_rate)(attention_output)
    attention_output = layers.LayerNormalization(epsilon=1e-6)(attention_output + x)
    ffn = layers.Dense(ffn_units, activation='relu', kernel_regularizer=regularizer)(attention_output)
    ffn_output = layers.Dense(embedding_dim, kernel_regularizer=regularizer)(ffn)
    ffn_output = layers.Dropout(dropout_rate)(ffn_output)
    ffn_output = layers.LayerNormalization(epsilon=1e-6)(ffn_output + attention_output)
    x = layers.GlobalAveragePooling1D()(ffn_output)
    outputs = layers.Dense(output_dim, activation="softmax", kernel_regularizer=regularizer)(x)

    model = models.Model(inputs=[inputs, attention_mask], outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        weighted_metrics=["accuracy"]
    )
    return model


def define_gpt_model(seq_length, vocab_size, learning_rate, embedding_dim=64, num_heads=8, ffn_units=2048, dropout_rate=0.1, regularizer_type='l2', regularizer_value=0.01):
    if regularizer_type == 'l1':
        regularizer = regularizers.l1(regularizer_value)
    elif regularizer_type == 'l2':
        regularizer = regularizers.l2(regularizer_value)
    else:
        regularizer = None

    inputs = tf.keras.layers.Input(shape=(seq_length,), name='input_1')
    attention_mask = tf.keras.layers.Input(shape=(seq_length,), dtype=tf.float32, name='attention_mask')

    embedding_layer = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True, embeddings_regularizer=regularizer)(inputs)
    
    attention_layer = CustomMultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)(embedding_layer, embedding_layer, attention_mask=attention_mask)
    
    add_norm_layer = tf.keras.layers.Add()([embedding_layer, attention_layer])
    norm_layer = tf.keras.layers.LayerNormalization(epsilon=1e-6)(add_norm_layer)
    
    ffn = tf.keras.Sequential([
        tf.keras.layers.Dense(ffn_units, activation='relu', kernel_regularizer=regularizer),
        tf.keras.layers.Dense(embedding_dim, kernel_regularizer=regularizer)
    ])
    
    ffn_output = ffn(norm_layer)
    add_norm_layer2 = tf.keras.layers.Add()([norm_layer, ffn_output])
    norm_layer2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(add_norm_layer2)
    
    gap_layer = tf.keras.layers.GlobalAveragePooling1D()(norm_layer2)
    outputs = tf.keras.layers.Dense(vocab_size, activation="softmax", kernel_regularizer=regularizer)(gap_layer)

    model = tf.keras.Model(inputs=[inputs, attention_mask], outputs=outputs)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'],
        weighted_metrics=['accuracy']
    )

    return model