import tensorflow as tf
from tensorflow.keras import layers, models
from modules.custom_layers import CustomMultiHeadAttention


def define_gru_model(seq_length, output_dim, learning_rate, embedding_dim=64, gru_units=64, dropout_rate=0.2, recurrent_dropout_rate=0.2):
    inputs = layers.Input(shape=(seq_length,))
    x = layers.Embedding(input_dim=output_dim, output_dim=embedding_dim, mask_zero=True)(inputs)
    x = layers.GRU(gru_units, dropout=dropout_rate, recurrent_dropout=recurrent_dropout_rate)(x)
    outputs = layers.Dense(output_dim, activation="softmax", kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
        loss="sparse_categorical_crossentropy", 
        metrics=["accuracy"],
        weighted_metrics=["accuracy"]
    )
    return model




def define_lstm_model(seq_length, output_dim, learning_rate, embedding_dim=64, lstm_units=64, dropout_rate=0.2, recurrent_dropout_rate=0.2, num_layers=2):
    inputs = layers.Input(shape=(seq_length,))
    x = layers.Embedding(input_dim=output_dim, output_dim=embedding_dim, mask_zero=True)(inputs)
    for _ in range(num_layers - 1):
        x = layers.LSTM(lstm_units, return_sequences=True, dropout=dropout_rate, recurrent_dropout=recurrent_dropout_rate)(x)
    x = layers.LSTM(lstm_units, dropout=dropout_rate, recurrent_dropout=recurrent_dropout_rate)(x)
    outputs = layers.Dense(output_dim, activation="softmax", kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
        loss="sparse_categorical_crossentropy", 
        metrics=["accuracy"],
        weighted_metrics=["accuracy"]
    )
    return model


def define_bert_model(seq_length, output_dim, learning_rate, embedding_dim=64, num_heads=4, ffn_units=128, num_layers=2, dropout_rate=0.1):
    inputs = layers.Input(shape=(seq_length,))
    x = layers.Embedding(input_dim=output_dim, output_dim=embedding_dim, mask_zero=True)(inputs)
    for _ in range(num_layers):
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)(x, x)
        attention_output = layers.Dropout(dropout_rate)(attention_output)
        attention_output = layers.LayerNormalization(epsilon=1e-6)(attention_output + x)
        ffn = layers.Dense(ffn_units, activation='relu')(attention_output)
        ffn_output = layers.Dense(embedding_dim)(ffn)
        ffn_output = layers.Dropout(dropout_rate)(ffn_output)
        x = layers.LayerNormalization(epsilon=1e-6)(ffn_output + attention_output)
    x = layers.GlobalAveragePooling1D()(x)  # 出力の形状を修正
    outputs = layers.Dense(output_dim, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
        loss="sparse_categorical_crossentropy", 
        metrics=["accuracy"],
        weighted_metrics=["accuracy"]
    )
    return model

def define_transformer_model(seq_length, output_dim, learning_rate, embedding_dim=64, num_heads=4, ffn_units=128, dropout_rate=0.1):
    inputs = layers.Input(shape=(seq_length,), name='input_1')
    attention_mask = layers.Input(shape=(seq_length,), dtype=tf.float32, name='attention_mask')

    x = layers.Embedding(input_dim=output_dim, output_dim=embedding_dim, mask_zero=True)(inputs)
    attention_output = CustomMultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)(x, x, attention_mask=attention_mask)
    attention_output = layers.Dropout(dropout_rate)(attention_output)
    attention_output = layers.LayerNormalization(epsilon=1e-6)(attention_output + x)
    ffn = layers.Dense(ffn_units, activation='relu')(attention_output)
    ffn_output = layers.Dense(embedding_dim)(ffn)
    ffn_output = layers.Dropout(dropout_rate)(ffn_output)
    ffn_output = layers.LayerNormalization(epsilon=1e-6)(ffn_output + attention_output)
    x = layers.GlobalAveragePooling1D()(ffn_output)
    outputs = layers.Dense(output_dim, activation="softmax")(x)

    model = models.Model(inputs=[inputs, attention_mask], outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        weighted_metrics=["accuracy"]
    )
    return model

def define_gpt_model(seq_length, vocab_size, learning_rate, embedding_dim=64, num_heads=8, ffn_units=2048, dropout_rate=0.1):
    inputs = tf.keras.layers.Input(shape=(seq_length,), name='input_1')
    attention_mask = tf.keras.layers.Input(shape=(seq_length,), dtype=tf.float32, name='attention_mask')

    embedding_layer = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True)(inputs)
    
    attention_layer = CustomMultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)(embedding_layer, embedding_layer, attention_mask=attention_mask)
    
    add_norm_layer = tf.keras.layers.Add()([embedding_layer, attention_layer])
    norm_layer = tf.keras.layers.LayerNormalization(epsilon=1e-6)(add_norm_layer)
    
    ffn = tf.keras.Sequential([
        tf.keras.layers.Dense(ffn_units, activation='relu'),
        tf.keras.layers.Dense(embedding_dim)
    ])
    
    ffn_output = ffn(norm_layer)
    add_norm_layer2 = tf.keras.layers.Add()([norm_layer, ffn_output])
    norm_layer2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(add_norm_layer2)
    
    gap_layer = tf.keras.layers.GlobalAveragePooling1D()(norm_layer2)
    outputs = tf.keras.layers.Dense(vocab_size, activation="softmax")(gap_layer)  # vocab_sizeを使用

    model = tf.keras.Model(inputs=[inputs, attention_mask], outputs=outputs)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'],
        weighted_metrics=['accuracy']
    )

    return model



