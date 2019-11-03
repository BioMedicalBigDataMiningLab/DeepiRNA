from keras.layers import LSTM, Dense, Input, Dropout,BatchNormalization,Activation,LeakyReLU
from keras.models import Model
from keras.callbacks import EarlyStopping

import data_parameters as par
import win_unicode_console

win_unicode_console.enable()

early_stopping = EarlyStopping(monitor='acc', patience=par.patience, verbose=2)

# Define human model
def lstm_human():
    input1 = Input(shape=(par.timestep, par.human_max_length * par.x_dim))
    lstm_1 = LSTM(par.human_lstm, activation='relu', dropout=par.human_dropout,return_sequences=True)(input1)
    lstm_2 = LSTM(par.human_lstm, activation='relu', dropout=par.human_dropout,return_sequences=True)(lstm_1)
    lstm_3 = LSTM(par.human_lstm, activation='relu', dropout=par.human_dropout,return_sequences=False)(lstm_2)


    dense1 = Dense(par.human_dense1, activation='relu')(lstm_3)
    drop1 = Dropout(par.human_dropout)(dense1)
    out = Dense(par.output, activation='sigmoid')(drop1)

    model = Model(inputs=input1, outputs=out)

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


# Define mouse model
def lstm_mouse():
    input1 = Input(shape=(par.timestep, par.mouse_max_length * par.x_dim))
    lstm_1 = LSTM(par.mouse_lstm1, dropout=par.mouse_dropout, return_sequences=True)(input1)
    lstm_2 = LSTM(par.mouse_lstm2, dropout=par.mouse_dropout, return_sequences=False)(lstm_1)

    dense1 = Dense(par.mouse_dense1, activation='relu')(lstm_2)
    drop1 = Dropout(par.mouse_dropout)(dense1)
    dense2 = Dense(par.mouse_dense2, activation='relu')(drop1)
    drop2 = Dropout(par.mouse_dropout)(dense2)       
    out = Dense(par.output, activation='sigmoid')(drop2)

    model = Model(inputs=input1, outputs=out)

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


# Define drosophila model
def lstm_drosophila():
    input1 = Input(shape=(par.timestep, par.drosophila_max_length * par.x_dim))
    lstm_1 = LSTM(par.drosophila_lstm, dropout=par.drosophila_dropout, return_sequences=True)(input1)
    lstm_2 = LSTM(par.drosophila_lstm, dropout=par.drosophila_dropout, return_sequences=True)(lstm_1)
    lstm_3 = LSTM(par.drosophila_lstm, dropout=par.drosophila_dropout, return_sequences=False)(lstm_2)

    out = Dense(par.output, activation='softmax')(lstm_3)

    model = Model(inputs=input1, outputs=out)

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model
