# -*- coding: utf-8 -*-
"""CA1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/17_cEChRPcCgE2jtQxF3qD5goaDChbHwn
"""

from google.colab import files
uploaded=files.upload()

import pandas as pd
df=pd.read_csv("Bank_Personal_Loan_Modelling.csv")
df.info()

df.drop(['ID','ZIP Code'],axis=1,inplace=True)

df.columns

columns=['Age', 'Experience', 'Income', 'Family', 'CCAvg',
       'Education', 'Mortgage', 'Securities Account',
       'CD Account', 'Online', 'CreditCard','Personal Loan']
df=df[columns]

x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=666)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

import tensorflow.keras
from keras.models import Sequential
from keras.layers import Dense

import random
from keras.models import Sequential
from keras.layers import Dense


input_shape = x_train.shape[1:]
hidden_layer_neurons = [32, 64, 32]
output_neurons = 1

knowledge = {
    "input_shape": [x_train.shape[1:]],
    "hidden_layer_neurons": [[16, 32, 16], [32, 64, 32], [64, 128, 64]],
    "output_neurons": [1]
}

pop_size = 10
num_gen = 5

mut_rate = 0.1
elitism_rate = 0.2

# Define the fitness function
def fitness_function(params):
    model = Sequential()
    model.add(Dense(params["hidden_layer_neurons"][0], activation='relu', input_shape=params["input_shape"]))

    for neurons in params["hidden_layer_neurons"][1:]:
        model.add(Dense(neurons, activation='relu'))

    model.add(Dense(params["output_neurons"], activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test), verbose=0)
    return history.history['accuracy'][-1]

# Initialize the population
population = []
for i in range(pop_size):
    params = {}
    for key in knowledge:
        params[key] = random.choice(knowledge[key])
    population.append(params)

# Evolve the population
for gen in range(num_gen):
    
    fitness = [fitness_function(params) for params in population]
    
    sorted_pop = [params for _, params in sorted(zip(fitness, population), reverse=True)]
    
    num_elite = int(elitism_rate * pop_size)
    elite = sorted_pop[:num_elite]
    
    parents = []
    for i in range(pop_size - num_elite):
        parent1 = random.choice(sorted_pop[:pop_size//2])
        parent2 = random.choice(sorted_pop[:pop_size//2])
        parents.append((parent1, parent2))
    
    offspring = []
    for parent1, parent2 in parents:
        child = {}
        for key in knowledge:
            if random.random() < mut_rate:
                child[key] = random.choice(knowledge[key])
            else:
                child[key] = random.choice([parent1[key], parent2[key]])
        offspring.append(child)
    
    population = elite + offspring
    
best_params = elite[-1]
best_fitness = fitness_function(best_params)
print("Best Parameters:", best_params)
print("Fitness:", best_fitness)

