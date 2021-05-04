import random
import numpy as numpy
import flappy_bird_gym
from collections import deque
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import load_model, save_model, Sequential
from tensorflow.keras.optimizers import RMSprop

#Neural network for Agent
def NeuralNetwork(input_shape, output_shape):
    model = Sequential()
    model.add(Dense(512, input_shape=input_shape, activation="relu", kernel_initializer="he_uniform"))
    model.add(Dense(256, activation="relu", kernel_initializer="he_uniform"))
    model.add(Dense(64, activation="relu", kernel_initializer="he_uniform"))
    model.add(Dense(output_shape, activation="linear", kernel_initializer="he_uniform"))
    model.compile(loss="mse", optimizer=RMSprop(lr=0.0001, rho=0.95, epsilon=0.01), metrics=['accuracy'])
    model.summary()
    return model

#Brain of Agent || BluePrint of  Agent
class DQNAgent:
    #constructor
    def __init__(self):
        self.env = flappy_bird_gym.make('FlappyBird-v0')
        self.episodes = 1000
        self.state_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.n
        self.memory = deque(maxlen=2000)

        #Hyperparameters

        #gives more wight on imediate rewards
        self.gamma = 0.95
        #probability of taking a random action
        self.epsilon = 1
        self.epsilon_decay = 0.9999
        #once we decay epsilon, probability of takind a random action
        self.epsilon_min = 0.01
        #amount of datapoints added to neural network
        self.batch_number = 64


        self.train_start = 1000
        self.jump_prob = 0.01

        self.model = NeuralNetwork(input_shape=(self.state_space,), output_shape=self.action_space)


    def act(self, state):
        if np.random.random() > self.epsilon:
            return np.argmax(self.model.predict(state))
        return 1 if np.random.random() > self.jump_probe else 0


if __name__ == "__main__":
    agent = DQNAgent()



    print("hello")

