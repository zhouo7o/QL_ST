import DQN_futures_func as func
#import DQN_futures_agent
import tensorflow as tf
import numpy as np
import random
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Hyper Parameters for DQN
GAMMA = 0.9 # discount factor for target Q
INITIAL_EPSILON = 0.5 # starting value of epsilon
FINAL_EPSILON = 0.01 # final value of epsilon
REPLAY_SIZE = 10000 # experience replay buffer size
BATCH_SIZE = 32 # size of minibatch

EPISODE = 10000 # Episode limitation
STEP = 300 # Step limitation in an episode
TEST = 10 # The number of experiment test every 100 episode
PRICELENGTH = 1000
BARLEGNTH = 8

class DQN():
	# DQN Agent
	def __init__(self):
		# init some parameters
		self.time_step = 0
		self.epsilon = INITIAL_EPSILON
		self.state_dim = BARLEGNTH
		self.action_dim = 3
	
		self.create_Q_network()
		self.create_training_method()

		# Init session
		self.session = tf.InteractiveSession()
		self.session.run(tf.initialize_all_variables())
		
	def create_Q_network(self):
		# network weights
		W1 = self.weight_variable([self.state_dim,20])
		b1 = self.bias_variable([20])
		W2 = self.weight_variable([20,self.action_dim])
		b2 = self.bias_variable([self.action_dim])
		# input layer
		self.state_input = tf.placeholder("float",[None,self.state_dim])
		# hidden layers
		h_layer = tf.nn.relu(tf.matmul(self.state_input,W1) + b1)
		# Q Value layer
		self.Q_value = tf.matmul(h_layer,W2) + b2
		
	def create_training_method(self):
		self.action_input = tf.placeholder("float",[None,self.action_dim]) # one hot presentation
		self.y_input = tf.placeholder("float",[None])
		Q_action = tf.reduce_sum(tf.mul(self.Q_value,self.action_input),reduction_indices = 1)
		self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
		self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)

	def egreedy_action(self,state):
		Q_value = self.Q_value.eval(feed_dict = {self.state_input:[state]})[0]
		if random.random() <= self.epsilon:
			return random.randint(0,self.action_dim - 1)
		else:
			return np.argmax(Q_value)
		self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/10000

	def action(self,state):
		return np.argmax(self.Q_value.eval(feed_dict = {self.state_input:[state]})[0])

	def weight_variable(self,shape):
		initial = tf.truncated_normal(shape)
		return tf.Variable(initial)

	def bias_variable(self,shape):
		initial = tf.constant(0.01, shape = shape)
		return tf.Variable(initial)

def main():
	#initialized price list
	priceList, increList = func.priceGeneration(PRICELENGTH, BARLEGNTH)
    priceState = np.floor(np.sign(increList)/2) + 1
    # initialize dqn agent
	agent = DQN()

if __name__ == '__main__':
  main()
