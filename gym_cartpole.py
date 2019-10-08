#%%

from __future__ import absolute_import, division, print_function, unicode_literals

from typing import List, Tuple, Optional, Any, Dict

import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
import abc
# from PIL import Image
import cv2

#%%

is_ipython = 'inline' in matplotlib.get_backend()

print("is_ipython", is_ipython)

if is_ipython:
    from IPython import display

#%%
DEMO = False

#%%
if DEMO:
    env = gym.make('CartPole-v0')
    env.reset()

    for _ in range(350):
        env.render()
        env.step(env.action_space.sample())

    env.close()

#%%

import tensorflow as tf

layers = tf.keras.layers
optimizers = tf.keras.optimizers

# from tensorflow_core.python import layers

# layers = tf.keras.layers

#%%


def DQN(inputShape):
    x = layers.Input(shape=inputShape)
    inputs = x
    # Hidden layers
    # Dense layers
    x = layers.Flatten()(x)
    x = layers.Dense(24, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    # Output layer
    x = layers.Dense(2, activation='softmax')(x)
    outputs = x
    # Instantiate the model given inputs and outputs.
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


#%%
Experience = namedtuple(
    'Experience',
    ('state', 'action', 'next_state', 'reward')
)

#%%

if DEMO:
    e = Experience(2, 3, 1, 4)
    print(e)

#%%

class ReplayMemory():
    def __init__(self, capacity: int, batch_size: int):
        self.memory: List[Experience] = []
        self.capacity: int = capacity
        self.push_count: int = 0
        self.batch_size: int = batch_size


    def push(self, experience: Experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            # overwriting the oldest experiences first.
            # kinda like a circular queue that overwrites when full
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1


    def can_provide_sample(self) -> bool:
        return len(self.memory) >= self.batch_size


    def sample(self) -> List[Experience]:
        if self.can_provide_sample():
            return random.sample(self.memory, self.batch_size)
        else:
            return []

#%%


class Strategy(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_exploration_rate(self, current_step: int) -> float:
        raise NotImplementedError('returns exploration rate in current step')


class EpsilonGreedyStrategy(Strategy):
    def __init__(self, start: float, end: float, decay: float):
        self.start: float = start
        self.end: float = end
        self.decay: float = decay


    def get_exploration_rate(self, current_step: int) -> float:
        return self.end + (self.start - self.end) * \
            math.exp(-1.0 * current_step * self.decay)
#%%

class CartPoleEnvManager():
    def reset(self):
        """reset() on the gym environment when we want the environment to be reset to a starting state.
        """
        # reset the environment to get an initial observation of it
        self.env.reset()
        # we’re at the start of an episode and have not yet rendered the screen of the initial observation
        self.current_screen = None


    def __init__(self):
        self.env = gym.make('CartPole-v0').unwrapped
        self.reset()
        # done will track whether or not any taken action has ended an episode.
        self.done = False


    def close(self):
        self.env.close()


    def render(self, mode='human'):
        return self.env.render(mode)

    def renderArray(self):
        return self.env.render('rgb_array')


    def num_actions_available(self):
        return self.env.action_space.n


    def take_action(self, action):
        observation, reward, self.done, diag = self.env.step(action)
        return reward


    def just_starting(self):
        return self.current_screen is None

    # # Screen

    def get_screen_height(self):
        return self.renderArray().shape[0]

    def get_screen_width(self):
        return self.renderArray().shape[1]

    def get_screen_shape(self):
        return self.renderArray().shape

    # #

    def get_state(self):
        if self.just_starting() or self.done:
            # Return black screen
            # get screen to know its shape
            self.current_screen = self.renderArray()
            # zero array with the shape of the screen
            black_screen = np.zeros_like(self.current_screen)
            return [black_screen, black_screen]
        else:
            # Return screens
            # previous screen
            s1 = self.current_screen
            # updated screen
            s2 = self.renderArray()
            # Save current screen for later
            self.current_screen = s2
            # to have historical information
            # return a list of screens
            return [s2, s1]

#%%

class PolicyNet(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def qvalues(self, state):
        raise NotImplementedError('returns qvalues of a given state')

    @staticmethod
    @abc.abstractmethod
    def extract_tensors(state):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_current(self, state):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_next(self, state):
        raise NotImplementedError()

    @abc.abstractmethod
    def step(self, state):
        raise NotImplementedError()


class PolicyNetTF(PolicyNet):
    def __init__(self, model: tf.keras.Model):
        self.model = model
        self.loss_history: List[float] = []
        

    def set_model(self, model):
        self.model = model
    

    def qvalues(self, state):
        return self.model.predict(tf.expand_dims(state, 0))

    # # Screen

    def crop_screen(self, screen: np.ndarray) -> np.ndarray:
        screen_height = screen.shape[0]

        # Strip off top and bottom
        top = int(screen_height * 0.4)
        bottom = int(screen_height * 0.8)
        # crop height, maintain width and channels
        screen = screen[top:bottom, :, :]
        return screen

    def transform_screen_data(self, screen: np.ndarray) -> np.ndarray:
        # Rescale
        screen = cv2.resize(screen, (90, 40), interpolation=cv2.INTER_LINEAR)
        # Convert to float ndarray
        screen = np.ascontiguousarray(screen, dtype=np.float32)
        # Normalize
        screen = screen / 255
        # # Convert to tensor
        # screen = tf.convert_to_tensor(screen)
        # # add a batch dimension
        # screen = tf.expand_dims(screen, 0)
        return screen

    def preprocess_screen(self, screen: np.ndarray) -> np.ndarray:
        # PyTorch expects CHW
        screen = self.crop_screen(screen)
        return self.transform_screen_data(screen)


    def get_input_shape(self, screen: np.ndarray) -> Tuple[int, ...]:
        return self.preprocess_screen(screen).shape


    def get_state(self, screens: List[np.ndarray]) -> np.ndarray:
        screens = [self.preprocess_screen(screen) for screen in screens]
        return screens[0] - screens[1]


    @staticmethod
    def extract_tensors(experiences):
        """
        Convert batch of Experiences (tuples) to Experience of batches
        ```python
        # batch of Experiences
        [Experience(state=1, action=1, next_state=1, reward=1),
         Experience(state=2, action=2, next_state=2, reward=2),
         Experience(state=3, action=3, next_state=3, reward=3)]
        # experience of batches
        Experience(state=(1, 2, 3), action=(1, 2, 3),
                   next_state=(1, 2, 3), reward=(1, 2, 3))
        ```
        """
        experience_of_batches = zip(*experiences)
        batch = Experience(*experience_of_batches)

        t1 = tf.convert_to_tensor(batch.state)
        t2 = tf.convert_to_tensor(batch.action)
        t3 = tf.convert_to_tensor(batch.reward)
        t4 = tf.convert_to_tensor(batch.next_state)

        return (t1, t2, t3, t4)


    def step(self, current_q_values, target):
        target = np.expand_dims(target, 1)
        # mse
        # loss = ((current_q_values-target)**2)
        loss = tf.keras.losses.MSE(target, current_q_values)
        loss_mean = tf.math.reduce_mean(loss)

        self.loss_history.append(loss_mean)
        grads = self.tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss_mean

    def update(self, states, actions, target):

        with tf.GradientTape() as tape:
            # TODO:
            tape.watch(self.model.trainable_variables)
            output = self.model(states, training=True)

            current_q_values = tf.gather(output, indices=tf.expand_dims(actions, -1), axis=1)

            target = np.expand_dims(target, 1)
            # mse
            # loss = ((current_q_values-target)**2)
            loss = tf.keras.losses.MSE(target, current_q_values)
            loss_mean = tf.math.reduce_mean(loss)

        self.loss_history.append(loss_mean)
        grads = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss_mean


    def get_current(self, states, actions):
        with tf.GradientTape() as tape:
            # TODO:
            tape.watch(self.model.trainable_variables)
            output = self.model(states, training=True)

        self.tape = tape

        return tf.gather(output, indices=tf.expand_dims(actions, -1), axis=1)


    @staticmethod
    def non_final_state_idxs(next_states):
        # Detect black screens
        # States are images, flatten them, first dim is batch so keep it
        shape = [next_states.shape[0], -1]
        states_as_arrays = tf.reshape(next_states, shape)
        # from the array of each state, extrat the state max pixel value
        states_max_pixel_values = tf.math.reduce_max(states_as_arrays, axis=1)
        print(states_max_pixel_values.shape)
        # if the max pixel value is 0, its a black screen
        final_state_idxs = tf.math.equal(states_max_pixel_values, 0.0)

        non_final_state_idxs = (final_state_idxs == False)
        return non_final_state_idxs


    def get_next(self, target_net, next_states):
        # only calculate state qvals for non_final_states
        non_final_state_mask = PolicyNetTF.non_final_state_idxs(next_states)
        non_final_states = next_states[non_final_state_mask]
        # Dont GradientTape this :)
        qvals = target_net(non_final_states)

        batch_size = next_states.shape[0]
        maxi = tf.math.reduce_max(qvals, axis=1)
        # get list of indexes which should be updated
        # https://stackoverflow.com/questions/53632837/tensorflow-assign-tensor-to-tensor-with-array-indexing
        non_final_state_idxs = tf.where(non_final_state_mask)
        next_q_values = tf.scatter_nd(non_final_state_idxs, maxi, [batch_size])

        return next_q_values

#%%


class Agent():
    def __init__(self, strategy: Strategy, num_actions: int):
        self.strategy: Strategy = strategy
        self.num_actions: int = num_actions
        self.current_step: int = 0


    def select_action(self, state, policy: PolicyNet):
        """        
        Arguments:\n
            state {[type]} -- [description]
            policy {Policy} -- policy.qvalues(state) should return qvalues for actions (List[float])
        
        Returns:\n
            action -- action that either exploits based on a policy or explores randomly
        """
        rate_exploration = self.strategy.get_exploration_rate(
            self.current_step)
        self.current_step += 1

        if rate_exploration > random.random():
            # explore
            # Choose random action
            return random.randrange(self.num_actions)
        else:
            # exploit
            qvals = policy.qvalues(state)
            return qvals.argmax(axis=1).item()




#%%
em = CartPoleEnvManager()

em.reset()

#%%

if DEMO:
    screen = em.renderArray()

    plt.figure()
    plt.imshow(screen)
    plt.title('Non-processed screen example')
    plt.show()

    print("screen.shape", screen.shape)

#%%
# dummy strategy
strategy = EpsilonGreedyStrategy(0.0, 0.0, 0.0)

agent = Agent(strategy, em.num_actions_available())

#%%

if DEMO:
    policy = PolicyNetTF(None)
    screenPr = policy.preprocess_screen(screen)
    print("screenPr.shape", screenPr.shape)
    plt.figure()
    plt.imshow(screenPr, interpolation='none')
    plt.title('Processed screen example')
    plt.show()


#%%

if DEMO:
    screen = em.get_state()
            
    plt.figure()
    plt.imshow(screen[0], interpolation='none')
    plt.title('Starting state example')
    plt.show()


#%%

if DEMO:
    for i in range(5):
        em.take_action(1)

    screens = em.get_state()
    inputScreen = policy.get_state(screens)

    plt.figure()
    plt.imshow(inputScreen, interpolation='none')
    plt.title('Non starting state example')
    plt.show()


#%%

if DEMO:
    em.done = True
    screen = em.get_state()

    plt.figure()
    plt.imshow(screen[0], interpolation='none')
    plt.title('Ending state example')
    plt.show()

    em.close()


#%%

def moving_avg(period, values):
    if len(values) < period:
        period = len(values)
    valuesCumSum = np.cumsum(values, dtype=float)
    valuesCumSum[period:] = valuesCumSum[period:] - valuesCumSum[:-period]
    return valuesCumSum / period


def plot_moving_avg(values, moving_avg_period):
    plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    mv_avg = moving_avg(moving_avg_period, values)
    plt.plot(values)
    plt.plot(mv_avg)
    plt.pause(0.001)
    print("Episode", len(values), "\t",
          moving_avg_period, "episode moving avg:", mv_avg[-1])
    if is_ipython:
        display.clear_output(wait=True)


#%%

if DEMO:
    array = np.random.rand(300)
    plot_moving_avg(array, 100)

#%% [markdown]
# # Start training

#%%
hypers: Dict[str, Any] = {
    'batch_size': 256,
    # discount factor used in the Bellman equation
    'gamma': 0.999,
    # exploration rate
    'eps_start': 1,
    'eps_end': 0.01,
    'eps_decay': 0.001,
    # number of episodes to update target net
    'target_update': 10,
    # capacity of the replay memory
    'memory_size': 100000,
    'lr': 0.001,
    'optimizer': optimizers.Adam,
    # epoch quantity
    'num_episodes': 1000,
}

#%%

em = CartPoleEnvManager()

strategy = EpsilonGreedyStrategy(hypers['eps_start'], hypers['eps_end'], hypers['eps_decay'])

agent = Agent(strategy, em.num_actions_available())

memory = ReplayMemory(hypers['memory_size'], hypers['batch_size'])

#%%
# Networks
PolicyNetClass = PolicyNetTF
policy = PolicyNetClass(None)

input_shape = policy.get_input_shape(em.renderArray())

# policy net will get backprop
policy_net = DQN(input_shape)
# target is the frozen model, clone of policy net
target_net = tf.keras.models.clone_model(policy_net)

policy_net.summary()

policy.set_model(policy_net)

#%%

optimizer = hypers['optimizer'](lr=hypers['lr'])

#%%

episode_durations: List[int] = []

progressBar = tf.keras.utils.Progbar(None, stateful_metrics=['Episode', 'timestep'])
stepsCounter = 0
# Initialize replay memory capacity.
# Initialize the policy network with random weights.
# Clone the policy network, and call it the target network.
# For each episode:
for episode_num in range(hypers['num_episodes']):
    progressBar.update(stepsCounter, [('Episode', episode_num)])
    # Initialize the starting state.
    em.reset()
    # get initial state
    # preprocess it as the policy uses it
    state = policy.get_state(em.get_state())
    # For each time step; until end of simulation:
    for timestep in count():
        # Select an action.
        # Via exploration or exploitation
        action = agent.select_action(state, policy)
        # Execute selected action in an emulator.
        reward = em.take_action(action)
        # Observe reward and next state.
        next_state = policy.get_state(em.get_state())
        # Store experience in replay memory.
        memory.push(Experience(state, action, next_state, reward))
        state = next_state
        # Sample random batch from replay memory.
        experiences = memory.sample()
        if experiences:
            # policy.learn(experiences)
            # If you have enough experience
            # get batches of states, actions, rewards, next_states from the batch of experiences
            tupleOfBatches = PolicyNetClass.extract_tensors(experiences)
            states, actions, rewards, next_states = tupleOfBatches

            # # These states and actions were the result of either exploration or exploitation
            # current_q_values = policy.get_current(states, actions)
            # These are always exploitation, the policy's best known actions
            next_q_values = policy.get_next(target_net, next_states)
            
            target_q_values = (next_q_values * hypers['gamma']) + rewards
            # loss = policy.step(current_q_values, target_q_values)
            loss = policy.update(states, actions, target_q_values)
            progressBar.update(stepsCounter, [('loss', loss)])
            stepsCounter += 1
        # If episode reached the end
        if em.done:
            # Save how many steps it took
            episode_durations.append(timestep)
            # plot the durations
            plot_moving_avg(episode_durations, 100)
            progressBar.update(stepsCounter, [('timestep', timestep)])
            break

    if episode_num % hypers['target_update'] == 0:
        target_net = tf.keras.models.clone_model(policy_net)

    em.close()
            # Preprocess states from batch.
            # Pass batch of preprocessed states to policy network.
            # Calculate loss between output Q-values and target Q-values.
            # Requires a pass to the target network for the next state
            # Gradient descent updates weights in the policy network to minimize loss.
            # After 
            # x
            #  time steps, weights in the target network are updated to the weights in the policy network.
