#import and setup
 
import pygame
import random
import time
from pygame.locals import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
import os

#(models) directory to save checkpoints
if not os.path.exists('models'):
    os.makedirs('models')

SCREEN_WIDHT = 1000
SCREEN_HEIGHT = 700
SPEED = 20
GRAVITY = 2.5
GAME_SPEED = 15

GROUND_WIDHT = 2 * SCREEN_WIDHT
GROUND_HEIGHT = 100

PIPE_WIDHT = 80
PIPE_HEIGHT = 500

PIPE_GAP = 150

pygame.mixer.init()

# audio files for sounds
wing_sound = pygame.mixer.Sound('assets/audio/wing.wav')
hit_sound = pygame.mixer.Sound('assets/audio/hit.wav')


class Bird(pygame.sprite.Sprite):
#loads 3 bird image and sets initial pos, speed, mask for collision   
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.images = [
            pygame.image.load('assets/sprites/bluebird-upflap.png').convert_alpha(),
            pygame.image.load('assets/sprites/bluebird-midflap.png').convert_alpha(),
            pygame.image.load('assets/sprites/bluebird-downflap.png').convert_alpha()
        ]
        self.speed = SPEED
        self.current_image = 0
        self.image = self.images[self.current_image]
        self.mask = pygame.mask.from_surface(self.image)
        self.rect = self.image.get_rect()
        self.rect[0] = SCREEN_WIDHT / 6
        self.rect[1] = SCREEN_HEIGHT / 2

#animation and gravity
    def update(self):
        self.current_image = (self.current_image + 1) % 3
        self.image = self.images[self.current_image]
        self.speed += GRAVITY
        self.rect[1] += self.speed
#gives jump and sound plays
    def bump(self):
        self.speed = -SPEED
        if wing_sound:
            wing_sound.play()

    def begin(self):
        self.current_image = (self.current_image + 1) % 3
        self.image = self.images[self.current_image]


class Pipe(pygame.sprite.Sprite):
#loads pipe image and determines its orientation (inverted or not), sets position and size
    def __init__(self, inverted, x_pos, ysize):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load('assets/sprites/pipe-green.png').convert_alpha()
        self.image = pygame.transform.scale(self.image, (PIPE_WIDHT, PIPE_HEIGHT))
        self.rect = self.image.get_rect()
        self.rect[0] = x_pos
        self.inverted = inverted # Store the inverted flag as an instance attribute
        if inverted:
            self.image = pygame.transform.flip(self.image, False, True)
            self.rect[1] = -(self.rect[3] - ysize)
        else:
            self.rect[1] = SCREEN_HEIGHT - ysize
        self.mask = pygame.mask.from_surface(self.image)

#move each pipe to the left
    def update(self):
        self.rect[0] -= GAME_SPEED

class Ground(pygame.sprite.Sprite):
#creates ground segment and shifts it at same speed as pipe
    def __init__(self, x_pos):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load('assets/sprites/base.png').convert_alpha()
        self.image = pygame.transform.scale(self.image, (GROUND_WIDHT, GROUND_HEIGHT))
        self.mask = pygame.mask.from_surface(self.image)
        self.rect = self.image.get_rect()
        self.rect[0] = x_pos
        self.rect[1] = SCREEN_HEIGHT - GROUND_HEIGHT

    def update(self):
        self.rect[0] -= GAME_SPEED

def is_off_screen(sprite):
    return sprite.rect[0] < -(sprite.rect[2])

#generate random pipes
def get_random_pipes(x_pos):
    size = random.randint(100, 300)
    pipe = Pipe(False, x_pos, size)
    pipe_inverted = Pipe(True, x_pos, SCREEN_HEIGHT - size - PIPE_GAP)
    return pipe, pipe_inverted
# the rl environment
class FlappyBirdEnv:
    def __init__(self):
        pygame.init()
    # Initialize screen once, but only update it when rendering is enabled
        self.screen = pygame.display.set_mode((SCREEN_WIDHT, SCREEN_HEIGHT))
        pygame.display.set_caption('Flappy Bird RL')
        self.BACKGROUND = pygame.image.load('assets/sprites/background-day.png')
        self.BACKGROUND = pygame.transform.scale(self.BACKGROUND, (SCREEN_WIDHT, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.score = 0
        self.reset()

    def reset(self):
        self.bird_group = pygame.sprite.Group()
        self.bird = Bird()
        self.bird_group.add(self.bird)

        self.ground_group = pygame.sprite.Group()
        for i in range(2):
            ground = Ground(GROUND_WIDHT * i)
            self.ground_group.add(ground)

        self.pipe_group = pygame.sprite.Group()
        for i in range(2):
            pipes = get_random_pipes(SCREEN_WIDHT * i + 800)
            self.pipe_group.add(pipes[0])
            self.pipe_group.add(pipes[1])
        self.score = 0
        self.done = False
        return self._get_state()

# computes state (birds position, speed, next pipe positions)
    def _get_state(self):
        bird_y = self.bird.rect.centery
        bird_vel = self.bird.speed

        next_pipe_bottom = None
        next_pipe_top = None
        
        sorted_pipes = sorted(self.pipe_group.sprites(), key=lambda p: p.rect[0])

        for p in sorted_pipes:
            if p.rect[0] + p.rect.width > self.bird.rect[0]:
                if not p.inverted:
                    next_pipe_bottom = p
                    for p_top in sorted_pipes:
                        if p_top.rect[0] == p.rect[0] and p_top.inverted:
                            next_pipe_top = p_top
                            break
                else:
                    next_pipe_top = p
                    for p_bottom in sorted_pipes:
                        if p_bottom.rect[0] == p.rect[0] and not p_bottom.inverted:
                            next_pipe_bottom = p_bottom
                            break
                
                if next_pipe_bottom and next_pipe_top:
                    break

        if next_pipe_bottom is None or next_pipe_top is None:
        # If no pipes are visible, assume a far-off pipe
            next_pipe_x = SCREEN_WIDHT * 2
            next_pipe_top_opening_y = 0
            next_pipe_bottom_opening_y = SCREEN_HEIGHT
        else:
            next_pipe_x = next_pipe_bottom.rect[0]
            next_pipe_top_opening_y = next_pipe_top.rect.bottom
            next_pipe_bottom_opening_y = next_pipe_bottom.rect.top

        # State normalization
        state = np.array([
            bird_y / SCREEN_HEIGHT,
            bird_vel / (SPEED * 2),  # Normalize bird speed
            (next_pipe_x - self.bird.rect.right) / SCREEN_WIDHT, # Distance to next pipe
            (next_pipe_top_opening_y - bird_y) / SCREEN_HEIGHT, # Bird's y-pos relative to top pipe opening
            (next_pipe_bottom_opening_y - bird_y) / SCREEN_HEIGHT # Bird's y-pos relative to bottom pipe opening
        ])
        return state

    def step(self, action, render_game=True):
        reward = 0.1 # Small positive reward for staying alive
        self.score += 0.1

        for event in pygame.event.get():
            if event.type == QUIT:
                self.done = True

        if action == 1: # Flap action
            self.bird.bump()

        if render_game:
            self.screen.blit(self.BACKGROUND, (0, 0))


        if is_off_screen(self.ground_group.sprites()[0]):
            self.ground_group.remove(self.ground_group.sprites()[0])
            new_ground = Ground(GROUND_WIDHT - 20)
            self.ground_group.add(new_ground)

        # Pipe generation and scrolling
        if is_off_screen(self.pipe_group.sprites()[0]):
            self.pipe_group.remove(self.pipe_group.sprites()[0])
            self.pipe_group.remove(self.pipe_group.sprites()[0])
            pipes = get_random_pipes(SCREEN_WIDHT * 2)
            self.pipe_group.add(pipes[0])
            self.pipe_group.add(pipes[1])
            reward += 1.0 # Reward for passing a pipe

        self.bird_group.update()
        self.ground_group.update()
        self.pipe_group.update()

        if render_game:
            self.bird_group.draw(self.screen)
            self.pipe_group.draw(self.screen)
            self.ground_group.draw(self.screen)

            pygame.display.update()
            self.clock.tick(GAME_SPEED)

        # Collision detection with ground, pipes, and top of the screen
        if (pygame.sprite.groupcollide(self.bird_group, self.ground_group, False, False, pygame.sprite.collide_mask) or
                pygame.sprite.groupcollide(self.bird_group, self.pipe_group, False, False, pygame.sprite.collide_mask) or
                self.bird.rect[1] < 0): 
            if hit_sound:
                hit_sound.play()
            reward = -1.0 # Penalty for collision or going out of bounds
            self.done = True

        next_state = self._get_state()
        return next_state, reward, self.done, {}

    def render(self):
        pass # Rendering is handled within the step method for Pygame

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        # A simple feed-forward neural network with two hidden layers (64units)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
     # Apply ReLU activation to hidden layers
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    #outputs Q-values for each action 

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # Store a transition (s, a, r, s', done)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # Randomly sample a batch of experiences
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.tensor(np.array(states), dtype=torch.float32),
                torch.tensor(actions, dtype=torch.long),
                torch.tensor(rewards, dtype=torch.float32),
                torch.tensor(np.array(next_states), dtype=torch.float32),
                torch.tensor(dones, dtype=torch.bool))

    def __len__(self):
        return len(self.buffer)

#creates the DQN agent
class DQNAgent:
    def __init__(self, state_size, action_size, seed=0, model_path='models/flappy_bird_dqn.pth'):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = torch.manual_seed(seed)
        self.model_path = model_path

        self.qnetwork_local = QNetwork(state_size, action_size)
        self.qnetwork_target = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=0.001)
        self.memory = ReplayBuffer(capacity=100000) 
        self.gamma = 0.99
        self.epsilon = 1.0 # Initial epsilon value
        self.epsilon_decay = 0.999 # Slower decay for more exploration
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.target_update_freq = 100

        self.t_step = 0
        self.load_model() # Attempt to load existing model and its epsilon
        self.update_target_network()

    def step(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % self.target_update_freq
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        self.qnetwork_local.eval() # Set network to evaluation mode
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train() # Set network back to training mode
        if random.random() > self.epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        q_targets = rewards.unsqueeze(1) + (self.gamma * q_targets_next * (1 - dones.unsqueeze(1).float()))
        # Get expected Q values from local model
        q_expected = self.qnetwork_local(states).gather(1, actions.unsqueeze(1))

        # Compute loss
        loss = nn.functional.mse_loss(q_expected, q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decrease epsilon (for exploration-exploitation trade-off)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_network(self):
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

    def save_model(self):
        # Save both model state_dict and epsilon
        torch.save({
            'qnetwork_state_dict': self.qnetwork_local.state_dict(),
            'epsilon': self.epsilon
        }, self.model_path)
        print(f"Model and epsilon saved to {self.model_path}")

    def load_model(self):
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path)
            # Check if the loaded checkpoint is a dictionary and contains the expected key
            if isinstance(checkpoint, dict) and 'qnetwork_state_dict' in checkpoint:
                self.qnetwork_local.load_state_dict(checkpoint['qnetwork_state_dict'])
                self.qnetwork_target.load_state_dict(checkpoint['qnetwork_state_dict'])
                self.epsilon = checkpoint['epsilon'] # Load saved epsilon
                print(f"Model and epsilon loaded from {self.model_path}. Starting epsilon: {self.epsilon:.4f}")
            else:
                # Handle older saved models where only state_dict was saved directly
                self.qnetwork_local.load_state_dict(checkpoint)
                self.qnetwork_target.load_state_dict(checkpoint)
                # Reset epsilon to a sensible value for continued training with an old model
                self.epsilon = 0.5 
                print(f"Old model format loaded from {self.model_path}. Resetting epsilon to: {self.epsilon:.4f}")
        else:
            print("No saved model found. Starting training from scratch with epsilon: 1.0000")


def plot_scores(scores):
    plt.figure(figsize=(10, 5))
    plt.plot(scores)
    plt.title('Score per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.grid(True)
    plt.show()

def main():
    env = FlappyBirdEnv()
    state_size = 5
    action_size = 2
    agent = DQNAgent(state_size, action_size)

    num_episodes = 100 
    max_steps_per_episode = 1000
    scores = []

    for i_episode in range(1, num_episodes + 1):
        state = env.reset()
        score = 0
        done = False
        for t in range(max_steps_per_episode):
            # Pass render_game=False to prevent rendering during training
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action, render_game=False) 
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores.append(score)
        agent.update_target_network()
        print(f"Episode:{i_episode}, Score:{score:.2f}")
        

    agent.save_model() # Save the model after training
   
    plot_scores(scores) # Plot the scores after training
    
    agent.epsilon = 0.0 # Set epsilon to 0 for pure exploitation during demo
    num_demo_episodes = 3
    for i_episode in range(1, num_demo_episodes + 1):
        state = env.reset()
        score = 0
        done = False
        while not done:
            # Pass render_game=True to enable rendering during demo
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action, render_game=True)
            state = next_state
            score += reward
        print(f" result {i_episode}/{num_demo_episodes}, Score: {score:.2f}")

    pygame.quit()

if __name__ == '__main__':
    main()