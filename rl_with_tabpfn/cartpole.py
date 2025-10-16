import gymnasium as gym
import numpy as np
from tabpfn import TabPFNRegressor
import pickle

# or with q_learning

training = False

env = gym.make("CartPole-v1")

# check that the action space is discrete and the observation space is box
assert isinstance(env.action_space, gym.spaces.Discrete)
assert isinstance(env.observation_space, gym.spaces.Box)

replay_buffer: np.ndarray = np.empty((0, env.observation_space.shape[0] + 1 + 1)) # state, action, discounted reward
epsilon = 0.1

episode = 0

if training:
    while True:
        state, done = env.reset()[0], False

        step = 0
        states, actions, rewards = [], [], []
        while not done:

            if np.random.rand() < epsilon or len(replay_buffer) < 2000:
                action = env.action_space.sample()
            else:
                model = TabPFNRegressor()
                model.fit(replay_buffer[:, :-1], replay_buffer[:, -1])
                # action array look like [[state..., action], ...]
                actions_array = np.array([np.hstack((state, a)) for a in range(env.action_space.n)])
                action_values = model.predict(actions_array)
                action = np.argmax(action_values)
                print(f"Episode {episode}, step {step}, predicted action values: {action_values}, chosen action: {action}")

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state
            step += 1

        print(f"Episode {episode}, total reward: {sum(rewards)}")

        # compute discounted rewards
        discounted_rewards = []
        G = 0
        for r in reversed(rewards):
            G = r + 0.99 * G
            discounted_rewards.append(G)

        discounted_rewards.reverse()

        # store the episode in the replay buffer as a numpy array
        states_array = np.array(states) # shape (n_steps, 2)
        actions_array = np.array(actions).reshape(-1, 1) # shape (n_steps, 1)
        discounted_rewards_array = np.array(discounted_rewards).reshape(-1, 1) # shape (n_steps, 1)
        episode_data = np.hstack((
            states_array,
            actions_array,
            discounted_rewards_array
        ))
        replay_buffer = np.vstack((replay_buffer, episode_data))
        episode += 1

        if replay_buffer.shape[0] > 9_500:
            # keep only the last 9_500 samples
            replay_buffer = replay_buffer[-9_500:]

        # termination mechanism
        if sum(rewards) == 500:
            print("Solved!")
            # save the replay buffer
            with open("cartpole_replay_buffer.pkl", "wb") as f:
                pickle.dump(replay_buffer, f)

            break

else:
    # set the render mode to human
    env = gym.make("CartPole-v1", render_mode="human")
    
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert isinstance(env.observation_space, gym.spaces.Box)

    # load the replay buffer
    with open("cartpole_replay_buffer.pkl", "rb") as f:
        replay_buffer = pickle.load(f)

    model = TabPFNRegressor()
    model.fit(replay_buffer[:, :-1], replay_buffer[:, -1])

    for episode in range(1):
        state, done = env.reset()[0], False

        step = 0
        total_reward = 0
        while not done:
            env.render()

            actions_array = np.array([np.hstack((state, a)) for a in range(env.action_space.n)])
            action_values = model.predict(actions_array)
            action = np.argmax(action_values)
            print(f"Step {step}, predicted action values: {action_values}, chosen action: {action}")

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            state = next_state
            total_reward += float(reward)
            step += 1

        print(f"Episode {episode}, total reward: {total_reward}")



