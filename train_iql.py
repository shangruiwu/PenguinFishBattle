
import random
from collections import deque
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from penguin_env import PenguinFishEnv


# =========================================================
# 1. Hyperparameters
# =========================================================

EPISODES = 1500

GAMMA = 0.99
LR = 3e-4
BATCH_SIZE = 64

BUFFER_SIZE = 20_000
MIN_REPLAY_SIZE = 500

TRAIN_FREQ = 4
TARGET_UPDATE = 500

EPS_START = 1.0
EPS_MIN = 0.05
EPS_DECAY = 0.997

SEED = 42

# Random map training
RANDOM_INIT = True

# Save directory
MODEL_DIR = Path("models_random")
PLOT_FILE = Path("iql_training_random.png")

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =========================================================
# 2. DQN Network
# =========================================================

class QNetwork(nn.Module):

    def __init__(self, obs_dim, action_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.network(x)


# =========================================================
# 3. Observation Preprocessing
# =========================================================

def prepare_obs(obs):

    # Grid coordinates: 0 ~ 9
    # Missing objects: -1
    #
    # Normalize positions to approximately 0 ~ 1.
    # Missing objects remain negative.

    return np.asarray(
        obs,
        dtype=np.float32
    ) / 9.0


# =========================================================
# 4. Epsilon-Greedy Action Selection
# =========================================================

def select_action(
    network,
    observation,
    epsilon,
    action_dim=5
):

    # Exploration
    if random.random() < epsilon:
        return random.randrange(action_dim)

    # Exploitation
    state = torch.as_tensor(
        prepare_obs(observation),
        dtype=torch.float32,
        device=DEVICE
    ).unsqueeze(0)

    with torch.no_grad():
        q_values = network(state)

    return int(
        q_values.argmax(dim=1).item()
    )


# =========================================================
# 5. DQN Optimization
# =========================================================

def optimize(
    agent,
    buffers,
    policies,
    targets,
    optimizers
):

    memory = buffers[agent]

    # Wait until enough experiences are collected
    if len(memory) < max(
        MIN_REPLAY_SIZE,
        BATCH_SIZE
    ):
        return None

    # Random sampling from replay buffer
    batch = random.sample(
        memory,
        BATCH_SIZE
    )

    (
        states,
        actions,
        rewards,
        next_states,
        dones
    ) = zip(*batch)

    # Convert to tensors
    states = torch.as_tensor(
        np.stack(states),
        dtype=torch.float32,
        device=DEVICE
    )

    actions = torch.as_tensor(
        actions,
        dtype=torch.long,
        device=DEVICE
    ).unsqueeze(1)

    rewards = torch.as_tensor(
        rewards,
        dtype=torch.float32,
        device=DEVICE
    )

    next_states = torch.as_tensor(
        np.stack(next_states),
        dtype=torch.float32,
        device=DEVICE
    )

    dones = torch.as_tensor(
        dones,
        dtype=torch.float32,
        device=DEVICE
    )

    # ---------------------------------------------
    # Current Q(s, a)
    # ---------------------------------------------

    current_q = policies[agent](
        states
    ).gather(
        1,
        actions
    ).squeeze(1)

    # ---------------------------------------------
    # Target Q
    # ---------------------------------------------

    with torch.no_grad():

        next_q = targets[agent](
            next_states
        ).max(dim=1).values

        target_q = (
            rewards
            + GAMMA * (1.0 - dones) * next_q
        )

    # ---------------------------------------------
    # Huber Loss
    # ---------------------------------------------

    loss = F.smooth_l1_loss(
        current_q,
        target_q
    )

    # ---------------------------------------------
    # Backpropagation
    # ---------------------------------------------

    optimizers[agent].zero_grad()

    loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(
        policies[agent].parameters(),
        max_norm=10.0
    )

    optimizers[agent].step()

    return float(loss.item())


# =========================================================
# 6. Moving Average
# =========================================================

def moving_average(values, window=30):

    values = np.asarray(
        values,
        dtype=np.float32
    )

    return np.array([
        np.mean(
            values[max(0, i - window + 1):i + 1]
        )
        for i in range(len(values))
    ])


# =========================================================
# 7. Training
# =========================================================

def train():

    # ---------------------------------------------
    # Random seeds
    # ---------------------------------------------

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # ---------------------------------------------
    # Create environment
    # ---------------------------------------------

    env = PenguinFishEnv(
        render_mode=None,
        random_init=RANDOM_INIT
    )

    agent_names = env.possible_agents.copy()

    obs_dim = env.observation_space(
        agent_names[0]
    ).shape[0]

    action_dim = env.action_space(
        agent_names[0]
    ).n

    # ---------------------------------------------
    # Independent Policy Networks
    # ---------------------------------------------

    policies = {
        agent: QNetwork(
            obs_dim,
            action_dim
        ).to(DEVICE)
        for agent in agent_names
    }

    # ---------------------------------------------
    # Independent Target Networks
    # ---------------------------------------------

    targets = {
        agent: QNetwork(
            obs_dim,
            action_dim
        ).to(DEVICE)
        for agent in agent_names
    }

    # ---------------------------------------------
    # Independent Optimizers
    # ---------------------------------------------

    optimizers = {
        agent: torch.optim.Adam(
            policies[agent].parameters(),
            lr=LR
        )
        for agent in agent_names
    }

    # ---------------------------------------------
    # Independent Replay Buffers
    # ---------------------------------------------

    buffers = {
        agent: deque(
            maxlen=BUFFER_SIZE
        )
        for agent in agent_names
    }

    # Initialize target networks
    for agent in agent_names:

        targets[agent].load_state_dict(
            policies[agent].state_dict()
        )

        targets[agent].eval()

    # ---------------------------------------------
    # Training statistics
    # ---------------------------------------------

    reward_history = []
    fish_history = []
    caught_history = []

    agent_reward_history = {
        agent: []
        for agent in agent_names
    }

    agent_fish_history = {
        agent: []
        for agent in agent_names
    }

    agent_caught_history = {
        agent: []
        for agent in agent_names
    }

    epsilon = EPS_START
    global_step = 0

    print("=" * 65)
    print("Penguin Fish Battle - Independent DQN")
    print("=" * 65)

    print("Device:", DEVICE)
    print("Episodes:", EPISODES)
    print("Random initialization:", RANDOM_INIT)
    print("Observation dimension:", obs_dim)
    print("Action dimension:", action_dim)
    print("Number of agents:", len(agent_names))
    print("=" * 65)

    # =====================================================
    # Episode Loop
    # =====================================================

    for episode in range(1, EPISODES + 1):

        # New random map for every episode
        observations, infos = env.reset(
            seed=SEED + episode
        )

        episode_rewards = {
            agent: 0.0
            for agent in agent_names
        }

        episode_losses = []

        # =================================================
        # Environment Step Loop
        # =================================================

        while env.agents:

            acting_agents = env.agents.copy()

            # ---------------------------------------------
            # 1. Select actions
            # ---------------------------------------------

            actions = {
                agent: select_action(
                    policies[agent],
                    observations[agent],
                    epsilon,
                    action_dim
                )
                for agent in acting_agents
            }

            # ---------------------------------------------
            # 2. Environment step
            # ---------------------------------------------

            (
                next_observations,
                rewards,
                terminations,
                truncations,
                infos
            ) = env.step(actions)

            # ---------------------------------------------
            # 3. Store transitions
            # ---------------------------------------------

            for agent in acting_agents:

                done = (
                    terminations[agent]
                    or truncations[agent]
                )

                transition = (
                    prepare_obs(
                        observations[agent]
                    ),

                    actions[agent],

                    float(rewards[agent]),

                    prepare_obs(
                        next_observations[agent]
                    ),

                    float(done)
                )

                buffers[agent].append(
                    transition
                )

                episode_rewards[agent] += (
                    rewards[agent]
                )

            global_step += 1

            # ---------------------------------------------
            # 4. Train DQNs independently
            # ---------------------------------------------

            if global_step % TRAIN_FREQ == 0:

                for agent in agent_names:

                    loss = optimize(
                        agent,
                        buffers,
                        policies,
                        targets,
                        optimizers
                    )

                    if loss is not None:
                        episode_losses.append(loss)

            # ---------------------------------------------
            # 5. Update target networks
            # ---------------------------------------------

            if global_step % TARGET_UPDATE == 0:

                for agent in agent_names:

                    targets[agent].load_state_dict(
                        policies[agent].state_dict()
                    )

            # ---------------------------------------------
            # 6. Update observations
            # ---------------------------------------------

            observations = next_observations

        # =================================================
        # End of Episode
        # =================================================

        total_reward = sum(
            episode_rewards.values()
        )

        total_fish = sum(
            env.fish_count.values()
        )

        total_caught = sum(
            not env.alive[agent]
            for agent in agent_names
        )

        # ---------------------------------------------
        # Overall statistics
        # ---------------------------------------------

        reward_history.append(total_reward)
        fish_history.append(total_fish)

        caught_history.append(
            total_caught / len(agent_names)
        )

        # ---------------------------------------------
        # Individual agent statistics
        # ---------------------------------------------

        for agent in agent_names:

            agent_reward_history[agent].append(
                episode_rewards[agent]
            )

            agent_fish_history[agent].append(
                env.fish_count[agent]
            )

            agent_caught_history[agent].append(
                int(not env.alive[agent])
            )

        # ---------------------------------------------
        # Epsilon decay
        # ---------------------------------------------

        epsilon = max(
            EPS_MIN,
            epsilon * EPS_DECAY
        )

        # ---------------------------------------------
        # Training log
        # ---------------------------------------------

        if episode % 50 == 0:

            avg_reward = np.mean(
                reward_history[-50:]
            )

            avg_fish = np.mean(
                fish_history[-50:]
            )

            avg_caught = np.mean(
                caught_history[-50:]
            )

            avg_loss = (
                np.mean(episode_losses)
                if episode_losses
                else 0.0
            )

            print(
                f"Episode {episode:4d} | "
                f"Epsilon {epsilon:.3f} | "
                f"Avg Fish {avg_fish:.2f}/3 | "
                f"Avg Reward {avg_reward:.2f} | "
                f"Caught {avg_caught:.1%} | "
                f"Loss {avg_loss:.4f}"
            )

        # ---------------------------------------------
        # Individual agent log
        # ---------------------------------------------

        if episode % 100 == 0:

            print("  Individual agent performance:")

            for agent in agent_names:

                avg_agent_fish = np.mean(
                    agent_fish_history[agent][-100:]
                )

                avg_agent_reward = np.mean(
                    agent_reward_history[agent][-100:]
                )

                avg_agent_caught = np.mean(
                    agent_caught_history[agent][-100:]
                )

                print(
                    f"    {agent}: "
                    f"Fish={avg_agent_fish:.2f}, "
                    f"Reward={avg_agent_reward:.2f}, "
                    f"Caught={avg_agent_caught:.1%}"
                )

    # =====================================================
    # 8. Save Models
    # =====================================================

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for agent in agent_names:

        torch.save(
            policies[agent].state_dict(),
            MODEL_DIR / f"{agent}.pt"
        )

    print("\nModels saved to:", MODEL_DIR)

    # =====================================================
    # 9. Plot Fish Collection
    # =====================================================

    plt.figure(figsize=(10, 5))

    plt.plot(
        fish_history,
        alpha=0.25,
        label="Fish collected"
    )

    plt.plot(
        moving_average(
            fish_history,
            window=30
        ),
        linewidth=2,
        label="30-episode moving average"
    )

    plt.xlabel("Episode")
    plt.ylabel("Total fish collected")
    plt.title(
        "Independent DQN - Random Map Training"
    )

    plt.ylim(-0.1, 3.1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        PLOT_FILE,
        dpi=150
    )

    # =====================================================
    # 10. Plot Individual Agent Performance
    # =====================================================

    plt.figure(figsize=(10, 5))

    for agent in agent_names:

        plt.plot(
            moving_average(
                agent_fish_history[agent],
                window=50
            ),
            label=agent
        )

    plt.xlabel("Episode")
    plt.ylabel("Average fish collected")
    plt.title(
        "Individual Penguin Performance"
    )

    plt.ylim(-0.1, 3.1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "iql_agents_random.png",
        dpi=150
    )

    # =====================================================
    # 11. Plot Reward
    # =====================================================

    plt.figure(figsize=(10, 5))

    plt.plot(
        reward_history,
        alpha=0.2,
        label="Episode reward"
    )

    plt.plot(
        moving_average(
            reward_history,
            window=30
        ),
        linewidth=2,
        label="30-episode moving average"
    )

    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title(
        "Independent DQN - Reward"
    )

    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "iql_reward_random.png",
        dpi=150
    )

    plt.show()

    # =====================================================
    # 12. Final Summary
    # =====================================================

    print("\n" + "=" * 65)
    print("TRAINING FINISHED")
    print("=" * 65)

    print("Total episodes:", EPISODES)
    print("Total environment steps:", global_step)
    print("Final epsilon:", round(epsilon, 4))

    print(
        "Last 100 avg fish:",
        round(np.mean(fish_history[-100:]), 3)
    )

    print(
        "Last 100 avg reward:",
        round(np.mean(reward_history[-100:]), 3)
    )

    print(
        "Last 100 caught rate:",
        f"{np.mean(caught_history[-100:]):.1%}"
    )

    print("Saved models:", MODEL_DIR)

    env.close()


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":
    train()
