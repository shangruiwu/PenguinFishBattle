
import numpy as np
import torch

from penguin_env import PenguinFishEnv
from train_iql import QNetwork, DEVICE, prepare_obs


EPISODES = 300


# ============================================
# Load trained models
# ============================================
def load_models(model_dir):

    env = PenguinFishEnv()

    obs_dim = env.observation_space(
        "penguin_0"
    ).shape[0]

    action_dim = env.action_space(
        "penguin_0"
    ).n

    models = {}

    for agent in env.possible_agents:

        model = QNetwork(
            obs_dim, action_dim
        ).to(DEVICE)

        model.load_state_dict(
            torch.load(
                f"{model_dir}/{agent}.pt",
                map_location=DEVICE,
                weights_only=True
            )
        )

        model.eval()
        models[agent] = model

    env.close()
    return models


# ============================================
# Greedy policy
# ============================================
def greedy_action(model, observation):

    obs = torch.as_tensor(
        prepare_obs(observation),
        dtype=torch.float32,
        device=DEVICE
    ).unsqueeze(0)

    with torch.no_grad():
        q = model(obs)

    return int(q.argmax(dim=1).item())


# ============================================
# Evaluation
# ============================================
def evaluate(models, random_init, episodes=300):

    env = PenguinFishEnv(
        render_mode=None,
        random_init=random_init
    )

    agent_names = env.possible_agents.copy()

    stats = {
        agent: {
            "fish": 0,
            "reward": 0.0,
            "caught": 0,
            "steps_alive": 0,
        }
        for agent in agent_names
    }

    episode_lengths = []

    for episode in range(episodes):

        # Same seed for each evaluation setup
        observations, infos = env.reset(
            seed=10000 + episode
        )

        action_rng = np.random.default_rng(
            20000 + episode
        )

        while env.agents:

            acting_agents = env.agents.copy()

            # Count active steps
            for agent in acting_agents:
                stats[agent]["steps_alive"] += 1

            if models is None:

                actions = {
                    agent: int(action_rng.integers(5))
                    for agent in acting_agents
                }

            else:

                actions = {
                    agent: greedy_action(
                        models[agent],
                        observations[agent]
                    )
                    for agent in acting_agents
                }

            (
                observations,
                rewards,
                terminations,
                truncations,
                infos
            ) = env.step(actions)

            for agent in acting_agents:
                stats[agent]["reward"] += rewards[agent]

        # Collect final episode statistics
        episode_lengths.append(env.current_step)

        for agent in agent_names:

            stats[agent]["fish"] += env.fish_count[agent]

            if not env.alive[agent]:
                stats[agent]["caught"] += 1

    env.close()

    # ========================================
    # Print report
    # ========================================
    print("\n" + "=" * 65)

    print(
        "Map:",
        "Random" if random_init else "Fixed"
    )

    print("Episodes:", episodes)

    print("-" * 65)

    total_fish = 0

    for agent in agent_names:

        data = stats[agent]

        avg_fish = data["fish"] / episodes
        avg_reward = data["reward"] / episodes

        caught_rate = (
            data["caught"] / episodes * 100
        )

        avg_steps_alive = (
            data["steps_alive"] / episodes
        )

        total_fish += avg_fish

        print(f"\n{agent}")
        print(f"  Avg fish:       {avg_fish:.3f}")
        print(f"  Avg reward:     {avg_reward:.3f}")
        print(f"  Caught rate:    {caught_rate:.1f}%")
        print(f"  Avg alive steps:{avg_steps_alive:.1f}")

    print("\nOverall")
    print(f"  Avg total fish: {total_fish:.3f}")
    print(
        "  Avg episode length:",
        f"{np.mean(episode_lengths):.1f}"
    )

    return stats


# ============================================
# Main
# ============================================
if __name__ == "__main__":

    fixed_models = load_models("models")
    random_models = load_models("models_random")

    print("\n========== RANDOM BASELINE ==========")

    evaluate(
        models=None,
        random_init=True,
        episodes=EPISODES
    )

    print("\n========== FIXED MODEL / FIXED MAP ==========")

    evaluate(
        models=fixed_models,
        random_init=False,
        episodes=EPISODES
    )

    print("\n========== FIXED MODEL / RANDOM MAP ==========")

    evaluate(
        models=fixed_models,
        random_init=True,
        episodes=EPISODES
    )

    print("\n========== RANDOM MODEL / RANDOM MAP ==========")

    evaluate(
        models=random_models,
        random_init=True,
        episodes=EPISODES
    )
