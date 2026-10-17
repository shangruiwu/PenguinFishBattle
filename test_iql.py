
import numpy as np
import torch

from penguin_env import PenguinFishEnv
from train_iql import (
    QNetwork,
    DEVICE,
    prepare_obs,
)


def load_models():

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

        state_dict = torch.load(
            f"models/{agent}.pt",
            map_location=DEVICE,
            weights_only=True,
        )

        model.load_state_dict(state_dict)
        model.eval()

        models[agent] = model

    env.close()
    return models


def greedy_action(model, observation):

    state = torch.as_tensor(
        prepare_obs(observation),
        dtype=torch.float32,
        device=DEVICE,
    ).unsqueeze(0)

    with torch.no_grad():
        action = model(state).argmax(
            dim=1
        ).item()

    return int(action)


def evaluate(models=None, episodes=100):

    env = PenguinFishEnv(render_mode=None)

    total_fish = 0
    total_reward = 0.0
    total_caught = 0

    for episode in range(episodes):

        observations, _ = env.reset(
            seed=10000 + episode
        )

        rng = np.random.default_rng(
            20000 + episode
        )

        while env.agents:

            if models is None:
                actions = {
                    agent: int(rng.integers(5))
                    for agent in env.agents
                }
            else:
                actions = {
                    agent: greedy_action(
                        models[agent],
                        observations[agent],
                    )
                    for agent in env.agents
                }

            (
                observations,
                rewards,
                terminations,
                truncations,
                infos,
            ) = env.step(actions)

            total_reward += sum(rewards.values())

        total_fish += sum(env.fish_count.values())

        total_caught += sum(
            not env.alive[agent]
            for agent in env.possible_agents
        )

    env.close()

    return {
        "avg_fish": total_fish / episodes,
        "avg_reward_per_agent": (
            total_reward / (episodes * 3)
        ),
        "caught_rate": (
            total_caught / (episodes * 3)
        ),
    }


def visualize(models):

    env = PenguinFishEnv(render_mode="human")

    observations, _ = env.reset(seed=1234)

    while env.agents:

        actions = {
            agent: greedy_action(
                models[agent],
                observations[agent],
            )
            for agent in env.agents
        }

        observations, rewards, terms, truncs, infos = (
            env.step(actions)
        )

    print("\nVisualization finished.")
    print("Fish:", env.fish_count)
    print("Returns:", env.episode_returns)

    env.close()


if __name__ == "__main__":

    models = load_models()

    random_results = evaluate(models=None)
    trained_results = evaluate(models=models)

    print("\n===== Evaluation =====")

    for name, result in [
        ("Random", random_results),
        ("IQL", trained_results),
    ]:

        print(f"\n{name}")
        print(
            "Average fish:",
            round(result["avg_fish"], 3),
        )
        print(
            "Average reward per agent:",
            round(result["avg_reward_per_agent"], 3),
        )
        print(
            "Caught rate:",
            round(result["caught_rate"], 3),
        )

    visualize(models)
