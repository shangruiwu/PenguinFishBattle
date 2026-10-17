
from penguin_env import PenguinFishEnv


def main():

    env = PenguinFishEnv(render_mode="human")

    observations, infos = env.reset(seed=42)

    total_rewards = {
        agent: 0.0
        for agent in env.possible_agents
    }

    while env.agents:

        # Random actions for all living penguins
        actions = {
            agent: env.action_space(agent).sample()
            for agent in env.agents
        }

        (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        ) = env.step(actions)

        for agent, reward in rewards.items():
            total_rewards[agent] += reward

        print(f"\nStep {env.current_step}")
        print("Actions:", actions)
        print("Rewards:", rewards)
        print("Terminations:", terminations)
        print("Truncations:", truncations)

    print("\n===== Game Over =====")

    for agent in env.possible_agents:
        print(
            agent,
            "| Fish:", env.fish_count[agent],
            "| Total reward:",
            round(total_rewards[agent], 2),
        )

    env.close()


if __name__ == "__main__":
    main()
