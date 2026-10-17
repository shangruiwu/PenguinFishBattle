
import numpy as np
import pygame

from gymnasium.spaces import Box, Discrete
from pettingzoo.utils.env import ParallelEnv


class PenguinFishEnv(ParallelEnv):

    metadata = {
        "name": "penguin_fish_v1",
        "render_modes": ["human", "rgb_array"],
        "render_fps": 8,
    }

    # 0 stay, 1 up, 2 down, 3 left, 4 right
    MOVES = {
        0: (0, 0),
        1: (0, -1),
        2: (0, 1),
        3: (-1, 0),
        4: (1, 0),
    }

    def __init__(self, render_mode=None, random_init=False):
        self.random_init = random_init

        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError("Invalid render_mode")

        # ==========================================
        # Environment settings
        # ==========================================
        self.grid_size = 10
        self.max_steps = 200
        self.num_fish = 3

        self.render_mode = render_mode

        self.cell_size = 60
        self.width = self.grid_size * self.cell_size
        self.height = self.width + 85

        # ==========================================
        # Agents
        # ==========================================
        self.possible_agents = [
            "penguin_0",
            "penguin_1",
            "penguin_2",
        ]

        self.agents = []

        # ==========================================
        # Spaces
        # ==========================================
        self.action_spaces = {
            agent: Discrete(5)
            for agent in self.possible_agents
        }

        # Observation:
        # 3 penguins: 6
        # 1 seal:     2
        # 3 fish:     6
        # Total:     14
        #
        # -1 means absent/dead

        self.observation_spaces = {
            agent: Box(
                low=-1,
                high=self.grid_size - 1,
                shape=(14,),
                dtype=np.int32,
            )
            for agent in self.possible_agents
        }

        self.state_space = Box(
            low=-1,
            high=self.grid_size - 1,
            shape=(14,),
            dtype=np.int32,
        )

        # ==========================================
        # World state
        # ==========================================
        self.rng = np.random.default_rng()

        self.positions = {}
        self.alive = {}

        self.seal_pos = None
        self.fish_positions = []

        self.current_step = 0
        self.fish_count = {}
        self.episode_returns = {}

        # ==========================================
        # Rendering
        # ==========================================
        self.window = None
        self.clock = None
        self.font = None
        self.window_closed = False

        self.colors = {
            "penguin_0": (55, 115, 230),
            "penguin_1": (46, 173, 115),
            "penguin_2": (168, 88, 205),
        }

    # ==============================================
    # PettingZoo spaces
    # ==============================================
    def action_space(self, agent):
        return self.action_spaces[agent]

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    # ==============================================
    # Reset
    # ==============================================

    def reset(self, seed=None, options=None):

        # ==========================================
        # 1. Random seed
        # ==========================================
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # ==========================================
        # 2. Reset agents and counters
        # ==========================================
        self.agents = self.possible_agents.copy()
        self.current_step = 0

        self.alive = {
            agent: True
            for agent in self.possible_agents
        }

        self.fish_count = {
            agent: 0
            for agent in self.possible_agents
        }

        self.episode_returns = {
            agent: 0.0
            for agent in self.possible_agents
        }

        # ==========================================
        # 3. Initialize world positions
        # ==========================================
        if self.random_init:

            # 3 penguins + 1 seal + 3 fish
            num_penguins = len(self.possible_agents)
            num_objects = num_penguins + 1 + self.num_fish

            # Randomly select unique grid cells
            indices = self.rng.choice(
                self.grid_size ** 2,
                size=num_objects,
                replace=False
            )

            positions = [
                np.array(
                    [
                        int(idx % self.grid_size),
                        int(idx // self.grid_size)
                    ],
                    dtype=np.int32
                )
                for idx in indices
            ]

            # Penguin positions
            self.positions = {
                agent: positions[i].copy()
                for i, agent in enumerate(self.possible_agents)
            }

            # Seal position
            self.seal_pos = positions[num_penguins].copy()

            # Fish positions
            self.fish_positions = [
                pos.copy()
                for pos in positions[
                    num_penguins + 1:
                ]
            ]

        else:

            # ======================================
            # Original fixed map
            # ======================================
            self.positions = {
                "penguin_0": np.array(
                    [1, 1], dtype=np.int32
                ),
                "penguin_1": np.array(
                    [8, 1], dtype=np.int32
                ),
                "penguin_2": np.array(
                    [5, 8], dtype=np.int32
                ),
            }

            self.seal_pos = np.array(
                [5, 5], dtype=np.int32
            )

            self.fish_positions = [
                np.array([2, 6], dtype=np.int32),
                np.array([7, 6], dtype=np.int32),
                np.array([5, 2], dtype=np.int32),
            ]

        # ==========================================
        # 4. Generate observations
        # ==========================================
        observations = self._get_observations()

        # ==========================================
        # 5. Initialize info dictionaries
        # ==========================================
        infos = {
            agent: {}
            for agent in self.agents
        }

        # ==========================================
        # 6. Render initial state
        # ==========================================
        if self.render_mode == "human":
            self.render()

        return observations, infos

    # ==============================================
    # Observation
    # ==============================================
    def _get_single_observation(self):

        data = []

        # Penguin positions
        for agent in self.possible_agents:
            if self.alive[agent]:
                data.extend(
                    self.positions[agent].tolist()
                )
            else:
                data.extend([-1, -1])

        # Seal position
        data.extend(self.seal_pos.tolist())

        # Fish positions
        for fish in self.fish_positions:
            if fish is None:
                data.extend([-1, -1])
            else:
                data.extend(fish.tolist())

        return np.array(data, dtype=np.int32)

    def _get_observations(self):

        state = self._get_single_observation()

        return {
            agent: state.copy()
            for agent in self.agents
        }

    def state(self):
        return self._get_single_observation()

    # ==============================================
    # Movement helper
    # ==============================================
    def _move(self, position, action):

        movement = np.array(
            self.MOVES[int(action)],
            dtype=np.int32,
        )

        new_position = position + movement

        return np.clip(
            new_position,
            0,
            self.grid_size - 1,
        ).astype(np.int32)

    # ==============================================
    # Seal AI
    # ==============================================
    def _move_seal(self):

        alive_agents = [
            agent for agent in self.possible_agents
            if self.alive[agent]
        ]

        if not alive_agents:
            return

        # 70% chase, 30% random movement
        if self.rng.random() < 0.7:

            distances = [
                np.abs(
                    self.positions[agent] - self.seal_pos
                ).sum()
                for agent in alive_agents
            ]

            min_distance = min(distances)

            closest = [
                agent
                for agent, distance in zip(
                    alive_agents, distances
                )
                if distance == min_distance
            ]

            target_agent = closest[
                int(self.rng.integers(len(closest)))
            ]

            target = self.positions[target_agent]
            difference = target - self.seal_pos

            possible_actions = []

            if difference[0] < 0:
                possible_actions.append(3)
            elif difference[0] > 0:
                possible_actions.append(4)

            if difference[1] < 0:
                possible_actions.append(1)
            elif difference[1] > 0:
                possible_actions.append(2)

            if possible_actions:
                action = possible_actions[
                    int(
                        self.rng.integers(
                            len(possible_actions)
                        )
                    )
                ]
            else:
                action = 0

        else:
            action = int(self.rng.integers(5))

        self.seal_pos = self._move(
            self.seal_pos, action
        )

    # ==============================================
    # Step
    # ==============================================
    def step(self, actions):

        if not self.agents:
            return {}, {}, {}, {}, {}

        acting_agents = self.agents.copy()

        if set(actions) != set(acting_agents):
            raise ValueError(
                "Actions must contain exactly "
                "the current live agents."
            )

        for agent in acting_agents:
            if not self.action_space(agent).contains(
                actions[agent]
            ):
                raise ValueError(
                    f"Invalid action for {agent}"
                )

        self.current_step += 1

        rewards = {
            agent: -0.05
            for agent in acting_agents
        }

        terminations = {
            agent: False
            for agent in acting_agents
        }

        truncations = {
            agent: False
            for agent in acting_agents
        }

        infos = {
            agent: {
                "fish_eaten": 0,
                "caught": False,
            }
            for agent in acting_agents
        }

        # ------------------------------------------
        # 1. All penguins move simultaneously
        # ------------------------------------------
        new_positions = {}

        for agent in acting_agents:
            new_positions[agent] = self._move(
                self.positions[agent],
                actions[agent],
            )

        self.positions.update(new_positions)

        # ------------------------------------------
        # 2. Eat fish
        # ------------------------------------------
        for index, fish in enumerate(
            self.fish_positions
        ):

            if fish is None:
                continue

            contenders = [
                agent
                for agent in acting_agents
                if np.array_equal(
                    self.positions[agent], fish
                )
            ]

            if not contenders:
                continue

            # If multiple penguins reach one fish,
            # randomly choose one winner.
            winner = contenders[
                int(self.rng.integers(len(contenders)))
            ]

            rewards[winner] += 10.0
            self.fish_count[winner] += 1
            infos[winner]["fish_eaten"] += 1

            self.fish_positions[index] = None

        # ------------------------------------------
        # 3. Seal moves
        # ------------------------------------------
        self._move_seal()

        # ------------------------------------------
        # 4. Seal catches penguins
        # ------------------------------------------
        for agent in acting_agents:

            if np.array_equal(
                self.positions[agent],
                self.seal_pos,
            ):
                rewards[agent] -= 10.0

                self.alive[agent] = False
                terminations[agent] = True
                infos[agent]["caught"] = True

        # ------------------------------------------
        # 5. Episode end conditions
        # ------------------------------------------
        all_fish_eaten = all(
            fish is None
            for fish in self.fish_positions
        )

        no_penguins_alive = not any(
            self.alive.values()
        )

        time_limit = (
            self.current_step >= self.max_steps
        )

        game_finished = (
            all_fish_eaten or no_penguins_alive
        )

        for agent in acting_agents:

            if not self.alive[agent]:
                continue

            if game_finished:
                terminations[agent] = True

            elif time_limit:
                truncations[agent] = True

            # Survival bonus at the end
            if game_finished or time_limit:
                rewards[agent] += 2.0

        # ------------------------------------------
        # 6. Final observations and info
        # ------------------------------------------
        # Include observations for agents that
        # acted this step, even if they died.
        final_state = self._get_single_observation()

        observations = {
            agent: final_state.copy()
            for agent in acting_agents
        }

        for agent in acting_agents:

            self.episode_returns[agent] += rewards[agent]

            infos[agent]["total_fish"] = (
                self.fish_count[agent]
            )

        # Remove terminated/truncated agents
        self.agents = [
            agent
            for agent in acting_agents
            if not (
                terminations[agent]
                or truncations[agent]
            )
        ]

        if self.render_mode == "human":
            self.render()

        return (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        )

    # ==============================================
    # Pygame rendering
    # ==============================================
    def render(self):

        if self.render_mode is None:
            return None

        if self.window_closed:
            return None

        if not pygame.get_init():
            pygame.init()

        if self.font is None:
            self.font = pygame.font.SysFont(None, 22)

        if self.render_mode == "human":

            if self.window is None:
                self.window = pygame.display.set_mode(
                    (self.width, self.height)
                )
                pygame.display.set_caption(
                    "Penguin Fish Battle V1"
                )
                self.clock = pygame.time.Clock()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.window_closed = True
                    self.close()
                    return None

        canvas = pygame.Surface(
            (self.width, self.height)
        )
        canvas.fill((232, 245, 250))

        # ------------------------------------------
        # Draw grid
        # ------------------------------------------
        for i in range(self.grid_size + 1):

            line_pos = i * self.cell_size

            pygame.draw.line(
                canvas, (190, 211, 222),
                (line_pos, 0),
                (line_pos, self.width),
                1,
            )

            pygame.draw.line(
                canvas, (190, 211, 222),
                (0, line_pos),
                (self.width, line_pos),
                1,
            )

        def center(pos):
            return (
                int(pos[0] * self.cell_size
                    + self.cell_size / 2),
                int(pos[1] * self.cell_size
                    + self.cell_size / 2),
            )

        # ------------------------------------------
        # Draw fish
        # ------------------------------------------
        for fish in self.fish_positions:

            if fish is None:
                continue

            x, y = center(fish)

            pygame.draw.ellipse(
                canvas,
                (242, 183, 45),
                (x - 14, y - 9, 28, 18),
            )

            pygame.draw.polygon(
                canvas,
                (242, 183, 45),
                [
                    (x - 12, y),
                    (x - 23, y - 11),
                    (x - 23, y + 11),
                ],
            )

            pygame.draw.circle(
                canvas,
                (30, 50, 60),
                (x + 7, y - 3),
                2,
            )

        # ------------------------------------------
        # Draw seal
        # ------------------------------------------
        sx, sy = center(self.seal_pos)

        pygame.draw.circle(
            canvas,
            (97, 111, 121),
            (sx, sy),
            24,
        )

        pygame.draw.circle(
            canvas, (255, 255, 255),
            (sx - 8, sy - 6), 5
        )

        pygame.draw.circle(
            canvas, (255, 255, 255),
            (sx + 8, sy - 6), 5
        )

        pygame.draw.circle(
            canvas, (25, 35, 45),
            (sx - 8, sy - 6), 2
        )

        pygame.draw.circle(
            canvas, (25, 35, 45),
            (sx + 8, sy - 6), 2
        )

        pygame.draw.circle(
            canvas, (25, 35, 45),
            (sx, sy + 7), 5
        )

        # ------------------------------------------
        # Draw penguins
        # ------------------------------------------
        for i, agent in enumerate(
            self.possible_agents
        ):

            if not self.alive[agent]:
                continue

            x, y = center(self.positions[agent])

            # Penguin body
            pygame.draw.circle(
                canvas,
                self.colors[agent],
                (x, y),
                21,
            )

            # Belly
            pygame.draw.ellipse(
                canvas,
                (245, 250, 253),
                (x - 11, y - 4, 22, 19),
            )

            # Eyes
            pygame.draw.circle(
                canvas, (15, 25, 35),
                (x - 7, y - 9), 3
            )

            pygame.draw.circle(
                canvas, (15, 25, 35),
                (x + 7, y - 9), 3
            )

            # ID label
            label = self.font.render(
                f"P{i}",
                True,
                (20, 30, 40),
            )

            canvas.blit(
                label,
                label.get_rect(
                    center=(x, y + 30)
                ),
            )

        # ------------------------------------------
        # Scoreboard
        # ------------------------------------------
        pygame.draw.rect(
            canvas,
            (244, 248, 250),
            (0, self.width, self.width, 85),
        )

        remaining_fish = sum(
            fish is not None
            for fish in self.fish_positions
        )

        status = (
            f"Step: {self.current_step}/"
            f"{self.max_steps}    "
            f"Fish left: {remaining_fish}    "
            f"Alive: {sum(self.alive.values())}"
        )

        status_text = self.font.render(
            status, True, (35, 49, 62)
        )

        canvas.blit(
            status_text,
            (12, self.width + 8),
        )

        for i, agent in enumerate(
            self.possible_agents
        ):

            score = self.episode_returns[agent]
            count = self.fish_count[agent]

            text = (
                f"P{i}: {count} fish | "
                f"R={score:.1f}"
            )

            if not self.alive[agent]:
                text = f"P{i}: caught!"

            label = self.font.render(
                text, True, self.colors[agent]
            )

            canvas.blit(
                label,
                (10 + i * 200, self.width + 43),
            )

        # ------------------------------------------
        # Output
        # ------------------------------------------
        if self.render_mode == "rgb_array":
            return np.transpose(
                pygame.surfarray.array3d(canvas),
                (1, 0, 2),
            )

        if self.window is not None:
            self.window.blit(canvas, (0, 0))
            pygame.display.flip()
            self.clock.tick(
                self.metadata["render_fps"]
            )

        return None

    # ==============================================
    # Close
    # ==============================================
    def close(self):

        if pygame.get_init():
            pygame.quit()

        self.window = None
        self.clock = None
        self.font = None
