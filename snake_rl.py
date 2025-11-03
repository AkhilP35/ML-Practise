import argparse
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Sequence, Tuple

import pygame
import numpy as np

Point = Tuple[int, int]


@dataclass(frozen=True)
class SnakeStats:
    episode: int
    score: int
    steps: int
    epsilon: float


class SnakeGame:
    """Grid-based snake environment with relative actions.

    Actions:
        0 -> turn left
        1 -> go straight
        2 -> turn right
    """

    RIGHT: Point = (1, 0)
    DOWN: Point = (0, 1)
    LEFT: Point = (-1, 0)
    UP: Point = (0, -1)
    CLOCKWISE: Sequence[Point] = (RIGHT, DOWN, LEFT, UP)

    def __init__(self, width: int = 10, height: int = 10, seed: int | None = None) -> None:
        self.width = width
        self.height = height
        self.random = random.Random(seed)
        self.reset()

    def reset(self) -> Tuple[int, ...]:
        center_x = self.width // 2
        center_y = self.height // 2
        self.direction: Point = self.RIGHT
        self.snake: Deque[Point] = deque(
            [(center_x - i, center_y) for i in range(3)]
        )
        self.score = 0
        self.steps_without_food = 0
        self._place_food()
        return self._get_state()

    def step(self, action: int) -> Tuple[Tuple[int, ...], float, bool]:
        """Advance the environment one step."""
        if action not in (0, 1, 2):
            raise ValueError(f"Unsupported action {action}")

        self._move(action)
        reward = -0.01
        self.steps_without_food += 1

        if self._is_collision(self.head):
            return self._get_state(), -1.0, True

        self.snake.appendleft(self.head)

        if self.head == self.food:
            self.score += 1
            reward = 1.0
            self.steps_without_food = 0
            self._place_food()
        else:
            self.snake.pop()

        # Limit wandering to keep episodes focused on progress.
        max_idle_steps = self.width * self.height * 2
        if self.steps_without_food > max_idle_steps:
            return self._get_state(), -0.5, True

        return self._get_state(), reward, False

    def _place_food(self) -> None:
        available = set((x, y) for x in range(self.width) for y in range(self.height)) - set(self.snake)
        if not available:
            # Snake fills the board; treat as terminal win state.
            self.food = self.snake[0]
            return
        self.food = self.random.choice(tuple(available))

    def _move(self, action: int) -> None:
        idx = self.CLOCKWISE.index(self.direction)
        if action == 0:  # turn left
            idx = (idx - 1) % 4
        elif action == 2:  # turn right
            idx = (idx + 1) % 4
        # action == 1 keeps going straight
        self.direction = self.CLOCKWISE[idx]
        head_x, head_y = self.snake[0]
        dx, dy = self.direction
        self.head: Point = (head_x + dx, head_y + dy)

    def _is_collision(self, point: Point) -> bool:
        x, y = point
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True
        # Ignore tail because it will move away unless we're eating.
        body = list(self.snake)[:-1]
        return point in body

    def _danger_in_direction(self, direction: Point) -> int:
        head_x, head_y = self.snake[0]
        dx, dy = direction
        next_point = (head_x + dx, head_y + dy)
        return int(self._is_collision(next_point))

    def _get_state(self) -> Tuple[int, ...]:
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food

        dir_right = int(self.direction == self.RIGHT)
        dir_left = int(self.direction == self.LEFT)
        dir_up = int(self.direction == self.UP)
        dir_down = int(self.direction == self.DOWN)

        danger_straight = self._danger_in_direction(self.direction)
        danger_left = self._danger_in_direction(self._rotate_left(self.direction))
        danger_right = self._danger_in_direction(self._rotate_right(self.direction))

        food_left = int(food_x < head_x)
        food_right = int(food_x > head_x)
        food_up = int(food_y < head_y)
        food_down = int(food_y > head_y)

        return (
            danger_straight,
            danger_left,
            danger_right,
            dir_up,
            dir_down,
            dir_left,
            dir_right,
            food_up,
            food_down,
            food_left,
            food_right,
        )

    @staticmethod
    def _rotate_left(direction: Point) -> Point:
        idx = SnakeGame.CLOCKWISE.index(direction)
        return SnakeGame.CLOCKWISE[(idx - 1) % 4]

    @staticmethod
    def _rotate_right(direction: Point) -> Point:
        idx = SnakeGame.CLOCKWISE.index(direction)
        return SnakeGame.CLOCKWISE[(idx + 1) % 4]


class SnakeRenderer:
    """Minimal pygame renderer to watch trained Snake agents."""

    BACKGROUND = (30, 30, 30)
    FOOD = (200, 60, 60)
    HEAD = (80, 220, 100)
    BODY = (60, 170, 80)
    GRID = (45, 45, 45)
    HUD = (230, 230, 230)

    def __init__(self, width: int, height: int, block_size: int = 30) -> None:
        pygame.init()
        self.block_size = block_size
        self.width = width
        self.height = height
        try:
            self.surface = pygame.display.set_mode((width * block_size, height * block_size))
        except pygame.error as exc:  # pragma: no cover - headless environments
            pygame.quit()
            raise RuntimeError(
                "Unable to create a display surface. If you are running in a headless environment, "
                "run this script locally or configure an SDL video driver."
            ) from exc

        pygame.display.set_caption("Snake RL")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 18)

    def process_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True

    def draw(self, env: SnakeGame, episode: int) -> None:
        self.surface.fill(self.BACKGROUND)
        self._draw_grid()

        fx, fy = env.food
        food_rect = pygame.Rect(
            fx * self.block_size, fy * self.block_size, self.block_size, self.block_size
        )
        pygame.draw.rect(self.surface, self.FOOD, food_rect)

        for idx, (sx, sy) in enumerate(env.snake):
            segment_rect = pygame.Rect(
                sx * self.block_size, sy * self.block_size, self.block_size, self.block_size
            )
            color = self.HEAD if idx == 0 else self.BODY
            pygame.draw.rect(self.surface, color, segment_rect)

        hud_text = self.font.render(
            f"Episode {episode} | Score {env.score} | Length {len(env.snake)}",
            True,
            self.HUD,
        )
        self.surface.blit(hud_text, (10, 10))
        pygame.display.flip()

    def tick(self, fps: int) -> None:
        self.clock.tick(fps)

    def close(self) -> None:
        pygame.quit()

    def _draw_grid(self) -> None:
        for x in range(self.width):
            pygame.draw.line(
                self.surface,
                self.GRID,
                (x * self.block_size, 0),
                (x * self.block_size, self.height * self.block_size),
                1,
            )
        for y in range(self.height):
            pygame.draw.line(
                self.surface,
                self.GRID,
                (0, y * self.block_size),
                (self.width * self.block_size, y * self.block_size),
                1,
            )


class QLearningAgent:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        lr: float = 0.1,
        gamma: float = 0.9,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: int | None = None,
    ) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.random = random.Random(seed)

        self.q_table = np.zeros((state_size, action_size), dtype=float)

    def act(self, state_idx: int) -> int:
        if self.random.random() < self.epsilon:
            return self.random.randrange(self.action_size)
        return int(np.argmax(self.q_table[state_idx]))

    def learn(self, state_idx: int, action: int, reward: float, next_state_idx: int, done: bool) -> None:
        td_target = reward
        if not done:
            td_target += self.gamma * float(np.max(self.q_table[next_state_idx]))

        td_error = td_target - self.q_table[state_idx, action]
        self.q_table[state_idx, action] += self.lr * td_error

    def decay_epsilon(self) -> None:
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_end)


def state_to_index(state: Sequence[int]) -> int:
    idx = 0
    for bit in state:
        idx = (idx << 1) | int(bit)
    return idx


def train(
    episodes: int = 1000,
    width: int = 10,
    height: int = 10,
    seed: int | None = 0,
) -> Tuple[QLearningAgent, List[SnakeStats]]:
    env = SnakeGame(width=width, height=height, seed=seed)
    num_states = 2 ** len(env._get_state())
    agent = QLearningAgent(
        state_size=num_states,
        action_size=3,
        lr=0.1,
        gamma=0.9,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        seed=seed,
    )

    history: List[SnakeStats] = []
    for episode in range(1, episodes + 1):
        state = env.reset()
        state_idx = state_to_index(state)
        total_steps = 0

        while True:
            action = agent.act(state_idx)
            next_state, reward, done = env.step(action)
            next_state_idx = state_to_index(next_state)

            agent.learn(state_idx, action, reward, next_state_idx, done)
            state_idx = next_state_idx
            total_steps += 1

            if done:
                history.append(
                    SnakeStats(
                        episode=episode,
                        score=env.score,
                        steps=total_steps,
                        epsilon=agent.epsilon,
                    )
                )
                break

        agent.decay_epsilon()

    return agent, history


def play_episode(agent: QLearningAgent, width: int = 10, height: int = 10, seed: int | None = None) -> SnakeStats:
    env = SnakeGame(width=width, height=height, seed=seed)
    state = env.reset()
    state_idx = state_to_index(state)
    steps = 0

    while True:
        action = int(np.argmax(agent.q_table[state_idx]))
        next_state, _, done = env.step(action)
        state_idx = state_to_index(next_state)
        steps += 1
        if done:
            return SnakeStats(episode=0, score=env.score, steps=steps, epsilon=0.0)


def watch_agent(
    agent: QLearningAgent,
    width: int = 10,
    height: int = 10,
    seed: int | None = None,
    episodes: int = 1,
    speed: int = 12,
    block_size: int = 30,
) -> None:
    renderer = SnakeRenderer(width, height, block_size=block_size)
    base_seed = seed or 0

    try:
        for episode in range(1, episodes + 1):
            env = SnakeGame(width=width, height=height, seed=base_seed + episode)
            state = env.reset()
            state_idx = state_to_index(state)

            while True:
                if not renderer.process_events():
                    return

                action = int(np.argmax(agent.q_table[state_idx]))
                next_state, _, done = env.step(action)
                state_idx = state_to_index(next_state)

                renderer.draw(env, episode)
                renderer.tick(speed)

                if done:
                    renderer.draw(env, episode)
                    renderer.tick(speed)
                    time.sleep(0.4)
                    break
    finally:
        renderer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and visualize a Q-learning Snake agent.")
    parser.add_argument("--episodes", type=int, default=1500, help="Number of training episodes.")
    parser.add_argument("--width", type=int, default=10, help="Board width in grid cells.")
    parser.add_argument("--height", type=int, default=10, help="Board height in grid cells.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible training.")
    parser.add_argument("--watch", action="store_true", help="Render the trained agent playing the game.")
    parser.add_argument("--watch-episodes", type=int, default=3, help="Number of visualization episodes to run.")
    parser.add_argument("--fps", type=int, default=12, help="Playback speed in frames per second.")
    parser.add_argument("--block-size", type=int, default=32, help="Pixel size for each grid cell.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    trained_agent, stats = train(
        episodes=args.episodes,
        width=args.width,
        height=args.height,
        seed=args.seed,
    )

    tail = stats[-10:] if len(stats) >= 10 else stats
    print("Final epsilon:", round(trained_agent.epsilon, 3))
    print("Last episode scores:", [s.score for s in tail])

    evaluation = play_episode(trained_agent, width=args.width, height=args.height, seed=args.seed + 99)
    print(f"Evaluation run -> score: {evaluation.score}, steps survived: {evaluation.steps}")

    if args.watch:
        print("Opening visualization window (close it to exit)...")
        watch_agent(
            trained_agent,
            width=args.width,
            height=args.height,
            seed=args.seed,
            episodes=args.watch_episodes,
            speed=args.fps,
            block_size=args.block_size,
        )


if __name__ == "__main__":
    main()
