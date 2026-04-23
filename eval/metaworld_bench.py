"""MetaWorld benchmark evaluation logic."""

import logging
import time
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# MetaWorld ML10 and ML45 task lists
ML10_TASKS = [
    "reach-v2", "push-v2", "pick-place-v2", "door-open-v2", "drawer-open-v2",
    "drawer-close-v2", "button-press-topdown-v2", "peg-insert-side-v2",
    "window-open-v2", "window-close-v2",
]

ML45_TASKS = ML10_TASKS + [
    "coffee-button-v2", "coffee-pull-v2", "coffee-push-v2",
    "dial-turn-v2", "disassemble-v2", "door-close-v2",
    "door-lock-v2", "door-unlock-v2", "hand-insert-v2",
    "lever-pull-v2", "peg-unplug-side-v2",
    "pick-out-of-hole-v2", "pick-place-wall-v2",
    "plate-slide-v2", "plate-slide-back-v2",
    "plate-slide-back-side-v2", "plate-slide-side-v2",
    "push-back-v2", "push-wall-v2",
    "reach-wall-v2", "shelf-place-v2",
    "soccer-v2", "stick-pull-v2", "stick-push-v2",
    "sweep-into-v2", "sweep-v2",
    "bin-picking-v2", "box-close-v2",
    "button-press-v2", "button-press-topdown-wall-v2",
    "button-press-wall-v2", "faucet-close-v2",
    "faucet-open-v2", "hammer-v2", "handle-press-v2",
]


@dataclass
class MetaWorldResult:
    """Stores evaluation results for a single MetaWorld episode."""
    task_name: str
    success: bool
    total_steps: int
    end_effector_velocities: list
    trajectory: list
    final_reward: float
    elapsed_time: float


def load_metaworld_env(task_name="reach-v2", seed=42):
    """Load a MetaWorld environment.

    Args:
        task_name: Task identifier (e.g. reach-v2, push-v2).
        seed: Random seed for reproducibility.

    Returns:
        Initialized environment.
    """
    try:
        import metaworld
        ml1 = metaworld.ML1(task_name, seed=seed)
        env = ml1.train_classes[task_name]()
        task = ml1.train_tasks[0]
        env.set_task(task)
        logger.info("Loaded MetaWorld task: %s", task_name)
        return env
    except ImportError:
        logger.error("MetaWorld not installed. Install with: pip install metaworld")
        raise


def evaluate_episode(env, policy, max_steps=500):
    """Run a single evaluation episode.

    Args:
        env: MetaWorld environment instance.
        policy: Policy with a predict(obs) -> action method.
        max_steps: Maximum steps before timeout.

    Returns:
        MetaWorldResult with episode metrics.
    """
    obs, _ = env.reset()
    ee_velocities = []
    trajectory = []
    done = False
    step = 0
    total_reward = 0.0
    success = False
    start = time.time()

    while not done and step < max_steps:
        action = policy.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        ee_vel = obs[6:9] if len(obs) > 9 else np.zeros(3)
        ee_velocities.append(ee_vel.tolist())
        trajectory.append(action.tolist() if hasattr(action, "tolist") else action)

        if info.get("success", False):
            success = True

        step += 1

    elapsed = time.time() - start

    return MetaWorldResult(
        task_name=getattr(env, "_task_name", "unknown"),
        success=success,
        total_steps=step,
        end_effector_velocities=ee_velocities,
        trajectory=trajectory,
        final_reward=total_reward,
        elapsed_time=elapsed,
    )


def run_metaworld_benchmark(policy, task_suite="ml10", episodes_per_task=10, max_steps=500, seed=42):
    """Run the full MetaWorld benchmark.

    Args:
        policy: Policy object with predict(obs) method.
        task_suite: "ml10" or "ml45".
        episodes_per_task: Number of episodes per task.
        max_steps: Max steps per episode.
        seed: Random seed.

    Returns:
        Dict with aggregated metrics.
    """
    tasks = ML10_TASKS if task_suite == "ml10" else ML45_TASKS
    all_results = []

    for task_name in tasks:
        env = load_metaworld_env(task_name, seed)
        task_results = []

        for ep in range(episodes_per_task):
            result = evaluate_episode(env, policy, max_steps)
            result.task_name = task_name
            task_results.append(result)
            logger.info(
                "  %s ep %d: success=%s, reward=%.2f, steps=%d",
                task_name, ep, result.success, result.final_reward, result.total_steps,
            )

        all_results.extend(task_results)
        env.close()

    successes = [r.success for r in all_results]
    all_velocities = []
    for r in all_results:
        all_velocities.extend([np.linalg.norm(v) for v in r.end_effector_velocities])

    per_task_success = {}
    for r in all_results:
        per_task_success.setdefault(r.task_name, []).append(r.success)
    per_task_rates = {k: np.mean(v) for k, v in per_task_success.items()}

    metrics = {
        "task_suite": task_suite,
        "num_tasks": len(tasks),
        "episodes_per_task": episodes_per_task,
        "total_episodes": len(all_results),
        "success_rate": float(np.mean(successes)),
        "ee_velocity_variance": float(np.var(all_velocities)) if all_velocities else 0.0,
        "ee_velocity_mean": float(np.mean(all_velocities)) if all_velocities else 0.0,
        "avg_steps": float(np.mean([r.total_steps for r in all_results])),
        "avg_reward": float(np.mean([r.final_reward for r in all_results])),
        "avg_time": float(np.mean([r.elapsed_time for r in all_results])),
        "per_task_success_rate": per_task_rates,
    }

    logger.info("MetaWorld %s results: success_rate=%.2f%%, ee_vel_var=%.4f",
                task_suite, metrics["success_rate"] * 100, metrics["ee_velocity_variance"])
    return metrics
