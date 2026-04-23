"""LIBERO benchmark evaluation logic."""

import logging
import time
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LiberoResult:
    """Stores evaluation results for a single LIBERO episode."""
    task_name: str
    success: bool
    total_steps: int
    end_effector_velocities: list
    trajectory: list
    elapsed_time: float


def load_libero_env(task_suite="libero_spatial", task_id=0):
    """Load a LIBERO environment by suite and task ID.

    Args:
        task_suite: One of libero_spatial, libero_object, libero_goal, libero_10, libero_90.
        task_id: Task index within the suite.

    Returns:
        Initialized environment and task description.
    """
    try:
        from libero.libero import benchmark
        bench = benchmark.get_benchmark(task_suite)
        task = bench.get_task(task_id)

        from libero.libero.envs import OffScreenRenderEnv
        env = OffScreenRenderEnv(
            bddl_file_name=task.bddl_file,
            render_gpu_device_id=0,
        )
        task_description = task.language
        logger.info("Loaded LIBERO task: %s (suite=%s, id=%d)", task_description, task_suite, task_id)
        return env, task_description
    except ImportError:
        logger.error("LIBERO not installed. Install with: pip install libero")
        raise


def evaluate_episode(env, policy, max_steps=600):
    """Run a single evaluation episode.

    Args:
        env: LIBERO environment instance.
        policy: Policy with a predict(obs) -> action method.
        max_steps: Maximum steps before timeout.

    Returns:
        LiberoResult with episode metrics.
    """
    obs = env.reset()
    ee_velocities = []
    trajectory = []
    done = False
    step = 0
    start = time.time()

    while not done and step < max_steps:
        action = policy.predict(obs)
        obs, reward, done, info = env.step(action)

        if hasattr(obs, "get"):
            ee_vel = obs.get("robot0_eef_vel_lin", np.zeros(3))
        else:
            ee_vel = np.zeros(3)

        ee_velocities.append(ee_vel.tolist())
        trajectory.append(action.tolist() if hasattr(action, "tolist") else action)
        step += 1

    elapsed = time.time() - start
    success = bool(info.get("success", reward > 0.5)) if isinstance(info, dict) else reward > 0.5

    return LiberoResult(
        task_name=getattr(env, "task_description", "unknown"),
        success=success,
        total_steps=step,
        end_effector_velocities=ee_velocities,
        trajectory=trajectory,
        elapsed_time=elapsed,
    )


def run_libero_benchmark(policy, task_suite="libero_spatial", num_tasks=5, episodes_per_task=10, max_steps=600):
    """Run the full LIBERO benchmark.

    Args:
        policy: Policy object with predict(obs) method.
        task_suite: LIBERO task suite name.
        num_tasks: Number of tasks to evaluate.
        episodes_per_task: Number of episodes per task.
        max_steps: Max steps per episode.

    Returns:
        Dict with aggregated metrics.
    """
    all_results = []

    for task_id in range(num_tasks):
        env, task_desc = load_libero_env(task_suite, task_id)
        task_results = []

        for ep in range(episodes_per_task):
            result = evaluate_episode(env, policy, max_steps)
            result.task_name = task_desc
            task_results.append(result)
            logger.info(
                "  Task %d ep %d: success=%s, steps=%d",
                task_id, ep, result.success, result.total_steps,
            )

        all_results.extend(task_results)
        env.close()

    successes = [r.success for r in all_results]
    all_velocities = []
    for r in all_results:
        all_velocities.extend([np.linalg.norm(v) for v in r.end_effector_velocities])

    metrics = {
        "task_suite": task_suite,
        "num_tasks": num_tasks,
        "episodes_per_task": episodes_per_task,
        "total_episodes": len(all_results),
        "success_rate": np.mean(successes),
        "ee_velocity_variance": float(np.var(all_velocities)) if all_velocities else 0.0,
        "ee_velocity_mean": float(np.mean(all_velocities)) if all_velocities else 0.0,
        "avg_steps": float(np.mean([r.total_steps for r in all_results])),
        "avg_time": float(np.mean([r.elapsed_time for r in all_results])),
    }

    logger.info("LIBERO %s results: success_rate=%.2f%%, ee_vel_var=%.4f",
                task_suite, metrics["success_rate"] * 100, metrics["ee_velocity_variance"])
    return metrics
