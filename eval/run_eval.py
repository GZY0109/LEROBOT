"""Unified evaluation entry point for ACT / Diffusion Policy / π0."""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

POLICY_CONFIGS = {
    "act": PROJECT_ROOT / "policies" / "act_config.yaml",
    "diffusion": PROJECT_ROOT / "policies" / "diffusion_config.yaml",
    "pi0": PROJECT_ROOT / "policies" / "pi0_config.yaml",
}


def load_config(policy_name):
    """Load YAML config for the specified policy."""
    config_path = POLICY_CONFIGS.get(policy_name)
    if config_path is None or not config_path.exists():
        raise FileNotFoundError(f"Config not found for policy '{policy_name}': {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_policy(policy_name, config, checkpoint=None):
    """Load and initialize a policy model.

    Args:
        policy_name: One of "act", "diffusion", "pi0".
        config: Parsed YAML config dict.
        checkpoint: Optional path to model checkpoint.

    Returns:
        Initialized policy object with a predict(obs) method.
    """
    if policy_name == "act":
        from policies.act_policy import ACTPolicy
        policy = ACTPolicy(config)
    elif policy_name == "diffusion":
        from policies.diffusion_policy import DiffusionPolicy
        policy = DiffusionPolicy(config)
    elif policy_name == "pi0":
        from policies.pi0_policy import Pi0Policy
        policy = Pi0Policy(config)
    else:
        raise ValueError(f"Unknown policy: {policy_name}")

    if checkpoint:
        policy.load(checkpoint)
        logger.info("Loaded checkpoint: %s", checkpoint)

    return policy


def run_evaluation(policy_name, env_name, episodes, config, checkpoint=None, task_suite=None):
    """Run evaluation on the specified environment.

    Args:
        policy_name: Policy identifier.
        env_name: "libero" or "metaworld".
        episodes: Number of episodes per task.
        config: Policy config dict.
        checkpoint: Optional checkpoint path.
        task_suite: Task suite within the environment.

    Returns:
        Dict of evaluation metrics.
    """
    policy = load_policy(policy_name, config, checkpoint)

    if env_name == "libero":
        from eval.libero_bench import run_libero_benchmark
        suite = task_suite or "libero_spatial"
        metrics = run_libero_benchmark(
            policy,
            task_suite=suite,
            episodes_per_task=episodes,
        )
    elif env_name == "metaworld":
        from eval.metaworld_bench import run_metaworld_benchmark
        suite = task_suite or "ml10"
        metrics = run_metaworld_benchmark(
            policy,
            task_suite=suite,
            episodes_per_task=episodes,
        )
    else:
        raise ValueError(f"Unknown environment: {env_name}")

    metrics["policy"] = policy_name
    metrics["env"] = env_name
    metrics["timestamp"] = datetime.now().isoformat()
    return metrics


def save_results(metrics, output_dir=None):
    """Save evaluation results to JSON."""
    if output_dir is None:
        output_dir = PROJECT_ROOT / "logs"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    policy = metrics.get("policy", "unknown")
    env = metrics.get("env", "unknown")
    filename = f"{date_str}_{policy}_{env}.json"
    filepath = output_dir / filename

    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    logger.info("Results saved to %s", filepath)
    return filepath


def main():
    parser = argparse.ArgumentParser(description="Unified evaluation for imitation learning policies")
    parser.add_argument("--policy", required=True, choices=["act", "diffusion", "pi0"],
                        help="Policy to evaluate")
    parser.add_argument("--env", required=True, choices=["libero", "metaworld"],
                        help="Evaluation environment")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes per task (default: 10)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--task-suite", type=str, default=None,
                        help="Task suite (e.g. libero_spatial, ml10)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save results")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Evaluation: policy=%s, env=%s, episodes=%d", args.policy, args.env, args.episodes)
    logger.info("=" * 60)

    config = load_config(args.policy)
    metrics = run_evaluation(
        args.policy, args.env, args.episodes, config,
        checkpoint=args.checkpoint,
        task_suite=args.task_suite,
    )

    save_results(metrics, args.output_dir)

    print("\n" + "=" * 60)
    print(f"Policy: {args.policy} | Env: {args.env}")
    print(f"Success Rate: {metrics['success_rate'] * 100:.1f}%")
    print(f"EE Velocity Variance: {metrics['ee_velocity_variance']:.4f}")
    print(f"Avg Steps: {metrics['avg_steps']:.0f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
