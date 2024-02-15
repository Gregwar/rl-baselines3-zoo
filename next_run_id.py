from rl_zoo3.utils import get_latest_run_id
import sys

if len(sys.argv) < 3:
    print("Usage: python next_run_id.py <algo> <env>")
    sys.exit(1)

algo = sys.argv[1]
env = sys.argv[2]
log_path = f"logs/{algo}/"

print(get_latest_run_id(log_path, env) + 1)

