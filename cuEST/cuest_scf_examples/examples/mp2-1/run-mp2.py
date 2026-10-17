import argparse
import os
import re
import subprocess
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument('geometry_dir')
parser.add_argument('--scripts', nargs='+', default=['test.py', 'test-strided.py', 'test-one-shot.py'])
parser.add_argument('--part-timers', action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--output-dir', default='mp2-runs')
args = parser.parse_args()

filedir = os.path.dirname(os.path.realpath(__file__))
geometries = sorted((f for f in os.listdir(args.geometry_dir) if f.endswith('.xyz')), key=lambda f: [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', f)])
os.makedirs(args.output_dir, exist_ok=True)

env = dict(os.environ, PYTHONUNBUFFERED='1')
timer_flag = '--part-timers' if args.part_timers else '--no-part-timers'

summary_path = os.path.join(args.output_dir, 'summary.txt')
with open(summary_path, 'w') as summary:
    for geometry in geometries:
        geometry_path = os.path.abspath(os.path.join(args.geometry_dir, geometry))
        for script in args.scripts:
            log_path = os.path.join(args.output_dir, f'{os.path.splitext(geometry)[0]}_{os.path.splitext(script)[0]}.log')
            command = [sys.executable, os.path.join(filedir, script), '--geometry', geometry_path, timer_flag]

            start = time.perf_counter()
            with open(log_path, 'w') as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
            wall = time.perf_counter() - start

            line = f'{geometry:24s} {script:20s} exit={result.returncode:4d} wall={wall:9.2f}s log={log_path}'
            print(line, flush=True)
            summary.write(line + '\n')
            summary.flush()
