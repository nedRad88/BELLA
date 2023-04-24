import sys
import subprocess
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Install necessary modules
file1 = open('requirements.txt', 'r')
requirements = file1.readlines()
for req in requirements:
    reqs = subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', req.strip("\n"), '--quiet'])

subprocess.call(['python3', 'experiments.py'])
