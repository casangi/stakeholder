from ast import Store
import sys
import argparse
import subprocess

parser = argparse.ArgumentParser(description="Parse input to determine stakeholder test.")

parser.add_argument('--stakeholder-test', type=str, nargs=1, required=True, dest='test_name', action='store')
args = parser.parse_args()

if args.test_name[0] == 'standard_cube_briggsbwtaper':
    cmd = 'python3 -m scripts.test_standard_cube_briggsbwtaper'
    p = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr, shell=True)

elif args.test_name[0] == 'mosaic_cube_briggsbwtaper':
    cmd = 'python3 -m scripts.test_mosaic_cube_briggsbwtaper'
    p = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr, shell=True)

else:
    print('Unknown test:  '  + str(args.test_name))
