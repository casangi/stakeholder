import os
import sys
import yaml
import argparse
import subprocess


def spawn_test(test:str)->'None':
    cmd = 'python3 -m scripts.{}'.format(test)
    p = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr, shell=True)



if __name__ == '__main__':
    
    # Load configuration file containing test dictionary
    with open(os.getcwd() + '/config/config.yaml') as file:
        config_file = yaml.safe_load(file)

    # Create the command-line parser
    parser = argparse.ArgumentParser(description="Parse input to determine stakeholder test.")

    # Parse command-line options
    parser.add_argument('--stakeholder-test', type=str, nargs=1, required=True, dest='test_name', action='store')
    args = parser.parse_args()

    if args.test_name[0] in config_file['tests'].keys():
        spawn_test(config_file['tests'][args.test_name[0]])
    
    else:
        print('Unknown test:  '  + str(args.test_name))
