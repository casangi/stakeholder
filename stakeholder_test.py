import os
import sys
import yaml
import argparse
import subprocess


def spawn_test(test:str)->'None':
    cmd = 'python3 -m scripts.{}'.format(test)
    try:
        p = subprocess.check_call(cmd, stdout=sys.stdout, stderr=sys.stderr, shell=True)

    except subprocess.CalledProcessError as cpe:
        print('{}: Error in completion of spawned process.'.format(cpe))
#    p = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr, shell=False)



if __name__ == '__main__':
    
    # Load configuration file containing test dictionary
    with open(os.getcwd() + '/config/config.yaml') as file:
        config_file = yaml.safe_load(file)

    # Create the command-line parser and make it mutually exclusinve
    parser = argparse.ArgumentParser(description="Parse input to determine stakeholder test.")
    group  = parser.add_mutually_exclusive_group()

    # Parse command-line options
    group.add_argument('--stakeholder-test', nargs='+',  dest='test_name', action='store')
    group.add_argument('--all', dest='full_test', action='store_true')
    
    args = parser.parse_args()

    if args.full_test == True:
        for entry in config_file['tests']['all']:
            spawn_test(entry)
        
    else:
        for entry in args.test_name:
            if entry in config_file['tests'].keys():
                spawn_test(config_file['tests'][entry])

            else:
                print('Unknown test:  '  + str(entry))
