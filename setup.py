#!/usr/bin/python3

import os

if os.path.exists('data/E2E6.1.00034.S_tclean.ms') is False:
    os.system('wget -r -np -nH --cut-dirs=4 --reject "index.html*" https://www.cv.nrao.edu/~jhoskins/E2E6.1.00034.S_tclean.ms.tar')
    os.system('tar -xvf E2E6.1.00034.S_tclean.ms.tar')
    try:
        os.remove(os.getcwd(), 'E2E6.1.00034.S_tclean.ms.tar')
    except FileNotFound:
        pass
    
    os.system('mv E2E6.1.00034.S_tclean.ms data/')

if os.path.exists('data/test_stk_alma_pipeline_imaging_exp_dicts.json') is False:
    os.system('wget -r -np -nH --cut-dirs=4 --reject "index.html*" https://www.cv.nrao.edu/~jhoskins/test_stk_alma_pipeline_imaging_exp_dicts.json')
    os.system('mv test_stk_alma_pipeline_imaging_exp_dicts.json data/')
