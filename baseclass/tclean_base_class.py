import os
import glob
import subprocess
import unittest
import numpy
import shutil
import scipy
import matplotlib.pyplot as pyplot
import json
import pickle

from casatestutils.imagerhelpers import TestHelpers

th = TestHelpers()

from casatestutils import generate_weblog
from casatestutils.stakeholder import almastktestutils

from casatools import ctsys, image
from casatasks import immoments
from casatasks.private.parallel.parallel_task_helper import ParallelTaskHelper
from casaviewer import imview

from baseclass.tclean_base import tclean_base_template

_ia = image()
ctsys_resolve = ctsys.resolve

# Location of data
data_path = ctsys_resolve('./data/')

# Save the dictionaries of the metrics to files (per test)
# mostly useful for the maintenance (updating the expected metric parameters based
# on the current metrics)
savemetricdict=True

## Base Test class with Utility functions
class test_tclean_base(unittest.TestCase, tclean_base_template):

    def set_file_path(self, path):
        if os.path.exists(path) is False:
            print('File path: ' + path + ' does not exist. Check input and try again.')
        else:
            self.data_path = path
            print('Setting data_path: ' + self.data_path)

    def set_test_dict(self, test_dict):
        self.test_dict = test_dict

    def setUp(self):
        self._myia = _ia
        
        # sets epsilon as a percentage (1%)
        self.epsilon = 0.01 
        
        self.msfile = ""
        self.img_subdir = 'testdir'
        self.parallel = False
        if ParallelTaskHelper.isMPIEnabled():
            self.parallel = True
        
        # Determine whether or not self.data_path exists. If it is, set_file_path() has
        # been run and self.data_path is a local data path. Otherwise, set self.data_path
        # to the path used for unittesting.

        if hasattr(self,'data_path'):
            pass
        else:
            print("Setting self.data_path to data_path")
            self.data_path = data_path  
        
        
        self.expdict_jsonfile = self.data_path+'test_stk_alma_pipeline_imaging_exp_dicts.json'
        self.refversion='6.3.0.22'

    def tearDown(self):
        generate_weblog("tclean_ALMA_pipeline", self.test_dict)
        print("Closing ia tool")
        self._myia.done()

    def getExpdicts(self, testname):
        ''' read the json file containung exp_dicts (fiducial metric values)
            for a specific test 
        '''
        self.exp_dicts = almastktestutils.read_testcase_expdicts(self.expdict_jsonfile, testname, self.refversion)

    # Separate functions here, for special-case tests that need their own MS.
    def prepData(self, msname=None):
        if msname != None:
            self.msfile = msname

    def delData(self, msname=None):
        del_files = [self.img_subdir]
        if msname != None:
            self.msfile=msname
        if (os.path.exists(self.msfile)):
            del_files.append(self.msfile)
        img_files = glob.glob(self.img+'*')
        del_files += img_files
        for f in del_files:
            shutil.rmtree(f)

    def prepInputmask(self, maskname=""):
        if maskname!="":
            self.maskname=maskname
        if (os.path.exists(self.maskname)):
            shutil.rmtree(self.maskname)
        shutil.copytree(refdatapath+self.maskname, self.maskname, symlinks=True)

    def check_dict_vals_beam(self, exp_dict, act_dict, suffix, epsilon=0.01):
        """ Compares expected dictionary with actual dictionary. Useful for comparing the restoring beam.

            Parameters
            ----------
            exp_dict: dictionary
                Expected values, as key:value pairs.
                Keys must match between exp_dict and act_dict.
                Values are compared between exp_dict and act_dict. A summary
                line is returned with the first mismatched value, or the last
                successfully matched value.
            act_dict: dictionary
                Actual values to compare to exp_dict (and just the values).
            suffix: string
                For use with summary print statements.
        """
        report = ''
        eps = epsilon
        passed = True
        chans = 0
        for key in exp_dict:
            result = th.check_val(act_dict[key], exp_dict[key],
                valname=suffix+' chan'+str(chans), epsilon=eps)[1]
            chans += 1
            if 'Fail' in result:
                passed = False
                break
        report += th.check_val(passed, True, valname=suffix+' chan'+str(chans), exact=True)[1]

        return report

    def copy_products(self, old_pname, new_pname, ignore=None):
        """ function to copy iter0 images to iter1 images
            (taken from pipeline)
        """
        imlist = glob.glob('%s.*' % old_pname)
        imlist = [xx for xx in imlist if ignore is None or ignore not in xx]
        for image_name in imlist:
            newname = image_name.replace(old_pname, new_pname)
            if image_name == old_pname + '.workdirectory':
                mkcmd = 'mkdir '+ newname
                os.system(mkcmd)
                self.copy_products(os.path.join(image_name, old_pname), \
                    os.path.join(newname, new_pname))
            else:
                shutil.copytree(image_name, newname, symlinks=True)

    def cube_beam_stats(self, image):
        """ function to return per-channel beam statistics
            will be deprecated and combined into image_stats
            once CASA beam issue is fixed
        """
        self._myia.open(image)

        bmin_dict = {}; bmaj_dict = {}; pa_dict = {}
        beam_dict = self._myia.restoringbeam()['beams']
        for item in beam_dict.keys():
            bmin_dict[item] = beam_dict[item]['*0']['minor']['value']
            bmaj_dict[item] = beam_dict[item]['*0']['major']['value']
            pa_dict[item] = beam_dict[item]['*0']['positionangle']['value']

        self._myia.close()

        return bmin_dict, bmaj_dict, pa_dict

    def save_dict_to_file(self, topkey, indict, outfilename, appendversion=True, outformat='JSON'):
        """ function that will save input Python dictionaries to a JSON file (default)
            or pickle file. topkey will be added as a top key for output (nested) dictionary
            and indict is stored under the key.
            Create a separate file with outfilename if appendversion=True casa version (based on
            casatasks version) will be appended to the output file name.
        """
        
        try:
            import casatasks as __casatasks
            casaversion = __casatasks.version_string()
            del __casatasks
        except:
            casaversion = ''

        if casaversion !='':
            casaversion = '_' + casaversion
        if type(indict) != dict:
            print("indict is not a dict. Saved file may not be in correct format")
        nestedDict={}
        nestedDict[topkey]=indict
        print("Saving %s dictionaries", len(indict))
        if outformat == 'pickle':
            
            # writing to pickle: note if writing this way (without protocol=2)
            # in casa6 and read in casa5 it will fail
            with open(outfilename+casaversion+'.pickle', 'wb') as outf:
                pickle.dump(nestedDict, outf)
        elif outformat== 'JSON':
            with open(outfilename+casaversion+'.json', 'w') as outf:
                json.dump(nestedDict, outf)
        else:
            print("no saving with format:", outformat)

    def modify_dict(self, output=None, testname=None, parallel=None):
        ''' Modified test_dict constructed by casatestutils add_to_dict to include only
            the task commands executed and also add self.parallel value to the dictionary.
            The cube imaging cases usually have if-else conditional based on parallel mode is on or not
            to trigger different set of tclean commands.
            Assumption: self.parallel is used to trigger different tclean commands at iter1 step.
            For self.parallel=True, iter1 has two tclean commands (2nd and 3rd tclean commands within
            each relevante test(cube) and so in test_dict['taskcall'], 1st(iter0) and 2nd and 3rd commands
            are the ones acutually executed and should remove 4th (self.parallel=False) case.
        '''
        if testname in output:
            if 'taskcall' in output[testname] and len(output[testname]['taskcall'])==3:
                if parallel:
                    # 0,1,2th in the list are used pop last one
                    output[testname]['taskcall'].pop()
                else:
                    output[testname]['taskcall'].pop(1)
            output[testname]['self.parallel']=parallel

    def remove_prefix(self,string, prefix):
        ''' Remove a specified prefix string from string '''
        return string[string.startswith(prefix) and len(prefix):]