import os
import glob
import unittest
import json
import pickle
import shutil

from casatestutils.imagerhelpers import TestHelpers

th = TestHelpers()

from casatestutils import generate_weblog
from casatestutils.stakeholder import almastktestutils

from casatools import ctsys, image
from casatasks.private.parallel.parallel_task_helper import ParallelTaskHelper

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
    
    def setUp(self):
        """ Setup function for unit testing. """

        self._myia = _ia
        self._test_dict = None
        
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
        """ Teardown function for unit testing. """

        if self.test_dict != None:
            generate_weblog("tclean_ALMA_pipeline", self._test_dict)
        print("Closing ia tool")
        self._myia.done()

    def set_file_path(self, path):
        """ Utility function that is sued to set the internal data path directory.

        Args:
            path (str): Path to the internally managed data path.
        """

        if os.path.exists(path) is False:
            print('File path: ' + path + ' does not exist. Check input and try again.')
        else:
            self.data_path = path
            print('Setting data_path: ' + self.data_path)

    @property
    def test_dict(self)->dict:
        """ Standard getter fucntion for test_dict value. 

        Returns:
            dict: Internal test_dict.
        """

        return self._test_dict

    @test_dict.setter
    def test_dict(self, test_dict:dict)->None:
        """ Standard setter function for test_dict.

        Args:
            test_dict (dict): Internal test_dict.
        """

        print('Setting test dictionary value.')

        self._test_dict = test_dict

    @property
    def exp_dict(self)->dict:
        """[summary]

        Returns:
            [dict]: Expected metric values JSON file
        """
        return self._exp_dicts

    @exp_dict.setter
    def exp_dict(self, exp_dict:dict)->None:
        """[summary]

        Args:
            exp_dict (dict): Expected metric values JSON file.
        """
        self._exp_dicts = exp_dict

    def load_exp_dicts(self, testname:str)->None:
        """ Sets the fiducial metric values for a specific unit test, in json format.

        Args:
            testname (str): Nmae of unit test.
        """
        
        self._exp_dicts = almastktestutils.read_testcase_expdicts(self.expdict_jsonfile, testname, self.refversion)

    # Separate functions here, for special-case tests that need their own MS.
    def prepData(self, msname=None):
        """ Prepare the data for the unit test.

        Args:
            msname (str, optional): Measurement file. Defaults to None.
        """

        if msname != None:
            self.msfile = msname

    def delData(self, msname=None):
        """ Clean up generated data for a given test.

        Args:
            msname (str, optional): Measurement file. Defaults to None.
        """

        del_files = []
        if (os.path.exists(self.img_subdir)):
            del_files.append(self.img_subdir)
        if msname != None:
            self.msfile=msname
        if (os.path.exists(self.msfile)):
            del_files.append(self.msfile)
        img_files = glob.glob(self.img+'*')
        del_files += img_files
        if hasattr(self, 'imgs'):
            for img in self.imgs:
                img_files = glob.glob(img+'*')
                del_files += img_files
        for f in del_files:
            shutil.rmtree(f)

    def prepInputmask(self, maskname=""):
        if maskname!="":
            self.maskname=maskname
        if (os.path.exists(self.maskname)):
            shutil.rmtree(self.maskname)
        shutil.copytree(refdatapath+self.maskname, self.maskname, symlinks=True)

    def check_dict_vals_beam(self, exp_dict:dict, act_dict:dict, suffix:str, epsilon=0.01):
        """ Compares expected dictionary with actual dictionary. Useful for comparing the restoring beam.

        Args:
            exp_dict (dict): Expected values, as key:value pairs.
                Keys must match between exp_dict and act_dict.
                Values are compared between exp_dict and act_dict. A summary
                line is returned with the first mismatched value, or the last
                successfully matched value.

            act_dict (dict): [description]
            suffix (str): Actual values to compare to exp_dict (and just the values).
            epsilon (float, optional): Allowed variance from fiducial values. Defaults to 0.01.

        Returns:
            [type]: Report detailing results of fiducial checks.
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

    def copy_products(self, old_pname:str, new_pname:str, ignore=None):
        """ Function to copy iter0 images to iter1 images (taken from pipeline).

        Args:
            old_pname (str): Old filename
            new_pname (str): New filename.
            ignore (bool, optional): [description]. Defaults to None.
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

    def cube_beam_stats(self, image:'CASAImage')->dict:
        """ Function to return per-channel beam statistics .

        Args:
            image (CASAImage): Image to analyze.

        Returns:
            dict: Beam statistics dictionaries.
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

    def save_dict_to_file(self, topkey:str, indict:str, outfilename:str, appendversion=True, outformat='JSON')->None:
        """ Function that will save input Python dictionaries to a JSON file (default)
            or pickle file. topkey will be added as a top key for output (nested) dictionary
            and indict is stored under the key.
            
            Create a separate file with outfilename if appendversion=True casa version (based on
            casatasks version) will be appended to the output file name.

        Args:
            topkey (str): [description]
            indict (dict): [description]
            outfilename (str): [description]
            appendversion (bool, optional): [description]. Defaults to True.
            outformat (str, optional): [description]. Defaults to 'JSON'.
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

    def modify_dict(self, output=None, testname=None, parallel=None)->None:
        """ Modified test_dict constructed by casatestutils add_to_dict to include only
            the task commands executed and also add self.parallel value to the dictionary.
            The cube imaging cases usually have if-else conditional based on parallel mode is on or not
            to trigger different set of tclean commands.

            Assumption: self.parallel is used to trigger different tclean commands at iter1 step.
            For self.parallel=True, iter1 has two tclean commands (2nd and 3rd tclean commands within
            each relevante test(cube) and so in test_dict['taskcall'], 1st(iter0) and 2nd and 3rd commands
            are the ones acutually executed and should remove 4th (self.parallel=False) case.

        Args:
            output (dict, optional): [description]. Defaults to None.
            testname (str, optional): [description]. Defaults to None.
            parallel (bool, optional): [description]. Defaults to None.
        """

        if testname in output:
            if 'taskcall' in output[testname] and len(output[testname]['taskcall'])==3:
                if parallel:
                    # 0,1,2th in the list are used pop last one
                    output[testname]['taskcall'].pop()
                else:
                    output[testname]['taskcall'].pop(1)
            output[testname]['self.parallel']=parallel

    def remove_prefix(self, string:str, prefix:str)->str:
        """ Remove a specified prefix string from string.

        Args:
            string (str): [description]
            prefix (str): [description]

        Returns:
            str: [description]
        """
        
        return string[string.startswith(prefix) and len(prefix):]
