import os
import numpy
import shutil
import scipy
import glob
import unittest
import json
import pickle
import matplotlib.pyplot as pyplot

from casatasks import immoments
from casatasks.private.parallel.parallel_task_helper import ParallelTaskHelper

from casaviewer import imview
from casatools import ctsys, image

from casatestutils import generate_weblog
from casatestutils.stakeholder import almastktestutils
from casatestutils.imagerhelpers import TestHelpers

th = TestHelpers()

from scripts.baseclass.stk_test_base import stakeholder_baseclass_template

_ia = image()
ctsys_resolve = ctsys.resolve

# Location of data
data_path = ctsys_resolve('stakeholder/alma/')

# Save the dictionaries of the metrics to files (per test)
# mostly useful for the maintenance (updating the expected metric parameters based
# on the current metrics)
savemetricdict=True

## Base Test class with Utility functions
class test_stakeholder_base(unittest.TestCase, stakeholder_baseclass_template):

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

        if (hasattr(self, 'test_dict')):
            generate_weblog("tclean_ALMA_pipeline", self._test_dict)
        print("Closing ia tool")
        self._myia.done()

    def get_exec_env(self):
        """ Attempt to determine whether we're running in a Jupyter notebook ('ipynb'/'ipynb_colab') or some other environment.

        See also: https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
        """
        try:
            shell = get_ipython().__class__.__name__
            if shell == 'ZMQInteractiveShell':        # Jupyter notebook or qtconsole
                if get_ipython().__class__.__module__ == "google.colab._shell":
                    return 'ipynb_colab'
                return 'ipynb'
            elif shell == 'TerminalInteractiveShell': # Terminal running IPython
                return 'shell'
            else:                                     # Other type (?)
                return 'unknown'
        except NameError:                             # Probably standard Python interpreter
            return 'python'

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

    def image_stats(self, image, fit_region=None, field_regions=None, masks=None):
        """ function that takes an image file and returns a statistics
            dictionary
        """
        self._myia.open(image)
        imagename=os.path.basename(image)
        stats_dict = {}

        statistics = self._myia.statistics()
        
        # Return data chunk; transpose to make channel selection easier
        chunk = numpy.transpose(self._myia.getchunk(dropdeg=True))

        # stats returned for all images
        im_size = self._myia.boundingbox()['imageShape'].tolist()
        stats_dict['npts'] = im_size[0]*im_size[1]*im_size[3]
        stats_dict['npts_unmasked'] = statistics['npts'][0]
        stats_dict['npts_real'] = numpy.count_nonzero(~numpy.isnan(chunk))
        stats_dict['freq_bin'] = self._myia.summary()['incr'][3]
        stats_dict['start'] = float( \
            statistics['blcf'].split(', ')[3].split('Hz')[0])
        stats_dict['end'] = float( \
            statistics['trcf'].split(', ')[3].split('Hz')[0])
        stats_dict['start_delta'] = stats_dict['start']
        stats_dict['end_delta'] = stats_dict['end']
        stats_dict['nchan'] = im_size[3]


        # stats returned for all images except .mask
        if not image.endswith('.mask'):
            stats_dict['max_val'] = statistics['max'][0]
            stats_dict['max_val_pos'] = statistics['maxpos'].tolist()
            max_loc = [stats_dict['max_val_pos'][0], \
                stats_dict['max_val_pos'][1]]
            stats_dict['min_val'] = statistics['min'][0]
            stats_dict['min_val_pos'] = statistics['minpos'].tolist()
            stats_dict['im_rms'] = statistics['rms'][0]

        # stats returned if a region file is given
        if fit_region != None:
            if '_cube' in imagename:
                if '.pb' in imagename:
                    fit_region = fit_region + ', range=[%schan,%schan]'\
                        % (int(im_size[3]/2), int(im_size[3]/2))
                if '.psf' in imagename:

                    # using chan 1 as first because ia.fitcomponents fits
                    # every channel if chan=0
                    fit_regions = [(fit_region + ', range=[%schan,%schan]' \
                                   % (1, 1)), \
                                  (fit_region + ', range=[%schan,%schan]' \
                                   % (int(im_size[3]/2), int(im_size[3]/2))), \
                                  (fit_region + ', range=[%schan,%schan]' \
                                   % ((im_size[3]-1), (im_size[3]-1)))]
                    i = 0
                    for region in fit_regions:
                        try:
                            fit_dict = self._myia.fitcomponents( \
                                region=region)['results']['component0']
                            stats_dict['fit_'+str(i)] = [ \
                                fit_dict['peak']['value'], \
                                fit_dict['shape']['majoraxis']['value'], \
                                fit_dict['shape']['minoraxis']['value']]
                            stats_dict['fit_loc_chan_'+str(i)] = fit_dict['spectrum']['channel']
                            stats_dict['fit_loc_freq_'+str(i)] = \
                                fit_dict['spectrum']['frequency']['m0']['value']
                            stats_dict['fit_pix_'+str(i)] = \
                                fit_dict['pixelcoords'].tolist()
                        except KeyError:
                            stats_dict['fit_'+str(i)] = [1.0, 1.0, 1.0]
                            stats_dict['fit_loc_chan_'+str(i)] = 1.0
                            stats_dict['fit_loc_freq_'+str(i)] = 1.0
                            stats_dict['fit_pix_'+str(i)] = [1.0, 1.0]
                        i += 1
                if '.model' in imagename:
                    fit_region = fit_region
                if '.model' not in imagename and '.pb' not in imagename and '.psf' not in imagename:

                    # WARN: If max value channel is 0, tool fits all channels
                    fit_region = fit_region + ', range=[%schan,%schan]' \
                        % (stats_dict['max_val_pos'][3], \
                        stats_dict['max_val_pos'][3])
            if '.psf' in imagename and '_cube' in imagename:
                stats_dict['regn_sum'] = self._myia.statistics( \
                    region=fit_regions[1])['sum'][0]
            else:
                stats_dict['regn_sum'] = self._myia.statistics( \
                    region=fit_region)['sum'][0]
            if ('image' in imagename and 'mosaic_cube_eph' not in imagename) or 'pb' in imagename or ('psf' in imagename and 'cube' not in imagename):
                try:
                    fit_dict = self._myia.fitcomponents( \
                        region=fit_region)['results']['component0']
                    stats_dict['fit'] = [fit_dict['peak']['value'], \
                        fit_dict['shape']['majoraxis']['value'], \
                        fit_dict['shape']['minoraxis']['value']]
                    stats_dict['fit_loc_chan'] = fit_dict['spectrum']['channel']
                    stats_dict['fit_loc_freq'] = fit_dict['spectrum']['frequency']['m0']['value']
                    stats_dict['fit_pix'] = fit_dict['pixelcoords'].tolist()
                except KeyError:
                    stats_dict['fit'] = [1.0, 1.0, 1.0]
                    stats_dict['fit_loc_chan'] = 1.0
                    stats_dict['fit_loc_freq'] = 1.0
                    stats_dict['fit_pix'] = [1.0, 1.0]

        # stats returned for .image(.tt0)
        if 'image' in imagename:
            commonbeam = self._myia.commonbeam()
            stats_dict['com_bmin'] = commonbeam['minor']['value']
            stats_dict['com_bmaj'] = commonbeam['major']['value']
            stats_dict['com_pa'] = commonbeam['pa']['value']
            if 'cube' in imagename:
                stats_dict['rms_per_chan'] = \
                    self._myia.statistics(axes=[0,1])['rms'].tolist()
                stats_dict['profile'] = self.cube_profile_fit( \
                    image, max_loc, stats_dict['nchan'])
            if 'mosaic' in imagename:
                stats_dict['rms_per_field'] = []
                for region in field_regions:
                    stats_dict['rms_per_field'].append( \
                        self._myia.statistics(region=region)['rms'][0])

        # stats returned if not .pb(.tt0), .sumwt(.tt0), or .mask
        # if 'pb' not in image and 'sumwt' not in image and not image.endswith('.mask'):
        stats_dict['im_sum'] = statistics['sum'][0]

        if image.endswith('.mask'):
            stats_dict['mask_pix'] = numpy.count_nonzero(chunk)
            stats_dict['mask_regns'] = scipy.ndimage.label(chunk)[1]
            stats_dict['mask'] = ~numpy.array(chunk, dtype=bool)

        if 'pb' in imagename:
            pb_mask_02 = chunk>0.2
            pb_mask_05 = chunk>0.5
            if 'cube' in image:
                pb_02_list = []
                pb_05_list = []
                i = 0
                for chan in chunk:
                    pb_02_list.append(numpy.count_nonzero(chan*pb_mask_02[i]))
                    pb_05_list.append(numpy.count_nonzero(chan*pb_mask_05[i]))
                    i += 1
                stats_dict['npts_0.2'] = pb_02_list
                stats_dict['npts_0.5'] = pb_05_list
            else:
                stats_dict['npts_0.2'] = numpy.count_nonzero(pb_mask_02)
                stats_dict['npts_0.5'] = numpy.count_nonzero(pb_mask_05)
            if 'mosaic' in imagename:
                stats_dict['pb_mask_0.2'] = pb_mask_02
                stats_dict['pb_mask_0.5'] = pb_mask_05

        if 'model' in imagename or image.endswith('.alpha'):
            stats_dict['mask_non0'] = numpy.count_nonzero(chunk*masks)

        if 'weight' in imagename:
            if 'cube' in imagename:
                wt_02_list = []
                wt_05_list = []
                i = 0
                for chan in chunk:
                    wt_02_list.append(numpy.count_nonzero(chan*masks[0][i]))
                    wt_05_list.append(numpy.count_nonzero(chan*masks[1][i]))
                    i += 1
                stats_dict['npts_0.2'] = wt_02_list
                stats_dict['npts_0.5'] = wt_05_list
            else:
                stats_dict['npts_0.2'] = numpy.count_nonzero(chunk*masks[0])
                stats_dict['npts_0.5'] = numpy.count_nonzero(chunk*masks[1])

        self._myia.close()

        return stats_dict

    def image_list(self, image, mode):
        """ function used to return expected imaging output files """
        standard = [image+'.psf', image+'.residual', image+'.image', \
            image+'.image.pbcor', image+'.mask', image+'.pb', image+'.model', \
            image+'.sumwt']
        mosaic = [image+'.weight']
        mtmfs = [image+'.alpha', image+'.alpha.error', image+'.alpha.pbcor', \
           image+'.psf.tt0', image+'.psf.tt1', image+'.psf.tt2', \
           image+'.residual.tt0', image+'.residual.tt1', image+'.image.tt0',\
           image+'.image.tt1', image+'.image.tt0.pbcor', image+'.image.tt1.pbcor', \
           image+'.mask', image+'.pb.tt0', image+'.model.tt0', image+'.model.tt1', \
           image+'.sumwt.tt0', image+'.sumwt.tt1', image+'.sumwt.tt2']
        mos_mtmfs = [image+'.weight.tt0', image+'.weight.tt1', image+'.weight.tt2']

        if mode == 'standard':
            img_list = standard
        if mode == 'mosaic':
            img_list = standard+mosaic
        if mode == 'mtmfs':
            img_list = mtmfs
        if mode == 'mos_mtmfs':
            img_list = mtmfs+mos_mtmfs

        return img_list

    def mom8_creator(self, image, range_list):
        """ function that takes and image and turns it into a .png for
            weblog
        """
        immoments(imagename = image, moments = 8, outfile = image+'.moment8')
        imview(raster={'file': image+'.moment8', 'range': range_list}, \
            out = {'file': image+'.moment8.png'})
        subprocess.call('mogrify -trim '+image+'.moment8.png', shell=True)

    def cube_profile_fit(self, image, max_loc, nchan):
        """ function that will retrieve a profile for cubes at the max position
            and create a png showing the profile plot; must be called with
            image already opened
        """
        
        pyplot.clf()
        
        box = str(max_loc[0])+','+str(max_loc[1])+','+str(max_loc[0])+','+str(max_loc[1])
        profile = self._myia.fitprofile(box=box)['gs']['amp'][0][0][0][0][0]
        
        X = self._myia.getchunk(blc=max_loc, trc=max_loc, axes=[0,1])[0][0][0]
        
        pyplot.title('Frequency Profile at Max Value Position')
        pyplot.xlabel('Channel Number')
        pyplot.xlim(0,(nchan+1))
        pyplot.ylabel('Amplitude (Jy/Beam)')
        pyplot.plot(X)
        pyplot.savefig(image+'.profile.png')
        pyplot.clf()

        return profile

    def filter_report(self, report, showonlyfail=True):
        """ function to filter the test report, the input report is expected to be a string with the newline code """
        
        ret = ''
        if showonlyfail:
            filter='Fail'
        else:
            filter='Pass'

        if report!='':
            testItems = report.split('\n')
            retitems=[]
            for testitem in testItems:
                if '[ check_ims ]' in testitem or '[ check_pixmask ]' in testitem or '[ check_val ]' in testitem:
                    if '( '+filter in testitem:
                        retitems.append(testitem)
            nfail = len(retitems)
            msg = str(nfail)+' individual test failure(s) '
            ret = '\n' + '\n'.join(retitems)
            ret += '\n' + msg
        return ret
