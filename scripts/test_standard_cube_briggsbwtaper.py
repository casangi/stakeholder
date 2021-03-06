##########################################################################
##########################################################################
# nb1_test_mosaic_cube_briggsbwtaper.py
#
# Copyright (C) 2018
# Associated Universities, Inc. Washington DC, USA.
#
# This script is free software; you can redistribute it and/or modify it
# under the terms of the GNU Library General Public License as published by
# the Free Software Foundation; either version 2 of the License, or (at your
# option) any later version.
#
# This library is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library General Public
# License for more details.
#
# [https://open-jira.nrao.edu/browse/CAS-12428]
#
#
##########################################################################

'''
    Datasets (MOUS)
    E2E6.1.00034.S (uid://A002/Xcff05c/X1ec)
    2018.1.00879.S (uid://A001/X133d/X169f)
    E2E6.1.00020.S (uid://A002/Xcff05c/Xe5)
    2017.1.00750.T (uid://A001/X131b/X57)

    Test list - 22 total
    1a.  Single field(SF) cube with perchanweightdensity=False(pcwdF), weighting=briggs - E2E6.1.00034.S
    1b.  SF cube with pcwdT, weighting=briggs - E2E6.1.00034.S
    1c.  SF cube with pcwdT, weighting=briggsbwtaper - E2E6.1.00034.S
    2.   SF MFS - E2E6.1.00020.S
    3.   SF mtmfs - E2E6.1.00020.S
    4a.  SF ephemeris cube (multi-EB) with pcwdF+briggs - 2017.1.00750.T
    4b.  SF ephemeris cube (multi-EB) with pcwdT+briggs - 2017.1.00750.T
    4c.  SF ephemeris cube (multi-EB) with pcwdT+briggsbwtaper - 2017.1.00750.T
    5.   SF ephemeris MFS - 2018.1.00879.S
    6.   SF ephemeris mtmfs - 2018.1.00879.S
    7.   SF Calibrator - E2E6.1.00034.S
    8.   SF ephemeris Calibrator - 2018.1.00879.S
    9a.  Mosaic cube with pcwdF, briggs- E2E6.1.00034.S
    9b.  Mosaic cube with pcwdT+brigs- E2E6.1.00034.S
    9c.  Mosaic cube with pcwdT+briggsbwtaper- E2E6.1.00034.S
    10.  Mosaic MFS - E2E6.1.00020.S
    11.  Mosaic mtmfs - E2E6.1.00020.S
    12a. Mosaic ephemeris cube with pcwdF- 2018.1.00879.S
    12b. Mosaic ephemeris cube with pcwdT+briggs - 2018.1.00879.S
    12c. Mosaic ephemeris cube with pcwdT+briggsbwtaper - 2018.1.00879.S
    13.  Mosaic ephemeris MFS - 2018.1.00879.S
    14.  Mosaic ephemeris mtmfs - 2018.1.00879.S

    Each test stores reference values in dictionaries for the metrics
    to be tested and these dictionaries are stored in a single nested dictionary
    in a json file located in the casatestdata repository. 
    The path of json file is stored in the variable, 
        self.expdict_jsonfile  
    in test_tclean_base.setUp(). 

    * NOTE for updating the tests and fiducial values in json file *
    When the json file is updated and its 'casa_version'
    could also be updated then self.refversion in the setUp() needs to be updated to
    match with the 'casa_version' as defined in the json file otherwise 
    almastkteestutils.read_testcase_expdicts() print an error message.

    The fudicial metric values for a specific image are stored with the following keys.
    
    For the standard tests, default sets are:
        exp_im_stats, exp_mask_stats, exp_pb_stats, exp_psf_stats,
        exp_model_stats, exp_resid_stats, exp_sumwt_stats
    For mosaic tests, the ones above and
        exp_wt_stats (for mosaic)
    Additionally, for cube imaging (if self.parallel=True),
        exp_bmin_dict, exp_bmaj_dict, exp_pa_dict
    And for mtmfs
        exp_im1_stats, exp_model1_stats, exp_resid1_stats, exp_sumwt1_stats

'''

##########################################################################
##########################################################################

import os
import unittest
import shutil
import matplotlib.pyplot as pyplot
import copy
import casatasks

from casatestutils.imagerhelpers import TestHelpers

th = TestHelpers()

from casatestutils import generate_weblog
from casatestutils import add_to_dict
from casatestutils import stats_dict
from casatestutils.stakeholder import almastktestutils

from casatools import ctsys, image
from casatasks import tclean, immoments
from casatasks.private.parallel.parallel_task_helper import ParallelTaskHelper
from casatasks.private.imagerhelpers.parallel_imager_helper import PyParallelImagerHelper

# ===== Make sure we can find the libraries =====
import sys

__basename  = os.path.basename(__file__)
__stakeholder_path = os.path.realpath(__file__).split("scripts/" + __basename)[0]
sys.path.append(__stakeholder_path)

# ===============================================

import stk_utils.plot_utils as plt_utils

from scripts.baseclass.stakeholder_base_class import test_stakeholder_base

_ia = image()
ctsys_resolve = ctsys.resolve

# location of data
data_path = ctsys_resolve('stakeholder/alma/')

# save the dictionaries of the metrics to files (per test)
# mostly useful for the maintenance (updating the expected metric parameters based
# on the current metrics)
savemetricdict=True

test_dict = {}
class Test_standard(test_stakeholder_base):

    #Test 1a
    @stats_dict(test_dict)
    def test_standard_cube_briggsbwtaper(self):
        ''' Standard (single field) cube imaging with briggsbwtaper - central field of SMIDGE_NWCloud (field 3), spw 22
        '''


        # If running as a unittest, the testMethodName will be set to the method name,
        # otherwise it will be "runTest". If the later is the case, get the method name
        # from the stack and and set the test_name manually.
        
        if self._testMethodName is "runTest":
            import inspect

            self.test_name = inspect.currentframe().f_code.co_name         

            self.file_name = self.remove_prefix(self.test_name, 'test_')+'.iter'
            self.img = os.getcwd()+'/'+self.file_name+'1'
            self.prepData(self.data_path+'E2E6.1.00034.S_tclean.ms')
            self.load_exp_dicts('test_standard_cube_briggsbwtaper')
        else:
            
            self.test_name = self._testMethodName

            self.file_name = self.remove_prefix(self.test_name, 'test_')+'.iter'
            self.img = os.getcwd()+'/'+self.file_name+'1'
            self.set_file_path(data_path)
            self.prepData(data_path+'E2E6.1.00034.S_tclean.ms')
            self.load_exp_dicts('test_standard_cube_briggsbwtaper')
            self.standard_cube_clean()
            self.standard_cube_report()

    def standard_cube_clean(self):
        print("\nSTARTING: iter0 routine")
        msfile = self.msfile
        file_name = self.file_name
        parallel = self.parallel

        # %% test_standard_cube_briggsbwtaper_tclean_1 start @

        casatasks.tclean(vis=msfile, 
                         imagename=file_name+'0', 
                         field='1',
                         spw=['0'], 
                         imsize=[80, 80], 
                         antenna=['0,1,2,3,4,5,6,7,8'], 
                         scan=['8,12,16'], 
                         intent='OBSERVE_TARGET#ON_SOURCE',
                         datacolumn='data', 
                         cell=['1.1arcsec'], 
                         phasecenter='ICRS 00:45:54.3836 -073.15.29.413', 
                         stokes='I', 
                         specmode='cube',
                         nchan=508, 
                         start='220.2526743594GHz', 
                         width='0.2441741MHz',
                         outframe='LSRK', 
                         pblimit=0.2, 
                         perchanweightdensity=True,
                         gridder='standard', 
                         mosweight=False,
                         deconvolver='hogbom', 
                         usepointing=False, 
                         restoration=False,
                         pbcor=False, 
                         weighting='briggsbwtaper', 
                         restoringbeam='common',
                         robust=0.5, npixels=0, 
                         niter=0, 
                         threshold='0.0mJy', 
                         nsigma=0.0,
                         interactive=0, 
                         usemask='auto-multithresh',
                         sidelobethreshold=1.25, 
                         noisethreshold=5.0,
                         lownoisethreshold=2.0, 
                         negativethreshold=0.0, 
                         minbeamfrac=0.1,
                         growiterations=75, 
                         dogrowprune=True, 
                         minpercentchange=1.0,
                         fastnoise=False, 
                         savemodel='none', 
                         parallel=parallel,
                         verbose=True)

        # %% test_standard_cube_briggsbwtaper_tclean_1 end @

        # move files to iter1
        print('Copying iter0 files to iter1')
        self.copy_products(file_name+'0', file_name+'1')

        print("STARTING: iter1 routine")

        # %% test_standard_cube_briggsbwtaper_tclean_2 start @

        casatasks.tclean(vis=msfile, 
                         imagename=file_name+'1', 
                         field='1',
                         spw=['0'], 
                         imsize=[80, 80], 
                         antenna=['0,1,2,3,4,5,6,7,8'],
                         scan=['8,12,16'], 
                         intent='OBSERVE_TARGET#ON_SOURCE',
                         datacolumn='data', 
                         cell=['1.1arcsec'], 
                         phasecenter='ICRS 00:45:54.3836 -073.15.29.413', 
                         stokes='I', 
                         specmode='cube',
                         nchan=508, 
                         start='220.2526743594GHz', 
                         width='0.2441741MHz',
                         outframe='LSRK', 
                         perchanweightdensity=True,
                         usepointing=False, 
                         pblimit=0.2, 
                         nsigma=0.0,
                         gridder='standard', 
                         mosweight=False, 
                         deconvolver='hogbom', 
                         restoration=True, 
                         restoringbeam='common', 
                         pbcor=True, 
                         weighting='briggsbwtaper', 
                         robust=0.5, 
                         npixels=0, 
                         niter=20000,
                         threshold='0.354Jy', 
                         interactive=0, 
                         usemask='auto-multithresh', 
                         sidelobethreshold=1.25, 
                         noisethreshold=5.0, 
                         lownoisethreshold=2.0, 
                         negativethreshold=0.0,
                         minbeamfrac=0.08, 
                         growiterations=75, 
                         dogrowprune=True,
                         minpercentchange=1.0, 
                         fastnoise=False, 
                         restart=True, 
                         calcres=False, 
                         calcpsf=False, 
                         savemodel='none',
                         parallel=parallel, 
                         verbose=True)

        # %% test_standard_cube_briggsbwtaper_tclean_2 end @

    def standard_cube_report(self):
        # retrieve per-channel beam statistics
        bmin_dict, bmaj_dict, pa_dict = \
            self.cube_beam_stats(image=self.img+'.psf')

        report0 = th.checkall(imgexist = self.image_list(self.img, 'standard'))

        # .image report(test_standard_cube)
        im_stats_dict = self.image_stats(image=self.img+'.image', fit_region = \
            'ellipse[[11.47881897deg, -73.25881015deg], [9.0414arcsec, 8.4854arcsec], 90.00000000deg]')

        # test_standard_cube.exp_im_stats
        exp_im_stats = self._exp_dicts['exp_im_stats']

        report1 = th.checkall(
            # checks for image and pb mask movement
            imgmask = [(self.img+'.image', True, [40, 70, 0, 0]),
                (self.img+'.image', False, [40, 71, 0, 0]), 
                (self.img+'.image', True, [10, 40, 0, 0]), 
                (self.img+'.image', False, [9, 40, 0, 0])])

        # .image report
        report2 = th.check_dict_vals(exp_im_stats, im_stats_dict, '.image', epsilon=self.epsilon)

        # .mask report
        mask_stats_dict = self.image_stats(image=self.img+'.mask')

        # test_standard_cube.exp_mask_stats
        exp_mask_stats = self._exp_dicts['exp_mask_stats']

        report3 = th.check_dict_vals(exp_mask_stats, mask_stats_dict, '.mask', epsilon=self.epsilon)

        # .pb report
        pb_stats_dict = self.image_stats(image=self.img+'.pb', fit_region = \
            'ellipse[[11.47659846deg, -73.25817055deg], [23.1086arcsec, 23.0957arcsec], 90.00000000deg]')

        # test_standard_cube.exp_mask_stats
        exp_pb_stats = self._exp_dicts['exp_pb_stats']

        report4 = th.check_dict_vals(exp_pb_stats, pb_stats_dict, '.pb', epsilon=self.epsilon)

        # .psf report
        psf_stats_dict = self.image_stats(image=self.img+'.psf', fit_region = \
            'ellipse[[11.47648725deg, -73.25812003deg], [8.0291arcsec, 6.8080arcsec], 90.00000000deg]')

        # test_standard_cube.exp_psf_stats
        exp_psf_stats = self._exp_dicts['exp_psf_stats']

        report5 = th.check_dict_vals(exp_psf_stats, psf_stats_dict, '.psf', epsilon=self.epsilon)

        # .residual report
        resid_stats_dict = self.image_stats(image=self.img+'.residual', fit_region = \
            'ellipse[[11.47881897deg, -73.25881015deg], [9.0414arcsec, 8.4854arcsec], 90.00000000deg]')

        # test_standard_cube.exp_resid_stats
        exp_resid_stats = self._exp_dicts['exp_resid_stats']

        report6 = th.check_dict_vals(exp_resid_stats, resid_stats_dict, \
            '.residual', epsilon=self.epsilon)

        # .model report
        model_stats_dict = self.image_stats(image=self.img+'.model', fit_region = \
            'ellipse[[11.47881897deg, -73.25881015deg], [9.0414arcsec, 8.4854arcsec], 90.00000000deg]', masks=mask_stats_dict['mask'])

        # test_standard_cube.exp_model_stats
        exp_model_stats = self._exp_dicts['exp_model_stats']

        report7 = th.check_dict_vals(exp_model_stats, model_stats_dict, \
            '.model', epsilon=self.epsilon)

        # .sumwt report
        sumwt_stats_dict = self.image_stats(image=self.img+'.sumwt')

        # test_standard_cube.exp_sumwt_stats
        exp_sumwt_stats = self._exp_dicts['exp_sumwt_stats']

        report8 = th.check_dict_vals(exp_sumwt_stats, sumwt_stats_dict, \
            '.sumwt', epsilon=self.epsilon)

        # report combination
        report = report0 + report1 + report2 + report3 + report4 + report5 + report6 + report7 + report8


        if self.parallel:
            # test_standard_cube.exp_bmin_dict
            exp_bmin_dict = self._exp_dicts['exp_bmin_dict']
            # test_standard_cube.exp_bmaj_dict
            exp_bmaj_dict = self._exp_dicts['exp_bmaj_dict']
            # test_standard_cube.exp_pa_dict
            exp_pa_dict = self._exp_dicts['exp_pa_dict']


            report += self.check_dict_vals_beam(exp_bmin_dict, bmin_dict, '.image bmin', epsilon=self.epsilon)
            report += self.check_dict_vals_beam(exp_bmaj_dict, bmaj_dict, '.image bmaj', epsilon=self.epsilon)
            report += self.check_dict_vals_beam(exp_pa_dict, pa_dict, '.image pa', epsilon=self.epsilon)

        failed=self.filter_report(report)
        add_to_dict(self, output = test_dict, dataset = \
            "E2E6.1.00034.S_tclean.ms")

        self.modify_dict(test_dict, self.test_name, self.parallel)

        test_dict[self.test_name]['report'] = report
        test_dict[self.test_name]['images'] = []

        self.img = shutil._basename(self.img)


        # This needs be add because regenerating the testing report fails if the files already exist.
        #
        # This should be replaced by something else, maybe by adding something similar into the immoments function
        # or adding date-time stamps.
        
        if os.path.isdir(os.getcwd() + '/' + self.img + '.image.moment8'):
            try:
                print('Removing moment8 file.')
                shutil.rmtree(os.getcwd() + '/' + self.img + '.image.moment8')
            except FileNotFoundError:
                print('Failure to remove file: ' + os.getcwd() + '/' + self.img + '.image.moment8')

        if os.path.isdir(os.getcwd() + '/' + self.img + '.residual.moment8'):
            try:
                shutil.rmtree(os.getcwd() + '/' + self.img + '.residual.moment8')
            except FileNotFoundError:
                print('Failure to remove file: ' + os.getcwd() + '/' + self.img + '.residual.moment8')

        immoments(imagename=self.img+'.image', moments = 8, outfile = self.img +'.image.moment8')
        plt_utils.plot_image(imname=self.img+'.image', type='.moment8', chan=0, trim=True)
        
        immoments(imagename=self.img+'.residual', moments = 8, outfile = self.img +'.residual.moment8')
        plt_utils.plot_image(imname=self.img+'.residual', type='.moment8', chan=0, trim=True)

        test_dict[self.test_name]['images'].extend( \
            (self.img+'.image.moment8.png',self.img+'.residual.moment8.png'))

        test_dict[self.test_name]['images'].append(self.img+'.image.profile.png')

        if savemetricdict:
            ### serialize ndarray in mask_stats_dcit
            mask_stats_mod_dict = copy.deepcopy(mask_stats_dict)
            mask_stats_mod_dict['mask'] = mask_stats_dict['mask'].tolist()
            #create a nested dictionary containing exp dictionaries to save
            savedict = {}
            #list of stats to save
            # im_stats, mask_stats, pb_stats, psf_stats,\
            # model_stats, resid_stats, sumwt_stats]
            savedict['im_stats_dict']=im_stats_dict
            savedict['mask_stats_dict']=mask_stats_mod_dict
            savedict['pb_stats_dict']=pb_stats_dict
            savedict['psf_stats_dict']=psf_stats_dict
            savedict['model_stats_dict']=model_stats_dict
            savedict['resid_stats_dict']=resid_stats_dict
            savedict['sumwt_stats_dict']=sumwt_stats_dict

            savedict['bmin_dict']=bmin_dict
            savedict['bmaj_dict']=bmaj_dict
            savedict['pa_dict']=pa_dict

            self.save_dict_to_file(self.test_name,savedict, self.test_name+'_cur_stats')

        self.assertTrue(th.check_final(pstr = report), msg = failed)
        self.test_dict = test_dict

        # In the case of running in a notebook the tearDown() doesn't get called so we call it manually.
        if self._testMethodName is "runTest":
            self.tearDown()

# End of test_standard_cube
#-------------------------------------------------#

def suite():
    return [Test_standard]

# Main #
if __name__ == '__main__':
    unittest.main()
