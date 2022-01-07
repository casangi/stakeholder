##########################################################################
#
# Run the tests as described in https://open-confluence.nrao.edu/pages/viewpage.action?spaceKey=CASA&title=Requirements+for+VLASS+Imaging+Pipeline+Stakeholders+Tests
# There wasn't a great definition of the work to be done, so much of it is being interpretted by me (BGB).
#
##########################################################################
#
# The basic idea:
#  Compare values from the most recent build of casa to other known values. Check for differences.
#
# Values to compare current versions against:
#  1. on-axis      ---   "ground truth", I think from the (fluxscale?) task
#  2. CASA 6.1.3   ---   pipeline approved version of casa
#
# What's good enough?
#  flux density: 5% goal, 10% ok
#  spectral index: 0.1 goal, 0.2 ok
# From https://drive.google.com/file/d/1zw6UeDEoXoxM05oFg3rir0hrCMEJMxkH/view and https://open-confluence.nrao.edu/display/VLASS/Updated+VLASS+survey+science+requirements+and+parameters
#
# Images:
#  J1302
#  J1927
#
# Impage processing methods:
#  mosaic        ---   "stokes I"
#  awproject     ---   "stokes I"
#  mosaic cube   ---   "cube"
#  mosaic QL     ---   "QL"
#
# Values to be compared:
#  Stokes I:
#   a. tt0:                                                    6.1.3, on-axis
#   b. tt1:                                                    6.1.3, on-axis
#   c. alpha:                                                  6.1.3, on-axis
#   e. Confirm presence of model column in resultant MS
#  Stokes I and Cube:
#   d. beamsize comparison:                                    6.1.3
#   f. Runtimes not significantly different relative to previous runs
#  Cube:
#   g. Fit F_nu0 and Alpha from three cube planes and compare: 6.1.3, on-axis
#   h. IQUV flux densities of all three spws:                  6.1.3
#   i. IQUV flux densities of all three spws:                  on-axis measurements
#   j. Beam of all three spws:                                 6.1.3
#  QL:
#   k. flux density of Calibrator source:                      6.1.3, on-axis
#  All:
#   l. Ensure intermediate products exist, pbcor images, RMS image (made by imdev), and cutouts (.subim) from imsubimage
# Note on (i): on-axis values derived from images of calibrator using standard gridder of pointed data on calibrator
#              ^-- does this mean these are also "ground truth" values? BGB211222
#
# Other goals:
#  AUX1: try to make these tests compatible with jupyter
#   https://github.com/casangi/stakeholder/
#   https://colab.research.google.com/drive/1IdnKmhNWonX1r1zJ5oCTExVMSMgTHjOm?usp=sharing#scrollTo=J88MWrnHFXZP
#   https://docs.google.com/document/d/1rKktZdg4IwV5Nr_giVRcFlvkFkZ2NxTv-BO6QFknDUI/edit#heading=h.46icuqlj57dy
#  AUX2: run each test as its own test script, eg "VLASS_mosaic_stakeholder_test_script.py"
#   not done (is this actually necessary? what is the goal of these scripts such that individual files are needed?)
#
##########################################################################
#
#  Tests
#
#J1302 Tests
#1. mosaic: Should match values for "Stokes I" in the "Values to be compared"
#vis:'J1302_12field.ms' gridder:'mosaic'
#testname: test_j1302_mosaic
#
#2. awproject: Should match values for "Stokes I" in the "Values to be compared"
#vis:'J1302_12field.ms' gridder:'awproject'
#testname: test_j1302_awproject
#
#3. mosaic cube: Should match values for "Cube" in the "Values to be compared"
#vis:'J1302_12field_cubedata.ms' gridder:'mosaic'
#testname: test_j1302_mosaic_cube
#
#4. QL: Should match values for "QL" in the "Values to be compared"
#vis:'J1302_12field.ms' gridder:'ql'
#testname: test_j1302_ql
#
#
#
#J1927 Tests
#5. mosaic: Should match values for "Stokes I" in the "Values to be compared"
#vis:'J1927_12field.ms', gridder:'mosaic'
#testname: test_j1927_mosaic
#
#6. awproject: Should match values for "Stokes I" in the "Values to be compared"
#vis:'J1927_12field.ms', gridder:'awproject'
#testname: test_j1927_awproject
#
#7. mosaic cube: Should match values for "Cube" in the "Values to be compared"
#vis:'J1927_12field_cubedata.ms', gridder:'mosaic'
#testname: test_j1927_mosaic_cube
#
#8. QL: Should match values for "QL" in the "Values to be compared"
#vis:'J1927_12field.ms', gridder:'ql'
#testname: test_j1927_ql
#
##########################################################################

import os
import unittest
import numpy as np
import shutil
from datetime import datetime

from casatasks import casalog, impbcor, imdev, imhead, imsubimage, imstat
from casatools import table
from casatestutils.imagerhelpers import TestHelpers

from baseclass.vlass_base_class import test_vlass_base

th = TestHelpers()
tb = table()

##############################################
##############################################
class test_j1302(test_vlass_base):
	
    def setUp(self):
        super().setUp()
        self.vis = 'J1302-12fields.ms'
        self.phasecenter = '13:03:13.874 -10.51.16.73'
        self._clean_imgs_exist_dict()

    def _run_tclean(self, vis='', selectdata=True, field='', spw='', timerange='', uvrange='', antenna='',
                    scan='', observation='', intent='', datacolumn='corrected', imagename='', imsize=[100],
                    cell=['1arcsec'], phasecenter='', stokes='I', projection='SIN', startmodel='',
                    specmode='mfs', reffreq='', nchan=- 1, start='', width='', outframe='LSRK',
                    veltype='radio', restfreq=[], interpolation='linear', perchanweightdensity=True,
                    gridder='standard', facets=1, psfphasecenter='', wprojplanes=1, vptable='',
                    mosweight=True, aterm=True, psterm=False, wbawp=True, conjbeams=False, cfcache='',
                    usepointing=False, computepastep=360.0, rotatepastep=360.0, pointingoffsetsigdev=[],
                    pblimit=0.2, normtype='flatnoise', deconvolver='hogbom', scales='', nterms=2,
                    smallscalebias=0.0, restoration=True, restoringbeam='', pbcor=False, outlierfile='',
                    weighting='natural', robust=0.5, noise='1.0Jy', npixels=0, uvtaper=[], niter=0,
                    gain=0.1, threshold=0.0, nsigma=0.0, cycleniter=- 1, cyclefactor=1.0, minpsffraction=0.05,
                    maxpsffraction=0.8, interactive=False, usemask='user', mask='', pbmask=0.0,
                    sidelobethreshold=3.0, noisethreshold=5.0, lownoisethreshold=1.5, negativethreshold=0.0,
                    smoothfactor=1.0, minbeamfrac=0.3, cutthreshold=0.01, growiterations=75, dogrowprune=True,
                    minpercentchange=- 1.0, verbose=False, fastnoise=True, restart=True, savemodel='none',
                    calcres=True, calcpsf=True, psfcutoff=0.35, parallel=False, compare_tclean_pars=None):
        """ Runs tclean with the default parameters from v6.4.0
        If the 'compare_tclean_pars' dict is provided, then compare these values to the other parameters of this function. """
        run_tclean_pars = locals()
        run_tclean_pars = {k:run_tclean_pars[k] for k in filter(lambda x: x not in ['self', 'compare_tclean_pars'] and '__' not in x, run_tclean_pars.keys())}
        if (compare_tclean_pars != None):
            self.print_task_diff_params('run_tclean', act_pars=run_tclean_pars, exp_pars=compare_tclean_pars)
        super().run_tclean(**run_tclean_pars)

    def replace_psf(old, new):
        """ Replaces [old] PSF image with [new] image. Clears parallel working directories."""
        for this_tt in ['tt0', 'tt1', 'tt2']:
            shutil.rmtree(self.imagename_base+old+'.psf.'+this_tt)
            shutil.copytree(self.imagename_base+new+'.psf.'+this_tt, self.imagename_base+old+'.psf.'+this_tt)

    def _clean_imgs_exist_dict(self):
        self.imgs_exist = { 'successes':[], 'reports':[] }

    def check_img_exists(self, img):
        """ Returns true if the image exists. A report is collected internally, to be returned as a group report in get_imgs_exist_results(...). """
        exists = th.image_exists(img)
        success, report = th.check_val(exists, True, valname=f"image_exists('{img1+toext}')", exact=True, testname=self._testMethodName)
        if not exists:
            # log immediately: missing images could cause the rest of the test to fail
            casalog.post(report, "SEVERE")
        self.imgs_exist['successes'].append(success)
        self.imgs_exist['reports'].append(report)
        return success

    def get_imgs_exist_results(self):
        """ Get a single collective result of check_img_exists(...) """
        success = all(self.imgs_exist['successes'])
        report = "".join(self.imgs_exist['reports'])
        return success, report

    def check_fracdiff(self, actual, expected, valname, desired_diff=0.05, max_diff=0.1):
        """ Logs a warning if outside of desired bounds, returns False if outside required bounds """
        # 5% desired, 10% required, as from https://drive.google.com/file/d/1zw6UeDEoXoxM05oFg3rir0hrCMEJMxkH/view and https://open-confluence.nrao.edu/display/VLASS/Updated+VLASS+survey+science+requirements+and+parameters
        fracdiff=abs(actual-expected)/abs(expected)
        val = max(fracdiff)
        if (val > desired_diff):
            casalog.post(f"Warning, {valname}: {fracdiff} vs desired {desired_diff}, (actual: {actual}, expected: {expected})", "WARN")
        out = (val <= max_diff)

        testname = self._testMethodName
        correctval = f"< {max_diff}"
        report = "[ {} ] {} is {} ( {} : should be {})\n".format(testname, valname, str(val), th.verdict(out), str(correctval) )
        report = report.rstrip() + f" (raw actual/expected values: {actual}/{expected})\n"
        return out, report

    def check_column_exists(self, colname):
        tb.open(self.vis)
        cnt = tb.colnames().count(colname)
        tb.done()
        tb.close()
        return th.check_val(cnt, 1, valname=f"count('{colname}')", exact=True, testname=self._testMethodName)

    # Test 1
    def test_j1302_mosaic(self):
        """ [j1302] test_j1302_mosaic """
        ######################################################################################
        # Should match values for "Stokes I" in the "Values to be compared"
        ######################################################################################
        img = self.imagename_base+'iter2'
        # self.prepData()
        # combine first and 2nd order masks
        # immath(imagename=['secondmask.mask','QLcatmask.mask'],expr='IM0+IM1',outfile='sum_of_masks.mask')
        # im.mask(image='sum_of_masks.mask',mask='combined.mask',threshold=0.5)

        # initialize iter2, no cleaning
        # self.run_tclean(imagename=img, datacolumn='corrected', gridder='mosaic', conjbeams=False, pblimit=0.1, nterms=2, cycleniter=500, parallel=True)
        # # resume iter2 with QL mask
        # self.run_tclean(imagename=img, datacolumn='corrected', gridder='mosaic', conjbeams=False, pblimit=0.1, scales=[0, 5, 12], nterms=2, niter=20000, nsigma=3.0, cycleniter=500, mask='QLcatmask.mask', calcres=False, calcpsf=False, parallel=True)
        # # save model column, doesn't happen here in acutal VLASS pipeline, but makes sure functionality works.
        # self.run_tclean(imagename=img, gridder='mosaic', conjbeams=False, pblimit=0.1, nterms=2, cycleniter=500, savemodel='modelcolumn', calcres=False, calcpsf=False, parallel=False)
        # # resume iter2 with combined mask, remove old mask first, pass new mask as parameter
        # os.system('rm -rf '+img+'.mask')
        # self.run_tclean(imagename=img, datacolumn='corrected', gridder='mosaic', conjbeams=False, pblimit=0.1, scales=[0, 5, 12], nterms=2, niter=20000, nsigma=3.0, cycleniter=500, mask='combined.mask', calcres=False, calcpsf=False, parallel=True)
        # # resume iter2 with pbmask, remove old mask first then specify pbmask in resumption of tclean
        # os.system('rm -rf '+img+'.mask')
        # self.run_tclean(imagename=img, datacolumn='corrected', gridder='mosaic', conjbeams=False, pblimit=0.1, scales=[0, 5, 12], nterms=2, niter=20000, nsigma=4.5, cycleniter=100, usemask='pb', pbmask=0.4, calcres=False, calcpsf=False, parallel=self.parallel)

        # TODO report=th.checkall(...)
        # TODO self.checkfinal(pstr=report)
        pass

    # Test 2
    def test_j1302_awproject(self):
        """ [j1302] test_j1302_awproject """
        ######################################################################################
        # Should match values for "Stokes I" in the "Values to be compared"
        ######################################################################################
        # TODO self.prepData(...)
        #previous steps in the pipeline would have created mask files from catalogs and images that were created as an 
        #intermediate pipeline step.
        # img0 = self.imagename_base+'iter0d'
        # img2 = self.imagename_base+'iter2'
        # self.prepData()

        # # combine first and 2nd order masks
        # immath(imagename=['secondmask.mask', 'QLcatmask.mask'],
        #            expr='IM0+IM1', outfile='sum_of_masks.mask')
        # im.mask(image='sum_of_masks.mask', mask='combined.mask', threshold=0.5)

        # # create robust psfs with wbawp=False using corrected column
        # self.run_tclean(imagename=img0, datacolumn='corrected', gridder='awproject', cfcache='', pblimit=0.02, nterms=2, calcres=False, parallel=self.parallel, pointingoffsetsigdev=[300, 30], wbawp=False)
        # # initialize iter2, no cleaning
        # self.run_tclean(imagename=img2, datacolumn='corrected', gridder='awproject', cfcache='', pblimit=0.02, nterms=2, parallel=self.parallel, pointingoffsetsigdev=[300, 30], wbawp=True)
        # #  replace iter2 psf with no_WBAP
        # replace_psf('iter2', 'iter0d')
        # #  resume iter2 with QL mask
        # self.run_tclean(imagename=img2, datacolumn='corrected', gridder='awproject', cfcache='', pblimit=0.02, scales=[0, 5, 12], nterms=2, niter=20000, nsigma=3.0, cycleniter=3000, mask='J1302_QLcatmask.mask', calcres=False, calcpsf=False, parallel=self.parallel, pointingoffsetsigdev=[300, 30], wbawp=True)
        # # save model column, doesn't happen here in acutal VLASS pipeline, but makes sure functionality works.
        # self.run_tclean(imagename=img2, gridder='awproject', cfcache='', pblimit=0.02, nterms=2, savemodel='modelcolumn', calcres=False, calcpsf=False, parallel=True, pointingoffsetsigdev=[300, 30], wbawp=True)
        # # resume iter2 with combined mask
        # os.system('rm -rf '+imagename_base+'iter2.mask')
        # self.run_tclean(imagename=img2, datacolumn='corrected', gridder='awproject', cfcache='', pblimit=0.02, scales=[0, 5, 12], nterms=2, niter=20000, nsigma=3.0, cycleniter=3000, mask='J1302_combined.mask', calcres=False, calcpsf=False, parallel=self.parallel, pointingoffsetsigdev=[300, 30], wbawp=True)
        # # resume iter2 with pbmask, removed old mask first then specify pbmask in resumption of tclean
        # os.system('rm -rf '+imagename_base+'iter2.mask')
        # self.run_tclean(imagename=img2, datacolumn='corrected', gridder='awproject', cfcache='', pblimit=0.02, scales=[0, 5, 12], nterms=2, niter=20000, nsigma=4.5, cycleniter=500, usemask='pb', calcres=False, calcpsf=False, parallel=self.parallel, pointingoffsetsigdev=[300, 30], wbawp=True, pbmask=0.4)

    # Test 3
    def test_j1302_mosaic_cube(self):
        """ [j1302] test_j1302_mosaic_cube """
        ######################################################################################
        # Should match values for "Cube" in the "Values to be compared"
        ######################################################################################
        # TODO self.prepData(...)
        #previous steps in the pipeline would have created mask files from catalogs and images that were created as an 
        #intermediate pipeline step.
        # self.prepData()

        #previous steps in the pipeline would have created mask files from catalogs and images that were created as an 
        #intermediate pipeline step.

        refFreqDict={'2' : '2.028GHz',
                 '3' :  '2.156GHz',
                 '4' :  '2.284GHz',
                 '5' :  '2.412GHz',
                 '6' :  '2.540GHz',
                 '7' :  '2.668GHz',
                 '8' :  '2.796GHz',
                 '9' :  '2.924GHz',
                 '10':  '3.052GHz',
                 '11':  '3.180GHz',
                 '12':  '3.308GHz',
                 '13':  '3.436GHz',
                 '14':  '3.564GHz',
                 '15':  '3.692GHz',
                 '16':  '3.820GHz',
                 '17':  '3.948GHz'
                }

        spws=['2']#,'8','14']
        stokesParams=['IQUV']
        spwstats={'2':  {'freq': 2.028, 'IQUV': np.array([0.0,0.0,0.0,0.0]), 'beam': np.array([0.0,0.0,0.0])},
                  '8':  {'freq': 2.796, 'IQUV': np.array([0.0,0.0,0.0,0.0]), 'beam': np.array([0.0,0.0,0.0])},
                  '14': {'freq': 3.594, 'IQUV': np.array([0.0,0.0,0.0,0.0]), 'beam': np.array([0.0,0.0,0.0])}
                 }
        for spw in spws:
            for stokes in stokesParams:
                # initialize iter2, no cleaning
                img = self.imagename_base+'iter2_'+spw.replace('~','-')+'_'+stokes
                # run_tclean(spw='2', datacolumn='corrected', imagename=img, reffreq='2.028GHz', gridder='mosaic', conjbeams=False, pblimit=0.1, nterms=1, cycleniter=500, stokes='IQUV', parallel=False)

                # resume iter2 with QL mask
                # run_tclean(spw='2', datacolumn='corrected', imagename=img, reffreq='2.028GHz', gridder='mosaic', conjbeams=False, pblimit=0.1, scales=[0, 5, 12], nterms=1, niter=20000, nsigma=3.0, cycleniter=500, mask='QLcatmask.mask', calcres=False, calcpsf=False, stokes='IQUV', parallel=False)

                # # resume iter2 with combined mask
                # os.system('rm -rf *.workdirectory')
                # os.system('rm -rf *iter2*.mask')
                # run_tclean(spw='2', datacolumn='corrected', imagename=img, reffreq='2.028GHz', gridder='mosaic', conjbeams=False, pblimit=0.1, scales=[0, 5, 12], nterms=1, niter=20000, nsigma=3.0, cycleniter=500, mask='combined.mask', calcres=False, calcpsf=False, stokes='IQUV', parallel=False)

                # os.system('rm -rf iter2*.mask')
                # run_tclean(spw='2', datacolumn='corrected', imagename=img, reffreq='2.028GHz', gridder='mosaic', conjbeams=False, pblimit=0.1, scales=[0, 5, 12], nterms=1, niter=20000, nsigma=4.5, cycleniter=100, usemask='pb', pbmask=0.4, calcres=False, calcpsf=False, stokes='IQUV', parallel=False)

                # tt0statsI=imstat(imagename=img+'.image.tt0',box='2000,2000,2000,2000',stokes='I')
                # tt0statsQ=imstat(imagename=img+'.image.tt0',box='2000,2000,2000,2000',stokes='Q')
                # tt0statsU=imstat(imagename=img+'.image.tt0',box='2000,2000,2000,2000',stokes='U')
                # tt0statsV=imstat(imagename=img+'.image.tt0',box='2000,2000,2000,2000',stokes='V')
                # spwstats[spw]['IQUV']=np.array([tt0statsI['max'],tt0statsQ['max'],tt0statsU['max'],tt0statsV['max']])
                # rmstt0statsI=imstat(imagename=img+'.residual.tt0',stokes='I')
                # rmstt0statsQ=imstat(imagename=img+'.residual.tt0',stokes='Q')
                # rmstt0statsU=imstat(imagename=img+'.residual.tt0',stokes='U')
                # rmstt0statsV=imstat(imagename=img+'.residual.tt0',stokes='V')
                # spwstats[spw]['IQUV']=np.array([tt0statsI['max'],tt0statsQ['max'],tt0statsU['max'],tt0statsV['max']])
                # spwstats[spw]['rmsIQUV']=np.array([rmstt0statsI['rms'],rmstt0statsQ['rms'],rmstt0statsU['rms'],rmstt0statsV['rms']])
                # spwstats[spw]['SNR']=((spwstats[spw]['IQUV']/spwstats[spw]['rmsIQUV'])**2)**0.5

                # header=imhead(img+'_.psf.tt0')
                # beamstats=np.array([header['perplanebeams']['beams']['*0']['*0']['major']['value'],header['perplanebeams']['beams']['*0']['*0']['minor']['value'],header['perplanebeams']['beams']['*0']['*0']['positionangle']['value']])
                # spwstats[spw]['beam']=beamstats

    # Test 4
    def test_j1302_ql(self):
        """ [j1302] test_j1302_ql """
        ######################################################################################
        # Should match values for "QL" in the "Values to be compared"
        ######################################################################################
        # not part of the jupyter scripts
        tstobj = self # jupyter equivalent: "tstobj = test_j1302()"

        ###########################################
        # %% Set local vars [test_j1302_ql] start @
        ###########################################

        #previous steps in the pipeline would have created mask files from catalogs and images that were created as an 
        #intermediate pipeline step.
        tstobj.data_path_dir  = 'J1302/Stakeholder-test-mosaic-data'
        img0 = 'VLASS1.2.ql.T08t20.J1302.10.2048.v1.I.iter0'
        img1 = 'VLASS1.2.ql.T08t20.J1302.10.2048.v1.I.iter1'
        tstobj.prepData()
        rundir = "/users/bbean/dev/CAS-12427/src/casalith/build-casalith/work/linux/test_vlass_j1302_QL_unittest"
        # os.system(f"mv {rundir}/run_results/VLASS* {rundir}/nosedir/test_vlass_1v2/")

        ###########################################
        # %% Set local vars [test_j1302_ql] start @
        ###########################################

        #########################################
        # %% Set local vars [test_j1302_ql] end @
        # .......................................

        # not part of the jupyter scripts
        starttime = datetime.now()

        # .......................................
        # %% Run tclean [test_j1302_ql] start   @
        #########################################
        pass

        def run_tclean(vis=tstobj.vis, field='',spw='', antenna='', scan='', stokes='I', intent='OBSERVE_TARGET#UNSPECIFIED', datacolumn='data',
                       imagename=None, niter=None, restoration=None, compare_tclean_pars=None,
                       phasecenter=tstobj.phasecenter, reffreq='3.0GHz', deconvolver='mtmfs',
                       cell='1.0arcsec', imsize=7290, gridder='mosaic', restoringbeam='common',
                       specmode='mfs', nchan=-1, outframe='LSRK', perchanweightdensity=False, 
                       wprojplanes=1, mosweight=False, conjbeams=False, usepointing=False,
                       rotatepastep=360.0, pblimit=0.2, scales=[0], nterms=2, pbcor=False,
                       weighting='briggs', robust=1.0, npixels=0, threshold=0.0, nsigma=0,
                       cycleniter=-1, cyclefactor=1.0, interactive=0, fastnoise=True,
                       calcres=True, calcpsf=True, savemodel='none', parallel=False):
            params = locals()
            params = {k:params[k] for k in filter(lambda x: x not in ['tstobj'], params.keys())}
            tstobj._run_tclean(**params)

        script_pars_vals_0 = None#tstobj.get_params_as_dict(vis='J1302-12fields.ms', selectdata=True, field='', spw='', timerange='', uvrange='', antenna='', scan='', observation='', intent='OBSERVE_TARGET#UNSPECIFIED', datacolumn='data', imagename='VLASS1.2.ql.T08t20.J1302.10.2048.v1.I.iter0', imsize=[7290, 7290], cell='1.0arcsec', phasecenter='13:03:13.874 -10.51.16.73', stokes='I', projection='SIN', startmodel='', specmode='mfs', reffreq='3.0GHz', nchan=-1, start='', width='', outframe='LSRK', veltype='radio', restfreq=[], interpolation='linear', perchanweightdensity=False, gridder='mosaic', facets=1, psfphasecenter='', chanchunks=1, wprojplanes=1, vptable='', mosweight=False, aterm=True, psterm=False, wbawp=True, conjbeams=False, cfcache='', usepointing=False, computepastep=360.0, rotatepastep=360.0, pointingoffsetsigdev=[], pblimit=0.2, normtype='flatnoise', deconvolver='mtmfs', scales=[0], nterms=2, smallscalebias=0.0, restoration=False, restoringbeam='common', pbcor=False, outlierfile='', weighting='briggs', robust=1.0, noise='1.0Jy', npixels=0, uvtaper=[], niter=0, gain=0.1, threshold='0.0mJy', nsigma=0.0, cycleniter=-1, cyclefactor=1.0, minpsffraction=0.05, maxpsffraction=0.8, interactive=0, usemask='user', mask='', pbmask=0.0, sidelobethreshold=3.0, noisethreshold=5.0, lownoisethreshold=1.5, negativethreshold=0.0, smoothfactor=1.0, minbeamfrac=0.3, cutthreshold=0.01, growiterations=75, dogrowprune=True, minpercentchange=-1.0, verbose=False, fastnoise=True, restart=True, savemodel='none', calcres=True, calcpsf=True, parallel=False)
        script_pars_vals_1 = None#tstobj.get_params_as_dict(vis='J1302-12fields.ms', selectdata=True, field='', spw='', timerange='', uvrange='', antenna='', scan='', observation='', intent='OBSERVE_TARGET#UNSPECIFIED', datacolumn='data', imagename='VLASS1.2.ql.T08t20.J1302.10.2048.v1.I.iter1', imsize=[7290, 7290], cell='1.0arcsec', phasecenter='13:03:13.874 -10.51.16.73', stokes='I', projection='SIN', startmodel='', specmode='mfs', reffreq='3.0GHz', nchan=-1, start='', width='', outframe='LSRK', veltype='radio', restfreq=[], interpolation='linear', perchanweightdensity=False, gridder='mosaic', facets=1, psfphasecenter='', chanchunks=1, wprojplanes=1, vptable='', mosweight=False, aterm=True, psterm=False, wbawp=True, conjbeams=False, cfcache='', usepointing=False, computepastep=360.0, rotatepastep=360.0, pointingoffsetsigdev=[], pblimit=0.2, normtype='flatnoise', deconvolver='mtmfs', scales=[0], nterms=2, smallscalebias=0.0, restoration=True, restoringbeam='common', pbcor=False, outlierfile='', weighting='briggs', robust=1.0, noise='1.0Jy', npixels=0, uvtaper=[], niter=20000, gain=0.1, threshold=0.0, nsigma=4.5, cycleniter=500, cyclefactor=2.0, minpsffraction=0.05, maxpsffraction=0.8, interactive=0, usemask='user', mask='', pbmask=0.0, sidelobethreshold=3.0, noisethreshold=5.0, lownoisethreshold=1.5, negativethreshold=0.0, smoothfactor=1.0, minbeamfrac=0.3, cutthreshold=0.01, growiterations=75, dogrowprune=True, minpercentchange=-1.0, verbose=False, fastnoise=True, restart=True, savemodel='none', calcres=False, calcpsf=False, parallel=False)
        run_tclean(imagename=img0, niter=0,     restoration=False, compare_tclean_pars=script_pars_vals_0)
        for ext in ['.weight.tt2', '.weight.tt0', '.psf.tt0', '.residual.tt0', '.weight.tt1', '.sumwt.tt2', '.psf.tt1', '.residual.tt1', '.psf.tt2', '.sumwt.tt1', '.model.tt0', '.pb.tt0', '.model.tt1', '.sumwt.tt0']:
            shutil.copytree(src=img0+ext, dst=img1+ext)
        run_tclean(imagename=img1, niter=20000, restoration=True, nsigma=4.5, cycleniter=500, cyclefactor=2.0, calcres=False, calcpsf=False, compare_tclean_pars=script_pars_vals_1)

        ###########################################
        # %% Run tclean [test_j1302_ql] end       @
        # %% Prepare Images [test_j1302_ql] start @
        ###########################################
        # hello
        pass

        # hifv_pbcor(pipelinemode="automatic")
        for fromext,toext in [('.image.tt0','.image.pbcor.tt0'), ('.residual.tt0','.image.residual.pbcor.tt0')]:
            impbcor(imagename=img1+fromext, pbimage=img1+'.pb.tt0', outfile=img1+toext, mode='divide', cutoff=-1.0, stretch=False)
            tstobj.check_img_exists(img1+toext)

        # hif_makermsimages(pipelinemode="automatic")
        imdev(imagename=img1+'.image.pbcor.tt0',
              outfile=img1+'.image.pbcor.tt0.rms',
              overwrite=True, stretch=False, grid=[10, 10], anchor='ref',
              xlength='60arcsec', ylength='60arcsec', interp='cubic', stattype='xmadm',
              statalg='chauvenet', zscore=-1, maxiter=-1)
        tstobj.check_img_exists(img1+'.image.pbcor.tt0.rms')

        # hif_makecutoutimages(pipelinemode="automatic")
        for ext in ['.image.tt0', '.residual.tt0', '.image.pbcor.tt0', '.image.pbcor.tt0.rms', '.psf.tt0', '.image.residual.pbcor.tt0', '.pb.tt0']:
            imhead(imagename=img1+ext)
            imsubimage(imagename=img1+ext, outfile=img1+ext+'.subim', box='1785.0,1785.0,5506.0,5506.0')
            tstobj.check_img_exists(img1+ext+'.subim')

        ####################################################
        # %% Prepare Images [test_j1302_ql] end            @
        # %% Compare Expected Values [test_j1302_ql] start @
        ####################################################

        # (l) Ensure intermediate products exist, pbcor images, RMS image (made by imdev), and cutouts (.subim) from imsubimage
        success0, report0 = tstobj.get_imgs_exist_results()

        # (a) tt0 vs 6.1.3, on-axis
        imstat_vals       = imstat(imagename=img1+'.image.pbcor.tt0.subim',box='1860,1860,1860,1860')
        curr_stats        = np.squeeze(np.array([imstat_vals['max']]))
        onaxis_stats      = np.array([0.9509])
        success1, report1 = tstobj.check_fracdiff(curr_stats, onaxis_stats, valname="Frac Diff F_nu vs. on-axis")
        casa613_stats     = np.array([0.888])
        success2, report2 = tstobj.check_fracdiff(curr_stats, casa613_stats, valname="Frac Diff F_nu vs. 6.1.3 image")

        # (b) tt1 vs 6.1.3, on-axis
        # no tt1 images for this test, skip

        # (c) alpha images
        # TODO
        # success3, report3 = ...

        # (d) beamsize comparison vs 6.1.3
        restbeam          = imhead(img1+'.image.pbcor.tt0.subim')['restoringbeam']
        beamstats_curr    = np.array([restbeam['major']['value'], restbeam['minor']['value'], restbeam['positionangle']['value']])
        beamstats_613     = np.array([3.1565470695495605, 2.58677792549133, 11.282347679138184])
        success4, report4 = tstobj.check_fracdiff(beamstats_curr, beamstats_613, valname="Frac Diff Maj, Min, PA vs 6.1.3")

        # (e) Confirm presence of model column in resultant MS
        success5, report5 = tstobj.check_column_exists("MODEL_DATA")

        report  = "".join([report0, report1, report2, report4, report5])
        success = success1 and success2 and success4 and success5 and th.check_final(report)
        casalog.post(report, "INFO")

        ##################################################
        # %% Compare Expected Valued [test_j1302_ql] end @
        ##################################################
        # not part of the jupyter scripts

        # (f) Runtimes not significantly different relative to previous runs
        # don't test this in jupyter notebooks - runtimes differ too much between machines
        endtime           = datetime.now()
        runtime           = (endtime-starttime).total_seconds()
        runtime613        = 1
        successt, reportt = th.check_val(runtime, runtime613, valname="6.1.3 runtime", exact=False, epsilon=0.1, testname=tstobj._testMethodName)

        report += reportt
        success = success and successt and th.check_final(report)
        if not success: # easier to read this way than in an assert statement
            casalog.post(report, "SEVERE")
        tstobj.assertTrue(success, msg=report)


##############################################
##############################################
# class test_j1927(test_vlass_base):
#     # Test 5
#     def test_j1927_mosaic(self):
#         # TODO
#         pass

##############################################
##############################################

## List to be run
def suite():
    return [test_j1302]#, test_j1927]

if __name__ == '__main__':
    unittest.main()