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
#   d. beamsize comparison:                                    6.1.3
#   e. Confirm presence of model column in resultant MS
#  Stokes I and Cube:
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

from casatasks import casalog, impbcor, imdev, imhead, imsubimage, imstat, immath
from casatools import table, imager
from casatasks.private.parallel.parallel_task_helper import ParallelTaskHelper
from casatestutils.imagerhelpers import TestHelpers

from baseclass.vlass_base_class import test_vlass_base

th = TestHelpers()
tb = table()
im = imager()

##############################################
##############################################
class test_j1302(test_vlass_base):

    def setUp(self):
        super().setUp()
        self.vis = 'J1302-12fields.ms'
        self.phasecenter = '13:03:13.874 -10.51.16.73'
        self._clean_imgs_exist_dict()
        self.parallel = self._get_enable_parallel()

    # Test 1
    def test_j1302_mosaic_noncube(self):
        """ [j1302] test_j1302_mosaic_noncube """
        ######################################################################################
        # Should match values for "Stokes I" in the "Values to be compared"
        ######################################################################################
        # "noncube" to allow give this test a unique prefix, for running with runtest
        # For example: runtest.py -v test_vlass_1v2.py[test_j1302_mosaic_noncube]

        # not part of the jupyter scripts
        tstobj = self # jupyter equivalent: "tstobj = test_j1302()"

        #######################################################
        # %% Set local vars [test_j1302_mosaic_noncube] start @
        #######################################################

        #previous steps in the pipeline would have created mask files from catalogs and images that were created as an
        #intermediate pipeline step.
        tstobj.data_path_dir  = 'J1302/Stakeholder-test-mosaic-data'
        img0 = 'J1302_iter2'
        tstobj.prepData('secondmask.mask','QLcatmask.mask')
        rundir = "/users/bbean/dev/CAS-12427/src/casalith/build-casalith/work/linux/test_vlass_j1302_mosaic_noncube_unittest"
        # os.system(f"mv {rundir}/run_results/VLASS* {rundir}/nosedir/test_vlass_1v2/")
        # os.system(f"mv {rundir}/*.ms {rundir}/nosedir/test_vlass_1v2/")
        os.system(f"mv {rundir}/*.mask {rundir}/nosedir/test_vlass_1v2/")

        #####################################################
        # %% Set local vars [test_j1302_mosaic_noncube] end @
        # .......................................

        # not part of the jupyter scripts
        starttime = datetime.now()

        # .......................................
        # %% Prepare masks [test_j1302_mosaic_noncube] start @
        ######################################################

        # combine first and 2nd order masks
        immath(imagename=['secondmask.mask','QLcatmask.mask'],expr='IM0+IM1',outfile='sum_of_masks.mask')
        im.mask(image='sum_of_masks.mask',mask='combined.mask',threshold=0.5)

        ####################################################
        # %% Prepare masks [test_j1302_mosaic_noncube] end @
        # %% Run tclean [test_j1302_mosaic_noncube] start  @
        ####################################################

        def run_tclean(vis=tstobj.vis, intent='OBSERVE_TARGET#UNSPECIFIED', uvrange='<12km',
                       niter=None, compare_tclean_pars=None, datacolumn=None,
                       imagename=img0, phasecenter=tstobj.phasecenter, reffreq='3.0GHz',
                       deconvolver='mtmfs', cell='0.6arcsec', imsize=4000, gridder='mosaic',
                       uvtaper=[''], restoringbeam=[], mosweight=False, rotatepastep=5.0,
                       smallscalebias=0.0, pblimit=0.1, scales=[0], weighting='briggs',
                       robust=1.0, nsigma=2.0, cycleniter=500, cyclefactor=3.0,
                       interactive=0, calcres=False, calcpsf=False):
            params = locals()
            params = {k:params[k] for k in filter(lambda x: x not in ['tstobj'], params.keys())}
            tstobj.run_tclean(**params)

        script_pars_vals_0 = None#tstobj.get_params_as_dict(vis='J1302-12fields.ms', selectdata=True, field='', spw='', timerange='', uvrange='<12km', antenna='', scan='', observation='', intent='OBSERVE_TARGET#UNSPECIFIED', datacolumn='corrected', imagename='J1302_iter2', imsize=4000, cell='0.6arcsec', phasecenter='13:03:13.874 -10.51.16.73', stokes='I', projection='SIN', startmodel='', specmode='mfs', reffreq='3.0GHz', nchan=-1, start='', width='', outframe='LSRK', veltype='radio', restfreq=[], interpolation='linear', perchanweightdensity=True, gridder='mosaic', facets=1, psfphasecenter='', chanchunks=1, wprojplanes=1, vptable='', mosweight=False, aterm=True, psterm=False, wbawp=True, conjbeams=False, cfcache='', usepointing=False, computepastep=360.0, rotatepastep=5.0, pointingoffsetsigdev=[], pblimit=0.1, normtype='flatnoise', deconvolver='mtmfs', scales=[0], nterms=2, smallscalebias=0.4, restoration=True, restoringbeam=[], pbcor=False, outlierfile='', weighting='briggs', robust=1.0, noise='1.0Jy', npixels=0, uvtaper=[''], niter=0, gain=0.1, threshold=0.0, nsigma=2.0, cycleniter=500, cyclefactor=3.0, minpsffraction=0.05, maxpsffraction=0.8, interactive=False, usemask='user', mask='', pbmask=0.0, sidelobethreshold=3.0, noisethreshold=5.0, lownoisethreshold=1.5, negativethreshold=0.0, smoothfactor=1.0, minbeamfrac=0.3, cutthreshold=0.01, growiterations=75, dogrowprune=True, minpercentchange=-1.0, verbose=False, fastnoise=True, restart=True, savemodel='none', calcres=True, calcpsf=True, parallel=False)
        script_pars_vals_1 = None#tstobj.get_params_as_dict(vis='J1302-12fields.ms', selectdata=True, field='', spw='', timerange='', uvrange='<12km', antenna='', scan='', observation='', intent='OBSERVE_TARGET#UNSPECIFIED', datacolumn='corrected', imagename='J1302_iter2', imsize=4000, cell='0.6arcsec', phasecenter='13:03:13.874 -10.51.16.73', stokes='I', projection='SIN', startmodel='', specmode='mfs', reffreq='3.0GHz', nchan=-1, start='', width='', outframe='LSRK', veltype='radio', restfreq=[], interpolation='linear', perchanweightdensity=True, gridder='mosaic', facets=1, psfphasecenter='', chanchunks=1, wprojplanes=1, vptable='', mosweight=False, aterm=True, psterm=False, wbawp=True, conjbeams=False, cfcache='', usepointing=False, computepastep=360.0, rotatepastep=5.0, pointingoffsetsigdev=[], pblimit=0.1, normtype='flatnoise', deconvolver='mtmfs', scales=[0, 5, 12], nterms=2, smallscalebias=0.4, restoration=True, restoringbeam=[], pbcor=False, outlierfile='', weighting='briggs', robust=1.0, noise='1.0Jy', npixels=0, uvtaper=[''], niter=20000, gain=0.1, threshold=0.0, nsigma=3.0, cycleniter=500, cyclefactor=3.0, minpsffraction=0.05, maxpsffraction=0.8, interactive=False, usemask='user', mask='QLcatmask.mask', pbmask=0.0, sidelobethreshold=3.0, noisethreshold=5.0, lownoisethreshold=1.5, negativethreshold=0.0, smoothfactor=1.0, minbeamfrac=0.3, cutthreshold=0.01, growiterations=75, dogrowprune=True, minpercentchange=-1.0, verbose=False, fastnoise=True, restart=True, savemodel='none', calcres=False, calcpsf=False, parallel=False)
        script_pars_vals_2 = None#tstobj.get_params_as_dict(vis='J1302-12fields.ms', selectdata=True, field='', spw='', timerange='', uvrange='<12km', antenna='', scan='', observation='', intent='OBSERVE_TARGET#UNSPECIFIED', datacolumn='data', imagename='J1302_iter2', imsize=4000, cell='0.6arcsec', phasecenter='13:03:13.874 -10.51.16.73', stokes='I', projection='SIN', startmodel='', specmode='mfs', reffreq='3.0GHz', nchan=-1, start='', width='', outframe='LSRK', veltype='radio', restfreq=[], interpolation='linear', perchanweightdensity=True, gridder='mosaic', facets=1, psfphasecenter='', chanchunks=1, wprojplanes=1, vptable='', mosweight=False, aterm=True, psterm=False, wbawp=True, conjbeams=False, cfcache='', usepointing=False, computepastep=360.0, rotatepastep=5.0, pointingoffsetsigdev=[], pblimit=0.1, normtype='flatnoise', deconvolver='mtmfs', scales=[0], nterms=2, smallscalebias=0.4, restoration=True, restoringbeam=[], pbcor=False, outlierfile='', weighting='briggs', robust=1.0, noise='1.0Jy', npixels=0, uvtaper=[''], niter=0, gain=0.1, threshold=0.0, nsigma=2.0, cycleniter=500, cyclefactor=3.0, minpsffraction=0.05, maxpsffraction=0.8, interactive=False, usemask='user', mask='', pbmask=0.0, sidelobethreshold=3.0, noisethreshold=5.0, lownoisethreshold=1.5, negativethreshold=0.0, smoothfactor=1.0, minbeamfrac=0.3, cutthreshold=0.01, growiterations=75, dogrowprune=True, minpercentchange=-1.0, verbose=False, fastnoise=True, restart=True, savemodel='modelcolumn', calcres=False, calcpsf=False, parallel=False)
        script_pars_vals_3 = None#tstobj.get_params_as_dict(vis='J1302-12fields.ms', selectdata=True, field='', spw='', timerange='', uvrange='<12km', antenna='', scan='', observation='', intent='OBSERVE_TARGET#UNSPECIFIED', datacolumn='corrected', imagename='J1302_iter2', imsize=4000, cell='0.6arcsec', phasecenter='13:03:13.874 -10.51.16.73', stokes='I', projection='SIN', startmodel='', specmode='mfs', reffreq='3.0GHz', nchan=-1, start='', width='', outframe='LSRK', veltype='radio', restfreq=[], interpolation='linear', perchanweightdensity=True, gridder='mosaic', facets=1, psfphasecenter='', chanchunks=1, wprojplanes=1, vptable='', mosweight=False, aterm=True, psterm=False, wbawp=True, conjbeams=False, cfcache='', usepointing=False, computepastep=360.0, rotatepastep=5.0, pointingoffsetsigdev=[], pblimit=0.1, normtype='flatnoise', deconvolver='mtmfs', scales=[0, 5, 12], nterms=2, smallscalebias=0.4, restoration=True, restoringbeam=[], pbcor=False, outlierfile='', weighting='briggs', robust=1.0, noise='1.0Jy', npixels=0, uvtaper=[''], niter=20000, gain=0.1, threshold=0.0, nsigma=3.0, cycleniter=500, cyclefactor=3.0, minpsffraction=0.05, maxpsffraction=0.8, interactive=False, usemask='user', mask='combined.mask', pbmask=0.0, sidelobethreshold=3.0, noisethreshold=5.0, lownoisethreshold=1.5, negativethreshold=0.0, smoothfactor=1.0, minbeamfrac=0.3, cutthreshold=0.01, growiterations=75, dogrowprune=True, minpercentchange=-1.0, verbose=False, fastnoise=True, restart=True, savemodel='none', calcres=False, calcpsf=False, parallel=False)
        script_pars_vals_4 = None#tstobj.get_params_as_dict(vis='J1302-12fields.ms', selectdata=True, field='', spw='', timerange='', uvrange='<12km', antenna='', scan='', observation='', intent='OBSERVE_TARGET#UNSPECIFIED', datacolumn='corrected', imagename='J1302_iter2', imsize=4000, cell='0.6arcsec', phasecenter='13:03:13.874 -10.51.16.73', stokes='I', projection='SIN', startmodel='', specmode='mfs', reffreq='3.0GHz', nchan=-1, start='', width='', outframe='LSRK', veltype='radio', restfreq=[], interpolation='linear', perchanweightdensity=True, gridder='mosaic', facets=1, psfphasecenter='', chanchunks=1, wprojplanes=1, vptable='', mosweight=False, aterm=True, psterm=False, wbawp=True, conjbeams=False, cfcache='', usepointing=False, computepastep=360.0, rotatepastep=5.0, pointingoffsetsigdev=[], pblimit=0.1, normtype='flatnoise', deconvolver='mtmfs', scales=[0, 5, 12], nterms=2, smallscalebias=0.4, restoration=True, restoringbeam=[], pbcor=False, outlierfile='', weighting='briggs', robust=1.0, noise='1.0Jy', npixels=0, uvtaper=[''], niter=20000, gain=0.1, threshold=0.0, nsigma=4.5, cycleniter=100, cyclefactor=3.0, minpsffraction=0.05, maxpsffraction=0.8, interactive=False, usemask='pb', mask='', pbmask=0.4, sidelobethreshold=3.0, noisethreshold=5.0, lownoisethreshold=1.5, negativethreshold=0.0, smoothfactor=1.0, minbeamfrac=0.3, cutthreshold=0.01, growiterations=75, dogrowprune=True, minpercentchange=-1.0, verbose=False, fastnoise=True, restart=True, savemodel='none', calcres=False, calcpsf=False, parallel=False)

        # initialize iter2, no cleaning
        run_tclean( niter=0,     datacolumn='corrected', calcres=True, calcpsf=True,                         compare_tclean_pars=script_pars_vals_0 )

        # # resume iter2 with QL mask
        run_tclean( niter=20000, datacolumn='corrected', mask="QLcatmask.mask", nsigma=3.0, scales=[0,5,12], compare_tclean_pars=script_pars_vals_1 )

        # save model column, doesn't happen here in acutal VLASS pipeline, but makes sure functionality works.
        run_tclean( niter=0,     datacolumn='data',      savemodel='modelcolumn',                            compare_tclean_pars=script_pars_vals_2 )

        # resume iter2 with combined mask, remove old mask first, pass new mask as parameter
        os.system(f"rm -rf {img0}.mask")
        run_tclean( niter=20000, datacolumn='corrected', mask="combined.mask",  nsigma=3.0, scales=[0,5,12], compare_tclean_pars=script_pars_vals_3 )

        # resume iter2 with pbmask, removed old mask first then specify pbmask in resumption of tclean
        os.system(f"rm -rf {img0}.mask")
        run_tclean( niter=20000, datacolumn='corrected', usemask='pb', mask="", pbmask=0.4,   nsigma=4.5, scales=[0,5,12], cycleniter=100, compare_tclean_pars=script_pars_vals_4 )

        ################################################################
        # %% Run tclean [test_j1302_mosaic_noncube] end                @
        # %% Compare Expected Values [test_j1302_mosaic_noncube] start @
        ################################################################

        # (l) Ensure intermediate products exist, pbcor images, RMS image (made by imdev), and cutouts (.subim) from imsubimage
        # N/A for this test

        tt0stats   = imstat(imagename=img0+'.image.tt0', box='2000,2000,2000,2000')
        tt1stats   = imstat(imagename=img0+'.image.tt1', box='2000,2000,2000,2000')
        alphastats = imstat(imagename=img0+'.alpha', box='2000,2000,2000,2000')
        currentstats  = np.squeeze(np.array([ tt0stats['max'], tt1stats['max'], alphastats['max']]))
        onaxis_stats  = np.array([            0.3337,          -0.01588,        -0.0476])
        casa613_stats = np.array([            0.3198292,       0.01994022,      0.06234646])

        # (a) tt0 vs 6.1.3, on-axis
        success0, report0 = tstobj.check_fracdiff(curr_stats[0], onaxis_stats[0],  valname="Frac Diff tt0 vs. on-axis")
        success1, report1 = tstobj.check_fracdiff(curr_stats[0], casa613_stats[0], valname="Frac Diff tt0 vs. 6.1.3 image")

        # (b) tt1 vs 6.1.3, on-axis
        success2, report2 = tstobj.check_fracdiff(curr_stats[1], onaxis_stats[1],  valname="Frac Diff tt1 vs. on-axis")
        success3, report3 = tstobj.check_fracdiff(curr_stats[1], casa613_stats[1], valname="Frac Diff tt1 vs. 6.1.3 image")

        # (c) alpha images
        success4, report4 = tstobj.check_fracdiff(curr_stats[2], onaxis_stats[2],  valname="Frac Diff alpha vs. on-axis")
        success5, report5 = tstobj.check_fracdiff(curr_stats[2], casa613_stats[2], valname="Frac Diff alpha vs. 6.1.3 image")

        # (d) beamsize comparison vs 6.1.3
        restbeam          = imhead(img0+'.image.tt0')['restoringbeam']
        beamstats_curr    = np.array([restbeam['major']['value'], restbeam['minor']['value'], restbeam['positionangle']['value']])
        beamstats_613     = np.array([3.1565470695495605,         2.58677792549133,           11.282347679138184])
        success6, report6 = tstobj.check_fracdiff(beamstats_curr, beamstats_613, valname="Frac Diff Maj, Min, PA vs 6.1.3")

        # (e) Confirm presence of model column in resultant MS
        success7, report7 = tstobj.check_column_exists("MODEL_DATA")

        report  = "".join([report0, report1, report2, report3, report4, report5, report6, report7])
        success = success0 and success1 and success2 and success3 and success4 and success5 and success6 and success7 and th.check_final(report)
        casalog.post(f"{report}\nSuccess: {success}", "INFO")

        ##############################################################
        # %% Compare Expected Values [test_j1302_mosaic_noncube] end @
        ##############################################################
        # not part of the jupyter scripts

        # (f) Runtimes not significantly different relative to previous runs
        # don't test this in jupyter notebooks - runtimes differ too much between machines
        success, report = tstobj.check_runtime(starttime, 1543, success, report)

        tstobj.assertTrue(success, msg=report)

    # Test 2
    def test_j1302_awproject(self):
        """ [j1302] test_j1302_awproject """
        ######################################################################################
        # Should match values for "Stokes I" in the "Values to be compared"
        ######################################################################################
        # not part of the jupyter scripts
        tstobj = self # jupyter equivalent: "tstobj = test_j1302()"

        ##################################################
        # %% Set local vars [test_j1302_awproject] start @
        ##################################################

        #previous steps in the pipeline would have created mask files from catalogs and images that were created as an
        #intermediate pipeline step.
        tstobj.data_path_dir  = 'J1302/Stakeholder-test-mosaic-data'
        img0 = 'J1302_iter0d'
        img1 = 'J1302_iter2'
        tstobj.prepData()
        # rundir = "/users/bbean/dev/CAS-12427/src/casalith/build-casalith/work/linux/test_vlass_j1302_QL_unittest"
        # os.system(f"mv {rundir}/run_results/VLASS* {rundir}/nosedir/test_vlass_1v2/")

        ################################################
        # %% Set local vars [test_j1302_awproject] end @
        # .......................................

        # not part of the jupyter scripts
        starttime = datetime.now()

        # .......................................
        # %% Run tclean [test_j1302_awproject] start   @
        ################################################

        def run_tclean(vis=tstobj.vis, intent='OBSERVE_TARGET#UNSPECIFIED', phasecenter=tstobj.phasecenter,
                       imagename=None, niter=None, compare_tclean_pars=None,
                       reffreq='3.0GHz', deconvolver='mtmfs', cell='0.6arcsec', imsize=5250,
                       gridder='awproject', uvtaper=[''], uvrange='<12km', restoringbeam=[], wprojplanes=32,
                       mosweight=False, conjbeams=True, usepointing=True, rotatepastep=5.0, pblimit=0.02,
                       scales=[0], weighting='briggs', robust=1.0, nsigma=2.0, cycleniter=5000,
                       cyclefactor=3, interactive=0, smallscalebias=0.4, pointingoffsetsigdev=[300, 30]):
            params = locals()
            params = {k:params[k] for k in filter(lambda x: x not in ['tstobj'], params.keys())}
            tstobj.run_tclean(**params)

        def replace_psf(old, new):
            """ Replaces [old] PSF image with [new] image. Clears parallel working directories."""
            for this_tt in ['tt0', 'tt1', 'tt2']:
                shutil.rmtree(old+'.psf.'+this_tt)
                shutil.copytree(new+'.psf.'+this_tt, old+'.psf.'+this_tt)

        script_pars_vals_0 = None#tstobj.get_params_as_dict(vis='J1302-12fields.ms', selectdata=True, field='', spw='', timerange='', uvrange='<12km', antenna='', scan='', observation='', intent='OBSERVE_TARGET#UNSPECIFIED', datacolumn='corrected', imagename='J1302_iter0d', imsize=5250, cell='0.6arcsec', phasecenter='13:03:13.874 -10.51.16.73', stokes='I', projection='SIN', startmodel='', specmode='mfs', reffreq='3.0GHz', nchan=-1, start='', width='', outframe='LSRK', veltype='radio', restfreq=[], interpolation='linear', perchanweightdensity=True, gridder='awproject', facets=1, psfphasecenter='', chanchunks=1, wprojplanes=32, vptable='', mosweight=False, aterm=True, psterm=False, wbawp=False, conjbeams=True, cfcache='', usepointing=True, computepastep=360.0, rotatepastep=5.0, pointingoffsetsigdev=[300, 30], pblimit=0.02, normtype='flatnoise', deconvolver='mtmfs', scales=[0], nterms=2, smallscalebias=0.4, restoration=True, restoringbeam=[], pbcor=False, outlierfile='', weighting='briggs', robust=1.0, noise='1.0Jy', npixels=0, uvtaper=[''], niter=0, gain=0.1, threshold=0.0, nsigma=2.0, cycleniter=5000, cyclefactor=3.0, minpsffraction=0.05, maxpsffraction=0.8, interactive=False, usemask='user', mask='', pbmask=0.0, sidelobethreshold=3.0, noisethreshold=5.0, lownoisethreshold=1.5, negativethreshold=0.0, smoothfactor=1.0, minbeamfrac=0.3, cutthreshold=0.01, growiterations=75, dogrowprune=True, minpercentchange=-1.0, verbose=False, fastnoise=True, restart=True, savemodel='none', calcres=False, calcpsf=True, parallel=False )
        script_pars_vals_1 = None#tstobj.get_params_as_dict(vis='J1302-12fields.ms', selectdata=True, field='', spw='', timerange='', uvrange='<12km', antenna='', scan='', observation='', intent='OBSERVE_TARGET#UNSPECIFIED', datacolumn='corrected', imagename='J1302_iter2', imsize=5250, cell='0.6arcsec', phasecenter='13:03:13.874 -10.51.16.73', stokes='I', projection='SIN', startmodel='', specmode='mfs', reffreq='3.0GHz', nchan=-1, start='', width='', outframe='LSRK', veltype='radio', restfreq=[], interpolation='linear', perchanweightdensity=True, gridder='awproject', facets=1, psfphasecenter='', chanchunks=1, wprojplanes=32, vptable='', mosweight=False, aterm=True, psterm=False, wbawp=True, conjbeams=True, cfcache='', usepointing=True, computepastep=360.0, rotatepastep=5.0, pointingoffsetsigdev=[300, 30], pblimit=0.02, normtype='flatnoise', deconvolver='mtmfs', scales=[0], nterms=2, smallscalebias=0.4, restoration=True, restoringbeam=[], pbcor=False, outlierfile='', weighting='briggs', robust=1.0, noise='1.0Jy', npixels=0, uvtaper=[''], niter=0, gain=0.1, threshold=0.0, nsigma=2.0, cycleniter=5000, cyclefactor=3.0, minpsffraction=0.05, maxpsffraction=0.8, interactive=False, usemask='user', mask='', pbmask=0.0, sidelobethreshold=3.0, noisethreshold=5.0, lownoisethreshold=1.5, negativethreshold=0.0, smoothfactor=1.0, minbeamfrac=0.3, cutthreshold=0.01, growiterations=75, dogrowprune=True, minpercentchange=-1.0, verbose=False, fastnoise=True, restart=True, savemodel='none', calcres=True, calcpsf=True, parallel=False )
        script_pars_vals_2 = None#tstobj.get_params_as_dict(vis='J1302-12fields.ms', selectdata=True, field='', spw='', timerange='', uvrange='<12km', antenna='', scan='', observation='', intent='OBSERVE_TARGET#UNSPECIFIED', datacolumn='corrected', imagename='J1302_iter2', imsize=5250, cell='0.6arcsec', phasecenter='13:03:13.874 -10.51.16.73', stokes='I', projection='SIN', startmodel='', specmode='mfs', reffreq='3.0GHz', nchan=-1, start='', width='', outframe='LSRK', veltype='radio', restfreq=[], interpolation='linear', perchanweightdensity=True, gridder='awproject', facets=1, psfphasecenter='', chanchunks=1, wprojplanes=32, vptable='', mosweight=False, aterm=True, psterm=False, wbawp=True, conjbeams=True, cfcache='', usepointing=True, computepastep=360.0, rotatepastep=5.0, pointingoffsetsigdev=[300, 30], pblimit=0.02, normtype='flatnoise', deconvolver='mtmfs', scales=[0, 5, 12], nterms=2, smallscalebias=0.4, restoration=True, restoringbeam=[], pbcor=False, outlierfile='', weighting='briggs', robust=1.0, noise='1.0Jy', npixels=0, uvtaper=[''], niter=20000, gain=0.1, threshold=0.0, nsigma=3.0, cycleniter=3000, cyclefactor=3.0, minpsffraction=0.05, maxpsffraction=0.8, interactive=False, usemask='user', mask='J1302_QLcatmask.mask', pbmask=0.0, sidelobethreshold=3.0, noisethreshold=5.0, lownoisethreshold=1.5, negativethreshold=0.0, smoothfactor=1.0, minbeamfrac=0.3, cutthreshold=0.01, growiterations=75, dogrowprune=True, minpercentchange=-1.0, verbose=False, fastnoise=True, restart=True, savemodel='none', calcres=False, calcpsf=False, parallel=False )
        script_pars_vals_3 = None#tstobj.get_params_as_dict(vis='J1302-12fields.ms', selectdata=True, field='', spw='', timerange='', uvrange='<12km', antenna='', scan='', observation='', intent='OBSERVE_TARGET#UNSPECIFIED', datacolumn='data', imagename='J1302_iter2', imsize=5250, cell='0.6arcsec', phasecenter='13:03:13.874 -10.51.16.73', stokes='I', projection='SIN', startmodel='', specmode='mfs', reffreq='3.0GHz', nchan=-1, start='', width='', outframe='LSRK', veltype='radio', restfreq=[], interpolation='linear', perchanweightdensity=True, gridder='awproject', facets=1, psfphasecenter='', chanchunks=1, wprojplanes=32, vptable='', mosweight=False, aterm=True, psterm=False, wbawp=True, conjbeams=True, cfcache='', usepointing=True, computepastep=360.0, rotatepastep=5.0, pointingoffsetsigdev=[300, 30], pblimit=0.02, normtype='flatnoise', deconvolver='mtmfs', scales=[0], nterms=2, smallscalebias=0.4, restoration=True, restoringbeam=[], pbcor=False, outlierfile='', weighting='briggs', robust=1.0, noise='1.0Jy', npixels=0, uvtaper=[''], niter=0, gain=0.1, threshold=0.0, nsigma=2.0, cycleniter=5000, cyclefactor=3.0, minpsffraction=0.05, maxpsffraction=0.8, interactive=False, usemask='user', mask='', pbmask=0.0, sidelobethreshold=3.0, noisethreshold=5.0, lownoisethreshold=1.5, negativethreshold=0.0, smoothfactor=1.0, minbeamfrac=0.3, cutthreshold=0.01, growiterations=75, dogrowprune=True, minpercentchange=-1.0, verbose=False, fastnoise=True, restart=True, savemodel='modelcolumn', calcres=False, calcpsf=False, parallel=False )
        script_pars_vals_4 = None#tstobj.get_params_as_dict(vis='J1302-12fields.ms', selectdata=True, field='', spw='', timerange='', uvrange='<12km', antenna='', scan='', observation='', intent='OBSERVE_TARGET#UNSPECIFIED', datacolumn='corrected', imagename='J1302_iter2', imsize=5250, cell='0.6arcsec', phasecenter='13:03:13.874 -10.51.16.73', stokes='I', projection='SIN', startmodel='', specmode='mfs', reffreq='3.0GHz', nchan=-1, start='', width='', outframe='LSRK', veltype='radio', restfreq=[], interpolation='linear', perchanweightdensity=True, gridder='awproject', facets=1, psfphasecenter='', chanchunks=1, wprojplanes=32, vptable='', mosweight=False, aterm=True, psterm=False, wbawp=True, conjbeams=True, cfcache='', usepointing=True, computepastep=360.0, rotatepastep=5.0, pointingoffsetsigdev=[300, 30], pblimit=0.02, normtype='flatnoise', deconvolver='mtmfs', scales=[0, 5, 12], nterms=2, smallscalebias=0.4, restoration=True, restoringbeam=[], pbcor=False, outlierfile='', weighting='briggs', robust=1.0, noise='1.0Jy', npixels=0, uvtaper=[''], niter=20000, gain=0.1, threshold=0.0, nsigma=3.0, cycleniter=3000, cyclefactor=3.0, minpsffraction=0.05, maxpsffraction=0.8, interactive=False, usemask='user', mask='J1302_combined.mask', pbmask=0.0, sidelobethreshold=3.0, noisethreshold=5.0, lownoisethreshold=1.5, negativethreshold=0.0, smoothfactor=1.0, minbeamfrac=0.3, cutthreshold=0.01, growiterations=75, dogrowprune=True, minpercentchange=-1.0, verbose=False, fastnoise=True, restart=True, savemodel='none', calcres=False, calcpsf=False, parallel=False )
        script_pars_vals_5 = None#tstobj.get_params_as_dict(vis='J1302-12fields.ms', selectdata=True, field='', spw='', timerange='', uvrange='<12km', antenna='', scan='', observation='', intent='OBSERVE_TARGET#UNSPECIFIED', datacolumn='corrected', imagename='J1302_iter2', imsize=5250, cell='0.6arcsec', phasecenter='13:03:13.874 -10.51.16.73', stokes='I', projection='SIN', startmodel='', specmode='mfs', reffreq='3.0GHz', nchan=-1, start='', width='', outframe='LSRK', veltype='radio', restfreq=[], interpolation='linear', perchanweightdensity=True, gridder='awproject', facets=1, psfphasecenter='', chanchunks=1, wprojplanes=32, vptable='', mosweight=False, aterm=True, psterm=False, wbawp=True, conjbeams=True, cfcache='', usepointing=True, computepastep=360.0, rotatepastep=5.0, pointingoffsetsigdev=[300, 30], pblimit=0.02, normtype='flatnoise', deconvolver='mtmfs', scales=[0, 5, 12], nterms=2, smallscalebias=0.4, restoration=True, restoringbeam=[], pbcor=False, outlierfile='', weighting='briggs', robust=1.0, noise='1.0Jy', npixels=0, uvtaper=[''], niter=20000, gain=0.1, threshold=0.0, nsigma=4.5, cycleniter=500, cyclefactor=3.0, minpsffraction=0.05, maxpsffraction=0.8, interactive=False, usemask='pb', mask='', pbmask=0.4, sidelobethreshold=3.0, noisethreshold=5.0, lownoisethreshold=1.5, negativethreshold=0.0, smoothfactor=1.0, minbeamfrac=0.3, cutthreshold=0.01, growiterations=75, dogrowprune=True, minpercentchange=-1.0, verbose=False, fastnoise=True, restart=True, savemodel='none', calcres=False, calcpsf=False, parallel=False )

        # create robust psfs with wbawp=False using corrected column
        run_tclean(imagename=img0, niter=0, calcres=False, wbawp=False, 
                   compare_tclean_pars=script_pars_vals_0)

        # initialize iter2, no cleaning
        run_tclean(imagename=img1, niter=0, compare_tclean_pars=script_pars_vals_1)

        #  replace iter2 psf with no_WBAP
        replace_psf(img1, img0)

        #  resume iter2 with QL mask
        run_tclean(imagename=img1, niter=20000, scales=[0, 5, 12], nsigma=3.0, cycleniter=3000,
                   mask="J1302_QLcatmask.mask", calcres=False, calcpsf=False, compare_tclean_pars=script_pars_vals_2)

        # save model column, doesn't happen here in acutal VLASS pipeline, but makes sure functionality works.
        run_tclean(imagename=img1, niter=0, datacolumn='data', calcres=False, calcpsf=False,
                   savemodel='modelcolumn', compare_tclean_pars=script_pars_vals_3)

        # resume iter2 with combined mask
        os.system(f"rm -rf {img1}.mask")
        run_tclean(imagename=img1, niter=20000, scales=[0, 5, 12], nsigma=3.0, cycleniter=3000,
                   mask="J1302_combined.mask", calcres=False, calcpsf=False, compare_tclean_pars=script_pars_vals_4)

        # resume iter2 with pbmask, removed old mask first then specify pbmask in resumption of tclean
        os.system(f"rm -rf {img1}.mask")
        run_tclean(imagename=img1, niter=20000, scales=[0, 5, 12], nsigma=4.5, cycleniter=500,
                   mask="", calcres=False, calcpsf=False, usemask='pb', pbmask=0.4, compare_tclean_pars=script_pars_vals_5)

        ###########################################################
        # %% Run tclean [test_j1302_awproject] end                @
        # %% Compare Expected Values [test_j1302_awproject] start @
        ###########################################################

        # (l) Ensure intermediate products exist, pbcor images, RMS image (made by imdev), and cutouts (.subim) from imsubimage
        # N/A: no pbcore, rms, or subim images are created for this test

        tt1stats=imstat(imagename=imagename_base+'iter2.image.tt1',box='2625,2625,2625,2625')
        alphastats=imstat(imagename=imagename_base+'iter2.alpha',box='2625,2625,2625,2625')
        currentstats=np.squeeze(np.array([tt0stats['max'],tt1stats['max'],alphastats['max']]))
        onaxis_stats=np.array([           0.3337,         -0.01588,       -0.0476])
        casa613_stats=np.array([          0.3174496,      -0.01514572,    -0.04771062])

        # (a) tt0 vs 6.1.3, on-axis
        success0, report0 = tstobj.check_fracdiff(curr_stats[0], onaxis_stats[0],  valname="Frac Diff tt0 vs. on-axis")
        success1, report1 = tstobj.check_fracdiff(curr_stats[0], casa613_stats[0], valname="Frac Diff tt0 vs. 6.1.3 image")

        # (b) tt1 vs 6.1.3, on-axis
        success2, report2 = tstobj.check_fracdiff(curr_stats[1], onaxis_stats[1],  valname="Frac Diff tt1 vs. on-axis")
        success3, report3 = tstobj.check_fracdiff(curr_stats[1], casa613_stats[1], valname="Frac Diff tt1 vs. 6.1.3 image")

        # (c) alpha images
        success4, report4 = tstobj.check_fracdiff(curr_stats[2], onaxis_stats[2],  valname="Frac Diff alpha vs. on-axis")
        success5, report5 = tstobj.check_fracdiff(curr_stats[2], casa613_stats[2], valname="Frac Diff alpha vs. 6.1.3 image")

        # (d) beamsize comparison vs 6.1.3
        restbeam          = imhead(img1+'.image.tt0')['restoringbeam']
        beamstats_curr    = np.array([restbeam['major']['value'], restbeam['minor']['value'], restbeam['positionangle']['value']])
        beamstats_613     = np.array([3.07221413,                 2.49312615,                 11.04310322])
        success6, report6 = tstobj.check_fracdiff(beamstats_curr, beamstats_613, valname="Frac Diff Maj, Min, PA vs 6.1.3")

        # (e) Confirm presence of model column in resultant MS
        success7, report7 = tstobj.check_column_exists("MODEL_DATA")

        report  = "".join([report0, report1, report2, report3, report4, report5, report6, report7])
        success = success0 and success1 and success2 and success3 and success4 and success5 and success6 and success7 and th.check_final(report)
        casalog.post(f"{report}\nSuccess: {success}", "INFO")

        #########################################################
        # %% Compare Expected Values [test_j1302_awproject] end @
        #########################################################
        # not part of the jupyter scripts

        # (f) Runtimes not significantly different relative to previous runs
        # don't test this in jupyter notebooks - runtimes differ too much between machines
        success, report = tstobj.check_runtime(starttime, 1543, success, report)

        tstobj.assertTrue(success, msg=report)

    # Test 3
    @unittest.skipIf(ParallelTaskHelper.isMPIEnabled(), "Skip test. Tclean crashes with mpicasa+mosaic gridder+stokes imaging.")
    def test_j1302_mosaic_cube(self):
        """ [j1302] test_j1302_mosaic_cube """
        ######################################################################################
        # Should match values for "Cube" in the "Values to be compared"
        ######################################################################################
        # not part of the jupyter scripts
        tstobj = self # jupyter equivalent: "tstobj = test_j1302()"

        ####################################################
        # %% Set local vars [test_j1302_mosaic_cube] start @
        ####################################################

        #previous steps in the pipeline would have created mask files from catalogs and images that were created as an
        #intermediate pipeline step.
        tstobj.data_path_dir  = 'J1302/Stakeholder-test-mosaic-cube-data'
        tstobj.prepData()
        # rundir = "/users/bbean/dev/CAS-12427/src/casalith/build-casalith/work/linux/test_vlass_j1302_cube_unittest"
        # os.system(f"mv {rundir}/run_results/VLASS* {rundir}/nosedir/test_vlass_1v2/")

        # reference frequence to use per spectral window (spw)
        refFreqDict  = {
            '2' :  '2.028GHz',
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

        ##################################################
        # %% Set local vars [test_j1302_mosaic_cube] end @
        # .......................................

        # not part of the jupyter scripts
        starttime = datetime.now()

        # .......................................
        # %% Run tclean [test_j1302_mosaic_cube] start   @
        ##################################################

        def iname(image_iter, spw, stokes): # imagename=imagename_base+image_iter+'_'+spw.replace('~','-')+'_'+stokes
            return 'J1302_'+image_iter+'_'+spw.replace('~','-')+'_'+stokes

        def run_tclean(vis=tstobj.vis, uvrange='<12km', imsize=4000, intent='OBSERVE_TARGET#UNSPECIFIED',
                       imagename=None, niter=None, compare_tclean_pars=None,
                       cell='0.6arcsec', datacolumn='data', phasecenter=tstobj.phasecenter,
                       reffreq='3.0GHz', deconvolver='mtmfs', gridder='mosaic', restoringbeam=[],
                       mosweight=False, rotatepastep=5.0, pblimit=0.1, scales=[0], nterms=1,
                       weighting='briggs', smallscalebias=0.4, robust=1.0, uvtaper=[''], nsigma=2.0,
                       cycleniter=500, cyclefactor=3, interactive=0):
            params = locals()
            params = {k:params[k] for k in filter(lambda x: x not in ['tstobj'], params.keys())}
            tstobj.run_tclean(**params)

        spws         = [ '2','8','14']
        stokesParams = ['IQUV']
        spwstats     = { '2':  {'freq': 2.028, 'IQUV': np.array([0.0,0.0,0.0,0.0]), 'beam': np.array([0.0,0.0,0.0])},
                         '8':  {'freq': 2.796, 'IQUV': np.array([0.0,0.0,0.0,0.0]), 'beam': np.array([0.0,0.0,0.0])},
                         '14': {'freq': 3.594, 'IQUV': np.array([0.0,0.0,0.0,0.0]), 'beam': np.array([0.0,0.0,0.0])} }

        script_pars_vals_0 = {
            '2':  { 'IQUV': None },#tstobj.get_params_as_dict(vis='J1302-12fields.ms', selectdata=True, field='', spw='2', timerange='', uvrange='<12km', antenna='', scan='', observation='', intent='OBSERVE_TARGET#UNSPECIFIED', datacolumn='corrected', imagename='J1302_iter2_2_IQUV', imsize=4000, cell='0.6arcsec', phasecenter='13:03:13.874 -10.51.16.73', stokes='IQUV', projection='SIN', startmodel='', specmode='mfs', reffreq='2.028GHz', nchan=-1, start='', width='', outframe='LSRK', veltype='radio', restfreq=[], interpolation='linear', perchanweightdensity=True, gridder='mosaic', facets=1, psfphasecenter='', chanchunks=1, wprojplanes=1, vptable='', mosweight=False, aterm=True, psterm=False, wbawp=True, conjbeams=False, cfcache='', usepointing=False, computepastep=360.0, rotatepastep=5.0, pointingoffsetsigdev=[], pblimit=0.1, normtype='flatnoise', deconvolver='mtmfs', scales=[0], nterms=1, smallscalebias=0.4, restoration=True, restoringbeam=[], pbcor=False, outlierfile='', weighting='briggs', robust=1.0, noise='1.0Jy', npixels=0, uvtaper=[''], niter=0, gain=0.1, threshold=0.0, nsigma=2.0, cycleniter=500, cyclefactor=3.0, minpsffraction=0.05, maxpsffraction=0.8, interactive=False, usemask='user', mask='', pbmask=0.0, sidelobethreshold=3.0, noisethreshold=5.0, lownoisethreshold=1.5, negativethreshold=0.0, smoothfactor=1.0, minbeamfrac=0.3, cutthreshold=0.01, growiterations=75, dogrowprune=True, minpercentchange=-1.0, verbose=False, fastnoise=True, restart=True, savemodel='none', calcres=True, calcpsf=True, parallel=False) },
            '8':  { 'IQUV': None },#tstobj.get_params_as_dict(vis='J1302-12fields.ms', selectdata=True, field='', spw='8', timerange='', uvrange='<12km', antenna='', scan='', observation='', intent='OBSERVE_TARGET#UNSPECIFIED', datacolumn='corrected', imagename='J1302_iter2_8_IQUV', imsize=4000, cell='0.6arcsec', phasecenter='13:03:13.874 -10.51.16.73', stokes='IQUV', projection='SIN', startmodel='', specmode='mfs', reffreq='2.796GHz', nchan=-1, start='', width='', outframe='LSRK', veltype='radio', restfreq=[], interpolation='linear', perchanweightdensity=True, gridder='mosaic', facets=1, psfphasecenter='', chanchunks=1, wprojplanes=1, vptable='', mosweight=False, aterm=True, psterm=False, wbawp=True, conjbeams=False, cfcache='', usepointing=False, computepastep=360.0, rotatepastep=5.0, pointingoffsetsigdev=[], pblimit=0.1, normtype='flatnoise', deconvolver='mtmfs', scales=[0], nterms=1, smallscalebias=0.4, restoration=True, restoringbeam=[], pbcor=False, outlierfile='', weighting='briggs', robust=1.0, noise='1.0Jy', npixels=0, uvtaper=[''], niter=0, gain=0.1, threshold=0.0, nsigma=2.0, cycleniter=500, cyclefactor=3.0, minpsffraction=0.05, maxpsffraction=0.8, interactive=False, usemask='user', mask='', pbmask=0.0, sidelobethreshold=3.0, noisethreshold=5.0, lownoisethreshold=1.5, negativethreshold=0.0, smoothfactor=1.0, minbeamfrac=0.3, cutthreshold=0.01, growiterations=75, dogrowprune=True, minpercentchange=-1.0, verbose=False, fastnoise=True, restart=True, savemodel='none', calcres=True, calcpsf=True, parallel=False) },
            '14': { 'IQUV': None },#tstobj.get_params_as_dict(vis='J1302-12fields.ms', selectdata=True, field='', spw='14', timerange='', uvrange='<12km', antenna='', scan='', observation='', intent='OBSERVE_TARGET#UNSPECIFIED', datacolumn='corrected', imagename='J1302_iter2_14_IQUV', imsize=4000, cell='0.6arcsec', phasecenter='13:03:13.874 -10.51.16.73', stokes='IQUV', projection='SIN', startmodel='', specmode='mfs', reffreq='3.564GHz', nchan=-1, start='', width='', outframe='LSRK', veltype='radio', restfreq=[], interpolation='linear', perchanweightdensity=True, gridder='mosaic', facets=1, psfphasecenter='', chanchunks=1, wprojplanes=1, vptable='', mosweight=False, aterm=True, psterm=False, wbawp=True, conjbeams=False, cfcache='', usepointing=False, computepastep=360.0, rotatepastep=5.0, pointingoffsetsigdev=[], pblimit=0.1, normtype='flatnoise', deconvolver='mtmfs', scales=[0], nterms=1, smallscalebias=0.4, restoration=True, restoringbeam=[], pbcor=False, outlierfile='', weighting='briggs', robust=1.0, noise='1.0Jy', npixels=0, uvtaper=[''], niter=0, gain=0.1, threshold=0.0, nsigma=2.0, cycleniter=500, cyclefactor=3.0, minpsffraction=0.05, maxpsffraction=0.8, interactive=False, usemask='user', mask='', pbmask=0.0, sidelobethreshold=3.0, noisethreshold=5.0, lownoisethreshold=1.5, negativethreshold=0.0, smoothfactor=1.0, minbeamfrac=0.3, cutthreshold=0.01, growiterations=75, dogrowprune=True, minpercentchange=-1.0, verbose=False, fastnoise=True, restart=True, savemodel='none', calcres=True, calcpsf=True, parallel=False) },
        }
        script_pars_vals_1 = {
            '2':  { 'IQUV': None },#tstobj.get_params_as_dict(vis='J1302-12fields.ms', selectdata=True, field='', spw='2', timerange='', uvrange='<12km', antenna='', scan='', observation='', intent='OBSERVE_TARGET#UNSPECIFIED', datacolumn='corrected', imagename='J1302_iter2_2_IQUV', imsize=4000, cell='0.6arcsec', phasecenter='13:03:13.874 -10.51.16.73', stokes='IQUV', projection='SIN', startmodel='', specmode='mfs', reffreq='2.028GHz', nchan=-1, start='', width='', outframe='LSRK', veltype='radio', restfreq=[], interpolation='linear', perchanweightdensity=True, gridder='mosaic', facets=1, psfphasecenter='', chanchunks=1, wprojplanes=1, vptable='', mosweight=False, aterm=True, psterm=False, wbawp=True, conjbeams=False, cfcache='', usepointing=False, computepastep=360.0, rotatepastep=5.0, pointingoffsetsigdev=[], pblimit=0.1, normtype='flatnoise', deconvolver='mtmfs', scales=[0, 5, 12], nterms=1, smallscalebias=0.4, restoration=True, restoringbeam=[], pbcor=False, outlierfile='', weighting='briggs', robust=1.0, noise='1.0Jy', npixels=0, uvtaper=[''], niter=20000, gain=0.1, threshold=0.0, nsigma=3.0, cycleniter=500, cyclefactor=3.0, minpsffraction=0.05, maxpsffraction=0.8, interactive=False, usemask='user', mask='QLcatmask.mask', pbmask=0.0, sidelobethreshold=3.0, noisethreshold=5.0, lownoisethreshold=1.5, negativethreshold=0.0, smoothfactor=1.0, minbeamfrac=0.3, cutthreshold=0.01, growiterations=75, dogrowprune=True, minpercentchange=-1.0, verbose=False, fastnoise=True, restart=True, savemodel='none', calcres=False, calcpsf=False, parallel=False) },
            '8':  { 'IQUV': None },#tstobj.get_params_as_dict(vis='J1302-12fields.ms', selectdata=True, field='', spw='8', timerange='', uvrange='<12km', antenna='', scan='', observation='', intent='OBSERVE_TARGET#UNSPECIFIED', datacolumn='corrected', imagename='J1302_iter2_8_IQUV', imsize=4000, cell='0.6arcsec', phasecenter='13:03:13.874 -10.51.16.73', stokes='IQUV', projection='SIN', startmodel='', specmode='mfs', reffreq='2.796GHz', nchan=-1, start='', width='', outframe='LSRK', veltype='radio', restfreq=[], interpolation='linear', perchanweightdensity=True, gridder='mosaic', facets=1, psfphasecenter='', chanchunks=1, wprojplanes=1, vptable='', mosweight=False, aterm=True, psterm=False, wbawp=True, conjbeams=False, cfcache='', usepointing=False, computepastep=360.0, rotatepastep=5.0, pointingoffsetsigdev=[], pblimit=0.1, normtype='flatnoise', deconvolver='mtmfs', scales=[0, 5, 12], nterms=1, smallscalebias=0.4, restoration=True, restoringbeam=[], pbcor=False, outlierfile='', weighting='briggs', robust=1.0, noise='1.0Jy', npixels=0, uvtaper=[''], niter=20000, gain=0.1, threshold=0.0, nsigma=3.0, cycleniter=500, cyclefactor=3.0, minpsffraction=0.05, maxpsffraction=0.8, interactive=False, usemask='user', mask='QLcatmask.mask', pbmask=0.0, sidelobethreshold=3.0, noisethreshold=5.0, lownoisethreshold=1.5, negativethreshold=0.0, smoothfactor=1.0, minbeamfrac=0.3, cutthreshold=0.01, growiterations=75, dogrowprune=True, minpercentchange=-1.0, verbose=False, fastnoise=True, restart=True, savemodel='none', calcres=False, calcpsf=False, parallel=False) },
            '14': { 'IQUV': None },#tstobj.get_params_as_dict(vis='J1302-12fields.ms', selectdata=True, field='', spw='14', timerange='', uvrange='<12km', antenna='', scan='', observation='', intent='OBSERVE_TARGET#UNSPECIFIED', datacolumn='corrected', imagename='J1302_iter2_14_IQUV', imsize=4000, cell='0.6arcsec', phasecenter='13:03:13.874 -10.51.16.73', stokes='IQUV', projection='SIN', startmodel='', specmode='mfs', reffreq='3.564GHz', nchan=-1, start='', width='', outframe='LSRK', veltype='radio', restfreq=[], interpolation='linear', perchanweightdensity=True, gridder='mosaic', facets=1, psfphasecenter='', chanchunks=1, wprojplanes=1, vptable='', mosweight=False, aterm=True, psterm=False, wbawp=True, conjbeams=False, cfcache='', usepointing=False, computepastep=360.0, rotatepastep=5.0, pointingoffsetsigdev=[], pblimit=0.1, normtype='flatnoise', deconvolver='mtmfs', scales=[0, 5, 12], nterms=1, smallscalebias=0.4, restoration=True, restoringbeam=[], pbcor=False, outlierfile='', weighting='briggs', robust=1.0, noise='1.0Jy', npixels=0, uvtaper=[''], niter=20000, gain=0.1, threshold=0.0, nsigma=3.0, cycleniter=500, cyclefactor=3.0, minpsffraction=0.05, maxpsffraction=0.8, interactive=False, usemask='user', mask='QLcatmask.mask', pbmask=0.0, sidelobethreshold=3.0, noisethreshold=5.0, lownoisethreshold=1.5, negativethreshold=0.0, smoothfactor=1.0, minbeamfrac=0.3, cutthreshold=0.01, growiterations=75, dogrowprune=True, minpercentchange=-1.0, verbose=False, fastnoise=True, restart=True, savemodel='none', calcres=False, calcpsf=False, parallel=False) },
        }
        script_pars_vals_2 = {
            '2':  { 'IQUV': None },#tstobj.get_params_as_dict(vis='J1302-12fields.ms', selectdata=True, field='', spw='2', timerange='', uvrange='<12km', antenna='', scan='', observation='', intent='OBSERVE_TARGET#UNSPECIFIED', datacolumn='corrected', imagename='J1302_iter2_2_IQUV', imsize=4000, cell='0.6arcsec', phasecenter='13:03:13.874 -10.51.16.73', stokes='IQUV', projection='SIN', startmodel='', specmode='mfs', reffreq='2.028GHz', nchan=-1, start='', width='', outframe='LSRK', veltype='radio', restfreq=[], interpolation='linear', perchanweightdensity=True, gridder='mosaic', facets=1, psfphasecenter='', chanchunks=1, wprojplanes=1, vptable='', mosweight=False, aterm=True, psterm=False, wbawp=True, conjbeams=False, cfcache='', usepointing=False, computepastep=360.0, rotatepastep=5.0, pointingoffsetsigdev=[], pblimit=0.1, normtype='flatnoise', deconvolver='mtmfs', scales=[0, 5, 12], nterms=1, smallscalebias=0.4, restoration=True, restoringbeam=[], pbcor=False, outlierfile='', weighting='briggs', robust=1.0, noise='1.0Jy', npixels=0, uvtaper=[''], niter=20000, gain=0.1, threshold=0.0, nsigma=3.0, cycleniter=500, cyclefactor=3.0, minpsffraction=0.05, maxpsffraction=0.8, interactive=False, usemask='user', mask='combined.mask', pbmask=0.0, sidelobethreshold=3.0, noisethreshold=5.0, lownoisethreshold=1.5, negativethreshold=0.0, smoothfactor=1.0, minbeamfrac=0.3, cutthreshold=0.01, growiterations=75, dogrowprune=True, minpercentchange=-1.0, verbose=False, fastnoise=True, restart=True, savemodel='none', calcres=False, calcpsf=False, parallel=False) },
            '8':  { 'IQUV': None },#tstobj.get_params_as_dict(vis='J1302-12fields.ms', selectdata=True, field='', spw='8', timerange='', uvrange='<12km', antenna='', scan='', observation='', intent='OBSERVE_TARGET#UNSPECIFIED', datacolumn='corrected', imagename='J1302_iter2_8_IQUV', imsize=4000, cell='0.6arcsec', phasecenter='13:03:13.874 -10.51.16.73', stokes='IQUV', projection='SIN', startmodel='', specmode='mfs', reffreq='2.796GHz', nchan=-1, start='', width='', outframe='LSRK', veltype='radio', restfreq=[], interpolation='linear', perchanweightdensity=True, gridder='mosaic', facets=1, psfphasecenter='', chanchunks=1, wprojplanes=1, vptable='', mosweight=False, aterm=True, psterm=False, wbawp=True, conjbeams=False, cfcache='', usepointing=False, computepastep=360.0, rotatepastep=5.0, pointingoffsetsigdev=[], pblimit=0.1, normtype='flatnoise', deconvolver='mtmfs', scales=[0, 5, 12], nterms=1, smallscalebias=0.4, restoration=True, restoringbeam=[], pbcor=False, outlierfile='', weighting='briggs', robust=1.0, noise='1.0Jy', npixels=0, uvtaper=[''], niter=20000, gain=0.1, threshold=0.0, nsigma=3.0, cycleniter=500, cyclefactor=3.0, minpsffraction=0.05, maxpsffraction=0.8, interactive=False, usemask='user', mask='combined.mask', pbmask=0.0, sidelobethreshold=3.0, noisethreshold=5.0, lownoisethreshold=1.5, negativethreshold=0.0, smoothfactor=1.0, minbeamfrac=0.3, cutthreshold=0.01, growiterations=75, dogrowprune=True, minpercentchange=-1.0, verbose=False, fastnoise=True, restart=True, savemodel='none', calcres=False, calcpsf=False, parallel=False) },
            '14': { 'IQUV': None },#tstobj.get_params_as_dict(vis='J1302-12fields.ms', selectdata=True, field='', spw='14', timerange='', uvrange='<12km', antenna='', scan='', observation='', intent='OBSERVE_TARGET#UNSPECIFIED', datacolumn='corrected', imagename='J1302_iter2_14_IQUV', imsize=4000, cell='0.6arcsec', phasecenter='13:03:13.874 -10.51.16.73', stokes='IQUV', projection='SIN', startmodel='', specmode='mfs', reffreq='3.564GHz', nchan=-1, start='', width='', outframe='LSRK', veltype='radio', restfreq=[], interpolation='linear', perchanweightdensity=True, gridder='mosaic', facets=1, psfphasecenter='', chanchunks=1, wprojplanes=1, vptable='', mosweight=False, aterm=True, psterm=False, wbawp=True, conjbeams=False, cfcache='', usepointing=False, computepastep=360.0, rotatepastep=5.0, pointingoffsetsigdev=[], pblimit=0.1, normtype='flatnoise', deconvolver='mtmfs', scales=[0, 5, 12], nterms=1, smallscalebias=0.4, restoration=True, restoringbeam=[], pbcor=False, outlierfile='', weighting='briggs', robust=1.0, noise='1.0Jy', npixels=0, uvtaper=[''], niter=20000, gain=0.1, threshold=0.0, nsigma=3.0, cycleniter=500, cyclefactor=3.0, minpsffraction=0.05, maxpsffraction=0.8, interactive=False, usemask='user', mask='combined.mask', pbmask=0.0, sidelobethreshold=3.0, noisethreshold=5.0, lownoisethreshold=1.5, negativethreshold=0.0, smoothfactor=1.0, minbeamfrac=0.3, cutthreshold=0.01, growiterations=75, dogrowprune=True, minpercentchange=-1.0, verbose=False, fastnoise=True, restart=True, savemodel='none', calcres=False, calcpsf=False, parallel=False) },
        }
        script_pars_vals_3 = {
            '2':  { 'IQUV': None },#tstobj.get_params_as_dict(vis='J1302-12fields.ms', selectdata=True, field='', spw='2', timerange='', uvrange='<12km', antenna='', scan='', observation='', intent='OBSERVE_TARGET#UNSPECIFIED', datacolumn='corrected', imagename='J1302_iter2_2_IQUV', imsize=4000, cell='0.6arcsec', phasecenter='13:03:13.874 -10.51.16.73', stokes='IQUV', projection='SIN', startmodel='', specmode='mfs', reffreq='2.028GHz', nchan=-1, start='', width='', outframe='LSRK', veltype='radio', restfreq=[], interpolation='linear', perchanweightdensity=True, gridder='mosaic', facets=1, psfphasecenter='', chanchunks=1, wprojplanes=1, vptable='', mosweight=False, aterm=True, psterm=False, wbawp=True, conjbeams=False, cfcache='', usepointing=False, computepastep=360.0, rotatepastep=5.0, pointingoffsetsigdev=[], pblimit=0.1, normtype='flatnoise', deconvolver='mtmfs', scales=[0, 5, 12], nterms=1, smallscalebias=0.4, restoration=True, restoringbeam=[], pbcor=False, outlierfile='', weighting='briggs', robust=1.0, noise='1.0Jy', npixels=0, uvtaper=[''], niter=20000, gain=0.1, threshold=0.0, nsigma=4.5, cycleniter=100, cyclefactor=3.0, minpsffraction=0.05, maxpsffraction=0.8, interactive=False, usemask='pb', mask='', pbmask=0.4, sidelobethreshold=3.0, noisethreshold=5.0, lownoisethreshold=1.5, negativethreshold=0.0, smoothfactor=1.0, minbeamfrac=0.3, cutthreshold=0.01, growiterations=75, dogrowprune=True, minpercentchange=-1.0, verbose=False, fastnoise=True, restart=True, savemodel='none', calcres=False, calcpsf=False, parallel=False) },
            '8':  { 'IQUV': None },#tstobj.get_params_as_dict(vis='J1302-12fields.ms', selectdata=True, field='', spw='8', timerange='', uvrange='<12km', antenna='', scan='', observation='', intent='OBSERVE_TARGET#UNSPECIFIED', datacolumn='corrected', imagename='J1302_iter2_8_IQUV', imsize=4000, cell='0.6arcsec', phasecenter='13:03:13.874 -10.51.16.73', stokes='IQUV', projection='SIN', startmodel='', specmode='mfs', reffreq='2.796GHz', nchan=-1, start='', width='', outframe='LSRK', veltype='radio', restfreq=[], interpolation='linear', perchanweightdensity=True, gridder='mosaic', facets=1, psfphasecenter='', chanchunks=1, wprojplanes=1, vptable='', mosweight=False, aterm=True, psterm=False, wbawp=True, conjbeams=False, cfcache='', usepointing=False, computepastep=360.0, rotatepastep=5.0, pointingoffsetsigdev=[], pblimit=0.1, normtype='flatnoise', deconvolver='mtmfs', scales=[0, 5, 12], nterms=1, smallscalebias=0.4, restoration=True, restoringbeam=[], pbcor=False, outlierfile='', weighting='briggs', robust=1.0, noise='1.0Jy', npixels=0, uvtaper=[''], niter=20000, gain=0.1, threshold=0.0, nsigma=4.5, cycleniter=100, cyclefactor=3.0, minpsffraction=0.05, maxpsffraction=0.8, interactive=False, usemask='pb', mask='', pbmask=0.4, sidelobethreshold=3.0, noisethreshold=5.0, lownoisethreshold=1.5, negativethreshold=0.0, smoothfactor=1.0, minbeamfrac=0.3, cutthreshold=0.01, growiterations=75, dogrowprune=True, minpercentchange=-1.0, verbose=False, fastnoise=True, restart=True, savemodel='none', calcres=False, calcpsf=False, parallel=False) },
            '14': { 'IQUV': None },#tstobj.get_params_as_dict(vis='J1302-12fields.ms', selectdata=True, field='', spw='14', timerange='', uvrange='<12km', antenna='', scan='', observation='', intent='OBSERVE_TARGET#UNSPECIFIED', datacolumn='corrected', imagename='J1302_iter2_14_IQUV', imsize=4000, cell='0.6arcsec', phasecenter='13:03:13.874 -10.51.16.73', stokes='IQUV', projection='SIN', startmodel='', specmode='mfs', reffreq='3.564GHz', nchan=-1, start='', width='', outframe='LSRK', veltype='radio', restfreq=[], interpolation='linear', perchanweightdensity=True, gridder='mosaic', facets=1, psfphasecenter='', chanchunks=1, wprojplanes=1, vptable='', mosweight=False, aterm=True, psterm=False, wbawp=True, conjbeams=False, cfcache='', usepointing=False, computepastep=360.0, rotatepastep=5.0, pointingoffsetsigdev=[], pblimit=0.1, normtype='flatnoise', deconvolver='mtmfs', scales=[0, 5, 12], nterms=1, smallscalebias=0.4, restoration=True, restoringbeam=[], pbcor=False, outlierfile='', weighting='briggs', robust=1.0, noise='1.0Jy', npixels=0, uvtaper=[''], niter=20000, gain=0.1, threshold=0.0, nsigma=4.5, cycleniter=100, cyclefactor=3.0, minpsffraction=0.05, maxpsffraction=0.8, interactive=False, usemask='pb', mask='', pbmask=0.4, sidelobethreshold=3.0, noisethreshold=5.0, lownoisethreshold=1.5, negativethreshold=0.0, smoothfactor=1.0, minbeamfrac=0.3, cutthreshold=0.01, growiterations=75, dogrowprune=True, minpercentchange=-1.0, verbose=False, fastnoise=True, restart=True, savemodel='none', calcres=False, calcpsf=False, parallel=False) },
        }

        for spw in spws:
           for stokes in stokesParams:
              # initialize iter2, no cleaning
              image_iter='iter2'
              imagename = iname(image_iter, spw, stokes)
              run_tclean( imagename=imagename, datacolumn='corrected', niter=0, spw=spw,
                          stokes=stokes, reffreq=refFreqDict[spw], compare_tclean_pars=script_pars_vals_0[spw][stokes] )

              # resume iter2 with QL mask
              run_tclean( imagename=imagename, datacolumn='corrected', scales=[0,5,12], nsigma=3.0, niter=20000, cycleniter=500,   
                          mask="QLcatmask.mask", calcres=False, calcpsf=False, spw=spw,
                          stokes=stokes, reffreq=refFreqDict[spw], compare_tclean_pars=script_pars_vals_1[spw][stokes] )

              # resume iter2 with combined mask
              os.system('rm -rf *.workdirectory')
              os.system('rm -rf *iter2*.mask')
              run_tclean( imagename=imagename, datacolumn='corrected', scales=[0,5,12], nsigma=3.0, niter=20000, cycleniter=500,
                          mask="combined.mask", calcres=False, calcpsf=False, spw=spw,
                          stokes=stokes, reffreq=refFreqDict[spw], compare_tclean_pars=script_pars_vals_2[spw][stokes])

              # os.system('rm -rf iter2*.mask')
              run_tclean( imagename=imagename, datacolumn='corrected', scales=[0,5,12], nsigma=4.5, niter=20000, cycleniter=100,
                          mask="", calcres=False, calcpsf=False, usemask='pb', pbmask=0.4, spw=spw,
                          stokes=stokes, reffreq=refFreqDict[spw], compare_tclean_pars=script_pars_vals_3[spw][stokes] )

              tt0statsI=imstat(imagename=imagename+'.image.tt0',box='2000,2000,2000,2000',stokes='I')
              tt0statsQ=imstat(imagename=imagename+'.image.tt0',box='2000,2000,2000,2000',stokes='Q')
              tt0statsU=imstat(imagename=imagename+'.image.tt0',box='2000,2000,2000,2000',stokes='U')
              tt0statsV=imstat(imagename=imagename+'.image.tt0',box='2000,2000,2000,2000',stokes='V')
              spwstats[spw]['IQUV']=np.array([tt0statsI['max'],tt0statsQ['max'],tt0statsU['max'],tt0statsV['max']])
              rmstt0statsI=imstat(imagename=imagename+'.residual.tt0',stokes='I')
              rmstt0statsQ=imstat(imagename=imagename+'.residual.tt0',stokes='Q')
              rmstt0statsU=imstat(imagename=imagename+'.residual.tt0',stokes='U')
              rmstt0statsV=imstat(imagename=imagename+'.residual.tt0',stokes='V')
              spwstats[spw]['IQUV']=np.array([tt0statsI['max'],tt0statsQ['max'],tt0statsU['max'],tt0statsV['max']])
              spwstats[spw]['rmsIQUV']=np.array([rmstt0statsI['rms'],rmstt0statsQ['rms'],rmstt0statsU['rms'],rmstt0statsV['rms']])
              spwstats[spw]['SNR']=((spwstats[spw]['IQUV']/spwstats[spw]['rmsIQUV'])**2)**0.5

              header = imhead(imagename+'.psf.tt0')
              beam   = header['perplanebeams']['beams']['*0']['*0']
              beamstats = np.array([ beam['major']['value'], beam['minor']['value'], beam['positionangle']['value'] ])
              spwstats[spw]['beam'] = beamstats

        ####################################################
        # %% Run tclean [test_j1302_mosaic_cube] end       @
        # %% Math stuff [test_j1302_mosaic_cube] start     @
        ####################################################

        spwlist=list(spwstats.keys())
        nspws=len(spwlist)
        freqs=np.zeros(nspws)
        fluxes=np.zeros(nspws)
        for i in range(nspws):
           freqs[i]=spwstats[spwlist[i]]['freq']
           fluxes[i]=spwstats[spwlist[i]]['IQUV'][0]

        logfreqs=np.log10(freqs)
        logfluxes=np.log10(fluxes)

        from scipy.optimize import curve_fit

        def func(x, a, b):
           nu_0=3.0
           return a*(x-np.log10(nu_0))+b

        popt, pcov = curve_fit(func, logfreqs, logfluxes)
        perr = np.sqrt(np.diag(pcov))

        #############################################################
        # %% Math stuff [test_j1302_mosaic_cube] end                @
        # %% Compare Expected Values [test_j1302_mosaic_cube] start @
        #############################################################

        # (l) Ensure intermediate products exist, pbcor images, RMS image (made by imdev), and cutouts (.subim) from imsubimage
        # N/A: no pbcore, rms, or subim images are created for this test

        # (g) Fit F_nu0 and Alpha from three cube planes and compare: 6.1.3, on-axis
        # compare to alpha (ground truth), and 6.1.3 (fitted for spws 2, 8, 14 from mosaic gridder in CASA 6.1.3)
        alpha = popt[0]
        f_nu0 = 10**popt[1]
        curr_stats        = np.squeeze(np.array([f_nu0,  alpha])) # [flux density, alpha]
        onaxis_stats      = np.array([           0.3337, -0.0476])
        casa613_stats     = np.array([           0.3127, 0.04134])
        success0, report0 = tstobj.check_fracdiff(curr_stats[0], onaxis_stats[0],  valname="Frac Diff F_nu (on-axis)")
        success1, report1 = tstobj.check_fracdiff(curr_stats[0], casa613_stats[0], valname="Frac Diff F_nu (6.1.3 image)")
        success2, report2 = tstobj.check_fracdiff(curr_stats[1], onaxis_stats[1],  valname="Frac Diff alpha (on-axis)")
        success3, report3 = tstobj.check_fracdiff(curr_stats[1], casa613_stats[1], valname="Frac Diff alpha (6.1.3 image)")

        spwstats_613={
          '2': {'IQUV': np.array([[ 0.3024486 ],
                 [-0.00169682],
                 [-0.00040808],
                 [-0.00172231]]),
          'beam': np.array([4.63033438, 3.7626121 , 8.42547607]),
          'rmsIQUV': np.array([[0.00058096],
                 [0.00053226],
                 [0.00052702],
                 [0.00055715]]),
          'SNR': np.array([[520.60473159],
                 [  3.18793882],
                 [  0.77430771],
                 [  3.09130732]]),
          'freq': 2.028},
          '8': {'IQUV': np.array([[ 3.24606597e-01],
                 [-1.02521779e-04],
                 [-2.74724560e-04],
                 [-9.82215162e-04]]),
          'beam': np.array([ 3.31582212,  2.81106067, 12.23698997]),
          'rmsIQUV': np.array([[0.00060875],
                 [0.00056557],
                 [0.00056437],
                 [0.00058742]]),
          'SNR': np.array([[5.33232366e+02],
                 [1.81271795e-01],
                 [4.86780379e-01],
                 [1.67209226e+00]]),
          'freq': 2.796},
          '14': {'IQUV': np.array([[ 0.30785725],
                 [ 0.00112553],
                 [-0.00233958],
                 [-0.00072553]]),
          'beam': np.array([2.06845379, 1.62075114, 8.15380859]),
          'rmsIQUV': np.array([[0.00120004],
                 [0.00096465],
                 [0.00095693],
                 [0.00094171]]),
          'SNR': np.array([[256.53982683],
                 [  1.16677654],
                 [  2.44488894],
                 [  0.77044591]]),
          'freq': 3.564}
        }

        spwstats_onaxis={
          '2': {'IQUV': np.array([[ 3.09755385e-01],
                [-1.39351614e-04],
                [-1.01510414e-04],
                [ 3.48565959e-06]]),
          'beam': np.array([ 4.39902544,  2.97761726, -1.9803896 ]),
          'rmsIQUV': np.array([[9.06101077e-05],
                [6.18512027e-05],
                [6.08855185e-05],
                [6.16381383e-05]]),
          'SNR': np.array([[3.41855222e+03],
                [2.25301381e+00],
                [1.66723411e+00],
                [5.65503710e-02]]),
          'freq': 2.028},
          '8': {'IQUV': np.array([[ 3.33031625e-01],
                [-1.70329688e-04],
                [-5.54503786e-05],
                [ 1.51384829e-05]]),
          'beam': np.array([ 3.17962098,  2.23836327, -3.84393311]),
          'rmsIQUV': np.array([[6.83661207e-05],
                [5.41465193e-05],
                [5.37294874e-05],
                [5.48078784e-05]]),
          'SNR': np.array([[4.87129621e+03],
                [3.14571813e+00],
                [1.03202880e+00],
                [2.76209979e-01]]),
          'freq': 2.796},
          '14': {'IQUV': np.array([[ 3.26020330e-01],
                [-1.19368524e-04],
                [ 1.94136555e-05],
                [-3.02872763e-06]]),
          'beam': np.array([ 2.4293716 ,  1.61365998, -3.72186279]),
          'rmsIQUV': np.array([[6.07943859e-05],
                [4.44997526e-05],
                [4.54793500e-05],
                [4.45020873e-05]]),
          'SNR': np.array([[5.36267166e+03],
                [2.68245365e+00],
                [4.26867480e-01],
                [6.80581028e-02]]),
          'freq': 3.564}
        }

        success4 = []
        report4 = []
        for spw in spws:
            # (h) IQUV flux densities of all three spws:              6.1.3
            successN, reportN = tstobj.check_fracdiff(spwstats[spw]['IQUV'], spwstats_613[spw]['IQUV'],    valname=f"Stokes Comparison (spw {spw}), Frac Diff IQUV vs 6.1.3")
            success4.append(successN)
            report4.append(reportN)
            # (i) IQUV flux densities of all three spws:              on-axis measurements
            successN, reportN = tstobj.check_fracdiff(spwstats[spw]['IQUV'], spwstats_onaxis[spw]['IQUV'], valname=f"Stokes Comparison (spw {spw}), Frac Diff IQUV vs on-axis")
            success4.append(successN)
            report4.append(reportN)
            # (j) Beam of all three spws:                             6.1.3
            successN, reportN = tstobj.check_fracdiff(spwstats[spw]['beam'], spwstats_613[spw]['beam'],    valname=f"Stokes Comparison (spw {spw}), Frac Diff Beam vs 6.1.3")
            success4.append(successN)
            report4.append(reportN)

        report  = "".join([report0, report1, report2, report3, *report4])
        success = success0 and success1 and success2 and success3 and all(success4) and th.check_final(report)
        casalog.post(f"{report}\nSuccess: {success}", "INFO")

        ###########################################################
        # %% Compare Expected Values [test_j1302_mosaic_cube] end @
        ###########################################################
        # not part of the jupyter scripts

        # (f) Runtimes not significantly different relative to previous runs
        # don't test this in jupyter notebooks - runtimes differ too much between machines
        success, report = tstobj.check_runtime(starttime, 1543, success, report)

        tstobj.assertTrue(success, msg=report)

    # Test 4
    @unittest.skipIf(ParallelTaskHelper.isMPIEnabled(), "Only run in serial, since John Tobin's only executed this test in serial (see 01/12/22 comment on CAS-12427).")
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
        # rundir = "/users/bbean/dev/CAS-12427/src/casalith/build-casalith/work/linux/test_vlass_j1302_QL_unittest"
        # os.system(f"mv {rundir}/run_results/VLASS* {rundir}/nosedir/test_vlass_1v2/")

        #########################################
        # %% Set local vars [test_j1302_ql] end @
        # .......................................

        # not part of the jupyter scripts
        starttime = datetime.now()

        # .......................................
        # %% Run tclean [test_j1302_ql] start   @
        #########################################

        def run_tclean(vis=tstobj.vis, intent='OBSERVE_TARGET#UNSPECIFIED', datacolumn='data',
                       imagename=None, niter=None, restoration=None, compare_tclean_pars=None,
                       phasecenter=tstobj.phasecenter, reffreq='3.0GHz', deconvolver='mtmfs',
                       cell='1.0arcsec', imsize=[7290,7290], gridder='mosaic',
                       restoringbeam='common', perchanweightdensity=False, mosweight=False,
                       scales=[0], weighting='briggs', robust=1.0):
            params = locals()
            params = {k:params[k] for k in filter(lambda x: x not in ['tstobj'], params.keys())}
            tstobj.run_tclean(**params)

        script_pars_vals_0 = None#tstobj.get_params_as_dict(vis='J1302-12fields.ms', selectdata=True, field='', spw='', timerange='', uvrange='', antenna='', scan='', observation='', intent='OBSERVE_TARGET#UNSPECIFIED', datacolumn='data', imagename='VLASS1.2.ql.T08t20.J1302.10.2048.v1.I.iter0', imsize=[7290, 7290], cell='1.0arcsec', phasecenter='13:03:13.874 -10.51.16.73', stokes='I', projection='SIN', startmodel='', specmode='mfs', reffreq='3.0GHz', nchan=-1, start='', width='', outframe='LSRK', veltype='radio', restfreq=[], interpolation='linear', perchanweightdensity=False, gridder='mosaic', facets=1, psfphasecenter='', chanchunks=1, wprojplanes=1, vptable='', mosweight=False, aterm=True, psterm=False, wbawp=True, conjbeams=False, cfcache='', usepointing=False, computepastep=360.0, rotatepastep=360.0, pointingoffsetsigdev=[], pblimit=0.2, normtype='flatnoise', deconvolver='mtmfs', scales=[0], nterms=2, smallscalebias=0.0, restoration=False, restoringbeam='common', pbcor=False, outlierfile='', weighting='briggs', robust=1.0, noise='1.0Jy', npixels=0, uvtaper=[], niter=0, gain=0.1, threshold='0.0mJy', nsigma=0.0, cycleniter=-1, cyclefactor=1.0, minpsffraction=0.05, maxpsffraction=0.8, interactive=0, usemask='user', mask='', pbmask=0.0, sidelobethreshold=3.0, noisethreshold=5.0, lownoisethreshold=1.5, negativethreshold=0.0, smoothfactor=1.0, minbeamfrac=0.3, cutthreshold=0.01, growiterations=75, dogrowprune=True, minpercentchange=-1.0, verbose=False, fastnoise=True, restart=True, savemodel='none', calcres=True, calcpsf=True, parallel=False)
        script_pars_vals_1 = None#tstobj.get_params_as_dict(vis='J1302-12fields.ms', selectdata=True, field='', spw='', timerange='', uvrange='', antenna='', scan='', observation='', intent='OBSERVE_TARGET#UNSPECIFIED', datacolumn='data', imagename='VLASS1.2.ql.T08t20.J1302.10.2048.v1.I.iter1', imsize=[7290, 7290], cell='1.0arcsec', phasecenter='13:03:13.874 -10.51.16.73', stokes='I', projection='SIN', startmodel='', specmode='mfs', reffreq='3.0GHz', nchan=-1, start='', width='', outframe='LSRK', veltype='radio', restfreq=[], interpolation='linear', perchanweightdensity=False, gridder='mosaic', facets=1, psfphasecenter='', chanchunks=1, wprojplanes=1, vptable='', mosweight=False, aterm=True, psterm=False, wbawp=True, conjbeams=False, cfcache='', usepointing=False, computepastep=360.0, rotatepastep=360.0, pointingoffsetsigdev=[], pblimit=0.2, normtype='flatnoise', deconvolver='mtmfs', scales=[0], nterms=2, smallscalebias=0.0, restoration=True, restoringbeam='common', pbcor=False, outlierfile='', weighting='briggs', robust=1.0, noise='1.0Jy', npixels=0, uvtaper=[], niter=20000, gain=0.1, threshold=0.0, nsigma=4.5, cycleniter=500, cyclefactor=2.0, minpsffraction=0.05, maxpsffraction=0.8, interactive=0, usemask='user', mask='', pbmask=0.0, sidelobethreshold=3.0, noisethreshold=5.0, lownoisethreshold=1.5, negativethreshold=0.0, smoothfactor=1.0, minbeamfrac=0.3, cutthreshold=0.01, growiterations=75, dogrowprune=True, minpercentchange=-1.0, verbose=False, fastnoise=True, restart=True, savemodel='none', calcres=False, calcpsf=False, parallel=False)
        run_tclean(imagename=img0, niter=0,     restoration=False,                                                                           compare_tclean_pars=script_pars_vals_0)
        for ext in ['.weight.tt2', '.weight.tt0', '.psf.tt0', '.residual.tt0', '.weight.tt1', '.sumwt.tt2', '.psf.tt1', '.residual.tt1', '.psf.tt2', '.sumwt.tt1', '.model.tt0', '.pb.tt0', '.model.tt1', '.sumwt.tt0']:
            shutil.copytree(src=img0+ext, dst=img1+ext)
        run_tclean(imagename=img1, niter=20000, restoration=True, nsigma=4.5, cycleniter=500, cyclefactor=2.0, calcres=False, calcpsf=False, compare_tclean_pars=script_pars_vals_1)

        ###########################################
        # %% Run tclean [test_j1302_ql] end       @
        # %% Prepare Images [test_j1302_ql] start @
        ###########################################

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
        onaxis_stats      = np.array([           0.3337])
        casa613_stats     = np.array([           0.320879])
        success1, report1 = tstobj.check_fracdiff(curr_stats, onaxis_stats, valname="Frac Diff F_nu vs. on-axis")
        success2, report2 = tstobj.check_fracdiff(curr_stats, casa613_stats, valname="Frac Diff F_nu vs. 6.1.3 image")

        # (b) tt1 vs 6.1.3, on-axis
        # no tt1 images for this test, skip

        # (c) alpha images
        # TODO
        # success3, report3 = ...

        # (d) beamsize comparison vs 6.1.3
        restbeam          = imhead(img1+'.image.pbcor.tt0.subim')['restoringbeam']
        beamstats_curr    = np.array([restbeam['major']['value'], restbeam['minor']['value'], restbeam['positionangle']['value']])
        beamstats_613     = np.array([3.16884375,                 2.59194756,                 11.36847878])
        success4, report4 = tstobj.check_fracdiff(beamstats_curr, beamstats_613, valname="Frac Diff Maj, Min, PA vs 6.1.3")

        # (e) Confirm presence of model column in resultant MS
        success5, report5 = tstobj.check_column_exists("MODEL_DATA")

        report  = "".join([report0, report1, report2, report4, report5])
        success = success1 and success2 and success4 and success5 and th.check_final(report)
        casalog.post(f"{report}\nSuccess: {success}", "INFO")

        ##################################################
        # %% Compare Expected Values [test_j1302_ql] end @
        ##################################################
        # not part of the jupyter scripts

        # (f) Runtimes not significantly different relative to previous runs
        # don't test this in jupyter notebooks - runtimes differ too much between machines
        success, report = tstobj.check_runtime(starttime, 1543, success, report)

        tstobj.assertTrue(success, msg=report)


##############################################
##############################################
class test_j1927(test_vlass_base):

    def setUp(self):
        super().setUp()
        self.vis = 'J1927_12fields.ms'
        self.phasecenter = '19:27:30.443 +61.17.32.898'
        self._clean_imgs_exist_dict()
        self.parallel = self._get_enable_parallel()

    # Test 1
    def test_j1927_mosaic_noncube(self):
        """ [j1927] test_j1927_mosaic_noncube """
        ######################################################################################
        # Should match values for "Stokes I" in the "Values to be compared"
        ######################################################################################
        # "noncube" to allow give this test a unique prefix, for running with runtest
        # For example: runtest.py -v test_vlass_1v2.py[test_j1927_mosaic_noncube]

        # not part of the jupyter scripts
        tstobj = self # jupyter equivalent: "tstobj = test_j1927()"

        #######################################################
        # %% Set local vars [test_j1927_mosaic_noncube] start @
        #######################################################

        #previous steps in the pipeline would have created mask files from catalogs and images that were created as an
        #intermediate pipeline step.
        tstobj.data_path_dir  = 'J1927/J1927-stakeholdertest-mosaic-data'
        img0 = 'J1927_iter2'
        tstobj.prepData("secondmask.mask", "QLcatmask.mask")
        # rundir = "/users/bbean/dev/CAS-12427/src/casalith/build-casalith/work/linux/test_vlass_j1927_mosaic_noncube_unittest"
        # os.system(f"mv {rundir}/run_results/VLASS* {rundir}/nosedir/test_vlass_1v2/")
        # os.system(f"mv {rundir}/*.ms {rundir}/nosedir/test_vlass_1v2/")
        # os.system(f"mv {rundir}*.mask {rundir}/nosedir/test_vlass_1v2/")

        #####################################################
        # %% Set local vars [test_j1927_mosaic_noncube] end @
        # ...................................................

        # not part of the jupyter scripts
        starttime = datetime.now()

        # ....................................................
        # %% Prepare masks [test_j1927_mosaic_noncube] start @
        ######################################################

        # combine first and 2nd order masks
        immath(imagename=['secondmask.mask','QLcatmask.mask'],expr='IM0+IM1',outfile='sum_of_masks.mask')
        im.mask(image='sum_of_masks.mask',mask='combined.mask',threshold=0.5)

        ####################################################
        # %% Prepare masks [test_j1927_mosaic_noncube] end @
        # %% Run tclean [test_j1927_mosaic_noncube] start  @
        ####################################################

        def run_tclean(vis=tstobj.vis, intent='OBSERVE_TARGET#UNSPECIFIED', uvrange='<12km',
                       niter=None, compare_tclean_pars=None, datacolumn=None,
                       imagename=img0, phasecenter=tstobj.phasecenter, reffreq='3.0GHz',
                       mask='', savemodel='none', usemask='user', pbmask=0.0,
                       deconvolver='mtmfs', cell='0.6arcsec', imsize=4000, gridder='mosaic',
                       uvtaper=[''], restoringbeam=[], mosweight=False, rotatepastep=5.0,
                       smallscalebias=0.4, pblimit=0.1, scales=[0], weighting='briggs',
                       robust=1.0, nsigma=2.0, cycleniter=500, cyclefactor=3.0,
                       interactive=0, calcres=False, calcpsf=False):
            params = locals()
            params = {k:params[k] for k in filter(lambda x: x not in ['tstobj'], params.keys())}
            tstobj.run_tclean(**params)

        script_pars_vals_0 = None#tstobj.get_params_as_dict(vis='J1927_12fields.ms', selectdata=True, field='', spw='', timerange='', uvrange='<12km', antenna='', scan='', observation='', intent='OBSERVE_TARGET#UNSPECIFIED', datacolumn='corrected', imagename='J1927_iter2', imsize=4000, cell='0.6arcsec', phasecenter='19:27:30.443 +61.17.32.898', stokes='I', projection='SIN', startmodel='', specmode='mfs', reffreq='3.0GHz', nchan=-1, start='', width='', outframe='LSRK', veltype='radio', restfreq=[], interpolation='linear', perchanweightdensity=True, gridder='mosaic', facets=1, psfphasecenter='', chanchunks=1, wprojplanes=1, vptable='', mosweight=False, aterm=True, psterm=False, wbawp=True, conjbeams=False, cfcache='', usepointing=False, computepastep=360.0, rotatepastep=5.0, pointingoffsetsigdev=[], pblimit=0.1, normtype='flatnoise', deconvolver='mtmfs', scales=[0], nterms=2, smallscalebias=0.4, restoration=True, restoringbeam=[], pbcor=False, outlierfile='', weighting='briggs', robust=1.0, noise='1.0Jy', npixels=0, uvtaper=[''], niter=0, gain=0.1, threshold=0.0, nsigma=2.0, cycleniter=500, cyclefactor=3.0, minpsffraction=0.05, maxpsffraction=0.8, interactive=False, usemask='user', mask='', pbmask=0.0, sidelobethreshold=3.0, noisethreshold=5.0, lownoisethreshold=1.5, negativethreshold=0.0, smoothfactor=1.0, minbeamfrac=0.3, cutthreshold=0.01, growiterations=75, dogrowprune=True, minpercentchange=-1.0, verbose=False, fastnoise=True, restart=True, savemodel='none', calcres=True, calcpsf=True, parallel=False)
        script_pars_vals_1 = None#tstobj.get_params_as_dict(vis='J1927_12fields.ms', selectdata=True, field='', spw='', timerange='', uvrange='<12km', antenna='', scan='', observation='', intent='OBSERVE_TARGET#UNSPECIFIED', datacolumn='corrected', imagename='J1927_iter2', imsize=4000, cell='0.6arcsec', phasecenter='19:27:30.443 +61.17.32.898', stokes='I', projection='SIN', startmodel='', specmode='mfs', reffreq='3.0GHz', nchan=-1, start='', width='', outframe='LSRK', veltype='radio', restfreq=[], interpolation='linear', perchanweightdensity=True, gridder='mosaic', facets=1, psfphasecenter='', chanchunks=1, wprojplanes=1, vptable='', mosweight=False, aterm=True, psterm=False, wbawp=True, conjbeams=False, cfcache='', usepointing=False, computepastep=360.0, rotatepastep=5.0, pointingoffsetsigdev=[], pblimit=0.1, normtype='flatnoise', deconvolver='mtmfs', scales=[0, 5, 12], nterms=2, smallscalebias=0.4, restoration=True, restoringbeam=[], pbcor=False, outlierfile='', weighting='briggs', robust=1.0, noise='1.0Jy', npixels=0, uvtaper=[''], niter=20000, gain=0.1, threshold=0.0, nsigma=3.0, cycleniter=500, cyclefactor=3.0, minpsffraction=0.05, maxpsffraction=0.8, interactive=False, usemask='user', mask='QLcatmask.mask', pbmask=0.0, sidelobethreshold=3.0, noisethreshold=5.0, lownoisethreshold=1.5, negativethreshold=0.0, smoothfactor=1.0, minbeamfrac=0.3, cutthreshold=0.01, growiterations=75, dogrowprune=True, minpercentchange=-1.0, verbose=False, fastnoise=True, restart=True, savemodel='none', calcres=False, calcpsf=False, parallel=False)
        script_pars_vals_2 = None#tstobj.get_params_as_dict(vis='J1927_12fields.ms', selectdata=True, field='', spw='', timerange='', uvrange='<12km', antenna='', scan='', observation='', intent='OBSERVE_TARGET#UNSPECIFIED', datacolumn='data', imagename='J1927_iter2', imsize=4000, cell='0.6arcsec', phasecenter='19:27:30.443 +61.17.32.898', stokes='I', projection='SIN', startmodel='', specmode='mfs', reffreq='3.0GHz', nchan=-1, start='', width='', outframe='LSRK', veltype='radio', restfreq=[], interpolation='linear', perchanweightdensity=True, gridder='mosaic', facets=1, psfphasecenter='', chanchunks=1, wprojplanes=1, vptable='', mosweight=False, aterm=True, psterm=False, wbawp=True, conjbeams=False, cfcache='', usepointing=False, computepastep=360.0, rotatepastep=5.0, pointingoffsetsigdev=[], pblimit=0.1, normtype='flatnoise', deconvolver='mtmfs', scales=[0], nterms=2, smallscalebias=0.4, restoration=True, restoringbeam=[], pbcor=False, outlierfile='', weighting='briggs', robust=1.0, noise='1.0Jy', npixels=0, uvtaper=[''], niter=0, gain=0.1, threshold=0.0, nsigma=2.0, cycleniter=500, cyclefactor=3.0, minpsffraction=0.05, maxpsffraction=0.8, interactive=False, usemask='user', mask='', pbmask=0.0, sidelobethreshold=3.0, noisethreshold=5.0, lownoisethreshold=1.5, negativethreshold=0.0, smoothfactor=1.0, minbeamfrac=0.3, cutthreshold=0.01, growiterations=75, dogrowprune=True, minpercentchange=-1.0, verbose=False, fastnoise=True, restart=True, savemodel='modelcolumn', calcres=False, calcpsf=False, parallel=False)
        script_pars_vals_3 = None#tstobj.get_params_as_dict(vis='J1927_12fields.ms', selectdata=True, field='', spw='', timerange='', uvrange='<12km', antenna='', scan='', observation='', intent='OBSERVE_TARGET#UNSPECIFIED', datacolumn='corrected', imagename='J1927_iter2', imsize=4000, cell='0.6arcsec', phasecenter='19:27:30.443 +61.17.32.898', stokes='I', projection='SIN', startmodel='', specmode='mfs', reffreq='3.0GHz', nchan=-1, start='', width='', outframe='LSRK', veltype='radio', restfreq=[], interpolation='linear', perchanweightdensity=True, gridder='mosaic', facets=1, psfphasecenter='', chanchunks=1, wprojplanes=1, vptable='', mosweight=False, aterm=True, psterm=False, wbawp=True, conjbeams=False, cfcache='', usepointing=False, computepastep=360.0, rotatepastep=5.0, pointingoffsetsigdev=[], pblimit=0.1, normtype='flatnoise', deconvolver='mtmfs', scales=[0, 5, 12], nterms=2, smallscalebias=0.4, restoration=True, restoringbeam=[], pbcor=False, outlierfile='', weighting='briggs', robust=1.0, noise='1.0Jy', npixels=0, uvtaper=[''], niter=20000, gain=0.1, threshold=0.0, nsigma=3.0, cycleniter=500, cyclefactor=3.0, minpsffraction=0.05, maxpsffraction=0.8, interactive=False, usemask='user', mask='combined.mask', pbmask=0.0, sidelobethreshold=3.0, noisethreshold=5.0, lownoisethreshold=1.5, negativethreshold=0.0, smoothfactor=1.0, minbeamfrac=0.3, cutthreshold=0.01, growiterations=75, dogrowprune=True, minpercentchange=-1.0, verbose=False, fastnoise=True, restart=True, savemodel='none', calcres=False, calcpsf=False, parallel=False)
        script_pars_vals_4 = None#tstobj.get_params_as_dict(vis='J1927_12fields.ms', selectdata=True, field='', spw='', timerange='', uvrange='<12km', antenna='', scan='', observation='', intent='OBSERVE_TARGET#UNSPECIFIED', datacolumn='corrected', imagename='J1927_iter2', imsize=4000, cell='0.6arcsec', phasecenter='19:27:30.443 +61.17.32.898', stokes='I', projection='SIN', startmodel='', specmode='mfs', reffreq='3.0GHz', nchan=-1, start='', width='', outframe='LSRK', veltype='radio', restfreq=[], interpolation='linear', perchanweightdensity=True, gridder='mosaic', facets=1, psfphasecenter='', chanchunks=1, wprojplanes=1, vptable='', mosweight=False, aterm=True, psterm=False, wbawp=True, conjbeams=False, cfcache='', usepointing=False, computepastep=360.0, rotatepastep=5.0, pointingoffsetsigdev=[], pblimit=0.1, normtype='flatnoise', deconvolver='mtmfs', scales=[0, 5, 12], nterms=2, smallscalebias=0.4, restoration=True, restoringbeam=[], pbcor=False, outlierfile='', weighting='briggs', robust=1.0, noise='1.0Jy', npixels=0, uvtaper=[''], niter=20000, gain=0.1, threshold=0.0, nsigma=4.5, cycleniter=100, cyclefactor=3.0, minpsffraction=0.05, maxpsffraction=0.8, interactive=False, usemask='pb', mask='', pbmask=0.4, sidelobethreshold=3.0, noisethreshold=5.0, lownoisethreshold=1.5, negativethreshold=0.0, smoothfactor=1.0, minbeamfrac=0.3, cutthreshold=0.01, growiterations=75, dogrowprune=True, minpercentchange=-1.0, verbose=False, fastnoise=True, restart=True, savemodel='none', calcres=False, calcpsf=False, parallel=False)

        # initialize iter2, no cleaning
        run_tclean( niter=0,     datacolumn='corrected', calcres=True, calcpsf=True,                         compare_tclean_pars=script_pars_vals_0 )

        # # resume iter2 with QL mask
        run_tclean( niter=20000, datacolumn='corrected', mask="QLcatmask.mask", nsigma=3.0, scales=[0,5,12], compare_tclean_pars=script_pars_vals_1 )

        # save model column, doesn't happen here in acutal VLASS pipeline, but makes sure functionality works.
        run_tclean( niter=0,     datacolumn='data',      savemodel='modelcolumn',                            compare_tclean_pars=script_pars_vals_2 )

        # resume iter2 with combined mask, remove old mask first, pass new mask as parameter
        os.system(f"rm -rf {img0}.mask")
        run_tclean( niter=20000, datacolumn='corrected', mask="combined.mask",  nsigma=3.0, scales=[0,5,12], compare_tclean_pars=script_pars_vals_3 )

        # resume iter2 with pbmask, removed old mask first then specify pbmask in resumption of tclean
        os.system(f"rm -rf {img0}.mask")
        run_tclean( niter=20000, datacolumn='corrected', usemask='pb', mask="", pbmask=0.4,   nsigma=4.5, scales=[0,5,12], cycleniter=100, compare_tclean_pars=script_pars_vals_4 )

        ################################################################
        # %% Run tclean [test_j1927_mosaic_noncube] end                @
        # %% Compare Expected Values [test_j1927_mosaic_noncube] start @
        ################################################################

        # (l) Ensure intermediate products exist, pbcor images, RMS image (made by imdev), and cutouts (.subim) from imsubimage
        # N/A for this test

        tt0stats   = imstat(imagename=img0+'.image.tt0', box='2000,2000,2000,2000')
        tt1stats   = imstat(imagename=img0+'.image.tt1', box='2000,2000,2000,2000')
        alphastats = imstat(imagename=img0+'.alpha',     box='2000,2000,2000,2000')
        curr_stats    = np.squeeze(np.array([ tt0stats['max'], tt1stats['max'], alphastats['max']]))
        onaxis_stats  = np.array([            0.9509,          0.3601,          0.3796])
        casa613_stats = np.array([            0.8887,          0.4148,          0.4668])

        # (a) tt0 vs 6.1.3, on-axis
        success0, report0 = tstobj.check_fracdiff(curr_stats[0], onaxis_stats[0],  valname="Frac Diff tt0 vs. on-axis")
        success1, report1 = tstobj.check_fracdiff(curr_stats[0], casa613_stats[0], valname="Frac Diff tt0 vs. 6.1.3 image")

        # (b) tt1 vs 6.1.3, on-axis
        success2, report2 = tstobj.check_fracdiff(curr_stats[1], onaxis_stats[1],  valname="Frac Diff tt1 vs. on-axis")
        success3, report3 = tstobj.check_fracdiff(curr_stats[1], casa613_stats[1], valname="Frac Diff tt1 vs. 6.1.3 image")

        # (c) alpha images
        success4, report4 = tstobj.check_fracdiff(curr_stats[2], onaxis_stats[2],  valname="Frac Diff alpha vs. on-axis")
        success5, report5 = tstobj.check_fracdiff(curr_stats[2], casa613_stats[2], valname="Frac Diff alpha vs. 6.1.3 image")

        # (d) beamsize comparison vs 6.1.3
        restbeam          = imhead(img0+'.image.tt0')['restoringbeam']
        beamstats_curr    = np.array([restbeam['major']['value'], restbeam['minor']['value'], restbeam['positionangle']['value']])
        beamstats_613     = np.array([2.46035814,                 2.05488539,                 -23.22724915])
        success6, report6 = tstobj.check_fracdiff(beamstats_curr, beamstats_613, valname="Frac Diff Maj, Min, PA vs 6.1.3")

        # (e) Confirm presence of model column in resultant MS
        success7, report7 = tstobj.check_column_exists("MODEL_DATA")

        report  = "".join([report0, report1, report2, report3, report4, report5, report6, report7])
        success = success0 and success1 and success2 and success3 and success4 and success5 and success6 and success7 and th.check_final(report)
        casalog.post(f"{report}\nSuccess: {success}", "INFO")

        ##############################################################
        # %% Compare Expected Values [test_j1927_mosaic_noncube] end @
        ##############################################################
        # not part of the jupyter scripts

        # (f) Runtimes not significantly different relative to previous runs
        # don't test this in jupyter notebooks - runtimes differ too much between machines
        success, report = tstobj.check_runtime(starttime, 1543, success, report)

        tstobj.assertTrue(success, msg=report)

    # N/A not implemented
    # def test_j1927_awproject(self):
    #     pass

    # Test 3
    @unittest.skipIf(ParallelTaskHelper.isMPIEnabled(), "Skip test. Tclean crashes with mpicasa+mosaic gridder+stokes imaging.")
    def test_j1927_mosaic_cube(self):
        """ [j1927] test_j1927_mosaic_cube """
        ######################################################################################
        # Should match values for "Cube" in the "Values to be compared"
        ######################################################################################
        # not part of the jupyter scripts
        tstobj = self # jupyter equivalent: "tstobj = test_j1927()"

        ####################################################
        # %% Set local vars [test_j1927_mosaic_cube] start @
        ####################################################

        #previous steps in the pipeline would have created mask files from catalogs and images that were created as an
        #intermediate pipeline step.
        tstobj.data_path_dir  = 'J1927/J1927-stakeholdertest-mosaic-cube-data'
        tstobj.prepData("QLcatmask.mask", "combined.mask")
        # rundir = "/users/bbean/dev/CAS-12427/src/casalith/build-casalith/work/linux/test_vlass_j1927_cube_unittest"
        # os.system(f"mv {rundir}/run_results/VLASS* {rundir}/nosedir/test_vlass_1v2/")

        # reference frequence to use per spectral window (spw)
        refFreqDict  = {
            '2' :  '2.028GHz',
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

        ##################################################
        # %% Set local vars [test_j1927_mosaic_cube] end @
        # .......................................

        # not part of the jupyter scripts
        starttime = datetime.now()

        # .......................................
        # %% Run tclean [test_j1927_mosaic_cube] start   @
        ##################################################

        def iname(image_iter, spw, stokes): # imagename=imagename_base+image_iter+'_'+spw.replace('~','-')+'_'+stokes
            return 'J1927_'+image_iter+'_'+spw.replace('~','-')+'_'+stokes

        def run_tclean(vis=tstobj.vis, uvrange='<12km', imsize=4000, intent='OBSERVE_TARGET#UNSPECIFIED',
                       imagename=None, niter=None, compare_tclean_pars=None,
                       cell='0.6arcsec', datacolumn='data', phasecenter=tstobj.phasecenter, spw='',
                       stokes='I', mask='', usemask='user', pbmask=0.0, reffreq='3.0GHz',
                       deconvolver='mtmfs', gridder='mosaic', restoringbeam=[], mosweight=False,
                       rotatepastep=5.0, pblimit=0.1, scales=[0], nterms=1, weighting='briggs',
                       smallscalebias=0.4, robust=1.0, uvtaper=[''], nsigma=2.0, calcres=True,
                       calcpsf=True, cycleniter=500, cyclefactor=3, interactive=0):
            params = locals()
            params = {k:params[k] for k in filter(lambda x: x not in ['tstobj'], params.keys())}
            tstobj.run_tclean(**params)

        spws         = [ '2','8','14']
        stokesParams = ['IQUV']
        spwstats     = { '2':  {'freq': 2.028, 'IQUV': np.array([0.0,0.0,0.0,0.0]), 'beam': np.array([0.0,0.0,0.0])},
                         '8':  {'freq': 2.796, 'IQUV': np.array([0.0,0.0,0.0,0.0]), 'beam': np.array([0.0,0.0,0.0])},
                         '14': {'freq': 3.594, 'IQUV': np.array([0.0,0.0,0.0,0.0]), 'beam': np.array([0.0,0.0,0.0])} }

        script_pars_vals_0 = {
            '2':  { 'IQUV': None },#tstobj.get_params_as_dict(vis='J1927_12fields.ms', selectdata=True, field='', spw='2', timerange='', uvrange='<12km', antenna='', scan='', observation='', intent='OBSERVE_TARGET#UNSPECIFIED', datacolumn='corrected', imagename='J1927_iter2_2_IQUV', imsize=4000, cell='0.6arcsec', phasecenter='19:27:30.443 +61.17.32.898', stokes='IQUV', projection='SIN', startmodel='', specmode='mfs', reffreq='2.028GHz', nchan=-1, start='', width='', outframe='LSRK', veltype='radio', restfreq=[], interpolation='linear', perchanweightdensity=True, gridder='mosaic', facets=1, psfphasecenter='', chanchunks=1, wprojplanes=1, vptable='', mosweight=False, aterm=True, psterm=False, wbawp=True, conjbeams=False, cfcache='', usepointing=False, computepastep=360.0, rotatepastep=5.0, pointingoffsetsigdev=[], pblimit=0.1, normtype='flatnoise', deconvolver='mtmfs', scales=[0], nterms=1, smallscalebias=0.4, restoration=True, restoringbeam=[], pbcor=False, outlierfile='', weighting='briggs', robust=1.0, noise='1.0Jy', npixels=0, uvtaper=[''], niter=0, gain=0.1, threshold=0.0, nsigma=2.0, cycleniter=500, cyclefactor=3.0, minpsffraction=0.05, maxpsffraction=0.8, interactive=False, usemask='user', mask='', pbmask=0.0, sidelobethreshold=3.0, noisethreshold=5.0, lownoisethreshold=1.5, negativethreshold=0.0, smoothfactor=1.0, minbeamfrac=0.3, cutthreshold=0.01, growiterations=75, dogrowprune=True, minpercentchange=-1.0, verbose=False, fastnoise=True, restart=True, savemodel='none', calcres=True, calcpsf=True, parallel=False) },
            '8':  { 'IQUV': None },#tstobj.get_params_as_dict(vis='J1927_12fields.ms', selectdata=True, field='', spw='8', timerange='', uvrange='<12km', antenna='', scan='', observation='', intent='OBSERVE_TARGET#UNSPECIFIED', datacolumn='corrected', imagename='J1927_iter2_8_IQUV', imsize=4000, cell='0.6arcsec', phasecenter='19:27:30.443 +61.17.32.898', stokes='IQUV', projection='SIN', startmodel='', specmode='mfs', reffreq='2.796GHz', nchan=-1, start='', width='', outframe='LSRK', veltype='radio', restfreq=[], interpolation='linear', perchanweightdensity=True, gridder='mosaic', facets=1, psfphasecenter='', chanchunks=1, wprojplanes=1, vptable='', mosweight=False, aterm=True, psterm=False, wbawp=True, conjbeams=False, cfcache='', usepointing=False, computepastep=360.0, rotatepastep=5.0, pointingoffsetsigdev=[], pblimit=0.1, normtype='flatnoise', deconvolver='mtmfs', scales=[0], nterms=1, smallscalebias=0.4, restoration=True, restoringbeam=[], pbcor=False, outlierfile='', weighting='briggs', robust=1.0, noise='1.0Jy', npixels=0, uvtaper=[''], niter=0, gain=0.1, threshold=0.0, nsigma=2.0, cycleniter=500, cyclefactor=3.0, minpsffraction=0.05, maxpsffraction=0.8, interactive=False, usemask='user', mask='', pbmask=0.0, sidelobethreshold=3.0, noisethreshold=5.0, lownoisethreshold=1.5, negativethreshold=0.0, smoothfactor=1.0, minbeamfrac=0.3, cutthreshold=0.01, growiterations=75, dogrowprune=True, minpercentchange=-1.0, verbose=False, fastnoise=True, restart=True, savemodel='none', calcres=True, calcpsf=True, parallel=False) },
            '14': { 'IQUV': None },#tstobj.get_params_as_dict(vis='J1927_12fields.ms', selectdata=True, field='', spw='14', timerange='', uvrange='<12km', antenna='', scan='', observation='', intent='OBSERVE_TARGET#UNSPECIFIED', datacolumn='corrected', imagename='J1927_iter2_14_IQUV', imsize=4000, cell='0.6arcsec', phasecenter='19:27:30.443 +61.17.32.898', stokes='IQUV', projection='SIN', startmodel='', specmode='mfs', reffreq='3.564GHz', nchan=-1, start='', width='', outframe='LSRK', veltype='radio', restfreq=[], interpolation='linear', perchanweightdensity=True, gridder='mosaic', facets=1, psfphasecenter='', chanchunks=1, wprojplanes=1, vptable='', mosweight=False, aterm=True, psterm=False, wbawp=True, conjbeams=False, cfcache='', usepointing=False, computepastep=360.0, rotatepastep=5.0, pointingoffsetsigdev=[], pblimit=0.1, normtype='flatnoise', deconvolver='mtmfs', scales=[0], nterms=1, smallscalebias=0.4, restoration=True, restoringbeam=[], pbcor=False, outlierfile='', weighting='briggs', robust=1.0, noise='1.0Jy', npixels=0, uvtaper=[''], niter=0, gain=0.1, threshold=0.0, nsigma=2.0, cycleniter=500, cyclefactor=3.0, minpsffraction=0.05, maxpsffraction=0.8, interactive=False, usemask='user', mask='', pbmask=0.0, sidelobethreshold=3.0, noisethreshold=5.0, lownoisethreshold=1.5, negativethreshold=0.0, smoothfactor=1.0, minbeamfrac=0.3, cutthreshold=0.01, growiterations=75, dogrowprune=True, minpercentchange=-1.0, verbose=False, fastnoise=True, restart=True, savemodel='none', calcres=True, calcpsf=True, parallel=False) },
        }
        script_pars_vals_1 = {
            '2':  { 'IQUV': None },#tstobj.get_params_as_dict(vis='J1927_12fields.ms', selectdata=True, field='', spw='2', timerange='', uvrange='<12km', antenna='', scan='', observation='', intent='OBSERVE_TARGET#UNSPECIFIED', datacolumn='corrected', imagename='J1927_iter2_2_IQUV', imsize=4000, cell='0.6arcsec', phasecenter='19:27:30.443 +61.17.32.898', stokes='IQUV', projection='SIN', startmodel='', specmode='mfs', reffreq='2.028GHz', nchan=-1, start='', width='', outframe='LSRK', veltype='radio', restfreq=[], interpolation='linear', perchanweightdensity=True, gridder='mosaic', facets=1, psfphasecenter='', chanchunks=1, wprojplanes=1, vptable='', mosweight=False, aterm=True, psterm=False, wbawp=True, conjbeams=False, cfcache='', usepointing=False, computepastep=360.0, rotatepastep=5.0, pointingoffsetsigdev=[], pblimit=0.1, normtype='flatnoise', deconvolver='mtmfs', scales=[0, 5, 12], nterms=1, smallscalebias=0.4, restoration=True, restoringbeam=[], pbcor=False, outlierfile='', weighting='briggs', robust=1.0, noise='1.0Jy', npixels=0, uvtaper=[''], niter=20000, gain=0.1, threshold=0.0, nsigma=3.0, cycleniter=500, cyclefactor=3.0, minpsffraction=0.05, maxpsffraction=0.8, interactive=False, usemask='user', mask='QLcatmask.mask', pbmask=0.0, sidelobethreshold=3.0, noisethreshold=5.0, lownoisethreshold=1.5, negativethreshold=0.0, smoothfactor=1.0, minbeamfrac=0.3, cutthreshold=0.01, growiterations=75, dogrowprune=True, minpercentchange=-1.0, verbose=False, fastnoise=True, restart=True, savemodel='none', calcres=False, calcpsf=False, parallel=False) },
            '8':  { 'IQUV': None },#tstobj.get_params_as_dict(vis='J1927_12fields.ms', selectdata=True, field='', spw='8', timerange='', uvrange='<12km', antenna='', scan='', observation='', intent='OBSERVE_TARGET#UNSPECIFIED', datacolumn='corrected', imagename='J1927_iter2_8_IQUV', imsize=4000, cell='0.6arcsec', phasecenter='19:27:30.443 +61.17.32.898', stokes='IQUV', projection='SIN', startmodel='', specmode='mfs', reffreq='2.796GHz', nchan=-1, start='', width='', outframe='LSRK', veltype='radio', restfreq=[], interpolation='linear', perchanweightdensity=True, gridder='mosaic', facets=1, psfphasecenter='', chanchunks=1, wprojplanes=1, vptable='', mosweight=False, aterm=True, psterm=False, wbawp=True, conjbeams=False, cfcache='', usepointing=False, computepastep=360.0, rotatepastep=5.0, pointingoffsetsigdev=[], pblimit=0.1, normtype='flatnoise', deconvolver='mtmfs', scales=[0, 5, 12], nterms=1, smallscalebias=0.4, restoration=True, restoringbeam=[], pbcor=False, outlierfile='', weighting='briggs', robust=1.0, noise='1.0Jy', npixels=0, uvtaper=[''], niter=20000, gain=0.1, threshold=0.0, nsigma=3.0, cycleniter=500, cyclefactor=3.0, minpsffraction=0.05, maxpsffraction=0.8, interactive=False, usemask='user', mask='QLcatmask.mask', pbmask=0.0, sidelobethreshold=3.0, noisethreshold=5.0, lownoisethreshold=1.5, negativethreshold=0.0, smoothfactor=1.0, minbeamfrac=0.3, cutthreshold=0.01, growiterations=75, dogrowprune=True, minpercentchange=-1.0, verbose=False, fastnoise=True, restart=True, savemodel='none', calcres=False, calcpsf=False, parallel=False) },
            '14': { 'IQUV': None },#tstobj.get_params_as_dict(vis='J1927_12fields.ms', selectdata=True, field='', spw='14', timerange='', uvrange='<12km', antenna='', scan='', observation='', intent='OBSERVE_TARGET#UNSPECIFIED', datacolumn='corrected', imagename='J1927_iter2_14_IQUV', imsize=4000, cell='0.6arcsec', phasecenter='19:27:30.443 +61.17.32.898', stokes='IQUV', projection='SIN', startmodel='', specmode='mfs', reffreq='3.564GHz', nchan=-1, start='', width='', outframe='LSRK', veltype='radio', restfreq=[], interpolation='linear', perchanweightdensity=True, gridder='mosaic', facets=1, psfphasecenter='', chanchunks=1, wprojplanes=1, vptable='', mosweight=False, aterm=True, psterm=False, wbawp=True, conjbeams=False, cfcache='', usepointing=False, computepastep=360.0, rotatepastep=5.0, pointingoffsetsigdev=[], pblimit=0.1, normtype='flatnoise', deconvolver='mtmfs', scales=[0, 5, 12], nterms=1, smallscalebias=0.4, restoration=True, restoringbeam=[], pbcor=False, outlierfile='', weighting='briggs', robust=1.0, noise='1.0Jy', npixels=0, uvtaper=[''], niter=20000, gain=0.1, threshold=0.0, nsigma=3.0, cycleniter=500, cyclefactor=3.0, minpsffraction=0.05, maxpsffraction=0.8, interactive=False, usemask='user', mask='QLcatmask.mask', pbmask=0.0, sidelobethreshold=3.0, noisethreshold=5.0, lownoisethreshold=1.5, negativethreshold=0.0, smoothfactor=1.0, minbeamfrac=0.3, cutthreshold=0.01, growiterations=75, dogrowprune=True, minpercentchange=-1.0, verbose=False, fastnoise=True, restart=True, savemodel='none', calcres=False, calcpsf=False, parallel=False) },
        }
        script_pars_vals_2 = {
            '2':  { 'IQUV': None },#tstobj.get_params_as_dict(vis='J1927_12fields.ms', selectdata=True, field='', spw='2', timerange='', uvrange='<12km', antenna='', scan='', observation='', intent='OBSERVE_TARGET#UNSPECIFIED', datacolumn='corrected', imagename='J1927_iter2_2_IQUV', imsize=4000, cell='0.6arcsec', phasecenter='19:27:30.443 +61.17.32.898', stokes='IQUV', projection='SIN', startmodel='', specmode='mfs', reffreq='2.028GHz', nchan=-1, start='', width='', outframe='LSRK', veltype='radio', restfreq=[], interpolation='linear', perchanweightdensity=True, gridder='mosaic', facets=1, psfphasecenter='', chanchunks=1, wprojplanes=1, vptable='', mosweight=False, aterm=True, psterm=False, wbawp=True, conjbeams=False, cfcache='', usepointing=False, computepastep=360.0, rotatepastep=5.0, pointingoffsetsigdev=[], pblimit=0.1, normtype='flatnoise', deconvolver='mtmfs', scales=[0, 5, 12], nterms=1, smallscalebias=0.4, restoration=True, restoringbeam=[], pbcor=False, outlierfile='', weighting='briggs', robust=1.0, noise='1.0Jy', npixels=0, uvtaper=[''], niter=20000, gain=0.1, threshold=0.0, nsigma=3.0, cycleniter=500, cyclefactor=3.0, minpsffraction=0.05, maxpsffraction=0.8, interactive=False, usemask='user', mask='combined.mask', pbmask=0.0, sidelobethreshold=3.0, noisethreshold=5.0, lownoisethreshold=1.5, negativethreshold=0.0, smoothfactor=1.0, minbeamfrac=0.3, cutthreshold=0.01, growiterations=75, dogrowprune=True, minpercentchange=-1.0, verbose=False, fastnoise=True, restart=True, savemodel='none', calcres=False, calcpsf=False, parallel=False) },
            '8':  { 'IQUV': None },#tstobj.get_params_as_dict(vis='J1927_12fields.ms', selectdata=True, field='', spw='8', timerange='', uvrange='<12km', antenna='', scan='', observation='', intent='OBSERVE_TARGET#UNSPECIFIED', datacolumn='corrected', imagename='J1927_iter2_8_IQUV', imsize=4000, cell='0.6arcsec', phasecenter='19:27:30.443 +61.17.32.898', stokes='IQUV', projection='SIN', startmodel='', specmode='mfs', reffreq='2.796GHz', nchan=-1, start='', width='', outframe='LSRK', veltype='radio', restfreq=[], interpolation='linear', perchanweightdensity=True, gridder='mosaic', facets=1, psfphasecenter='', chanchunks=1, wprojplanes=1, vptable='', mosweight=False, aterm=True, psterm=False, wbawp=True, conjbeams=False, cfcache='', usepointing=False, computepastep=360.0, rotatepastep=5.0, pointingoffsetsigdev=[], pblimit=0.1, normtype='flatnoise', deconvolver='mtmfs', scales=[0, 5, 12], nterms=1, smallscalebias=0.4, restoration=True, restoringbeam=[], pbcor=False, outlierfile='', weighting='briggs', robust=1.0, noise='1.0Jy', npixels=0, uvtaper=[''], niter=20000, gain=0.1, threshold=0.0, nsigma=3.0, cycleniter=500, cyclefactor=3.0, minpsffraction=0.05, maxpsffraction=0.8, interactive=False, usemask='user', mask='combined.mask', pbmask=0.0, sidelobethreshold=3.0, noisethreshold=5.0, lownoisethreshold=1.5, negativethreshold=0.0, smoothfactor=1.0, minbeamfrac=0.3, cutthreshold=0.01, growiterations=75, dogrowprune=True, minpercentchange=-1.0, verbose=False, fastnoise=True, restart=True, savemodel='none', calcres=False, calcpsf=False, parallel=False) },
            '14': { 'IQUV': None },#tstobj.get_params_as_dict(vis='J1927_12fields.ms', selectdata=True, field='', spw='14', timerange='', uvrange='<12km', antenna='', scan='', observation='', intent='OBSERVE_TARGET#UNSPECIFIED', datacolumn='corrected', imagename='J1927_iter2_14_IQUV', imsize=4000, cell='0.6arcsec', phasecenter='19:27:30.443 +61.17.32.898', stokes='IQUV', projection='SIN', startmodel='', specmode='mfs', reffreq='3.564GHz', nchan=-1, start='', width='', outframe='LSRK', veltype='radio', restfreq=[], interpolation='linear', perchanweightdensity=True, gridder='mosaic', facets=1, psfphasecenter='', chanchunks=1, wprojplanes=1, vptable='', mosweight=False, aterm=True, psterm=False, wbawp=True, conjbeams=False, cfcache='', usepointing=False, computepastep=360.0, rotatepastep=5.0, pointingoffsetsigdev=[], pblimit=0.1, normtype='flatnoise', deconvolver='mtmfs', scales=[0, 5, 12], nterms=1, smallscalebias=0.4, restoration=True, restoringbeam=[], pbcor=False, outlierfile='', weighting='briggs', robust=1.0, noise='1.0Jy', npixels=0, uvtaper=[''], niter=20000, gain=0.1, threshold=0.0, nsigma=3.0, cycleniter=500, cyclefactor=3.0, minpsffraction=0.05, maxpsffraction=0.8, interactive=False, usemask='user', mask='combined.mask', pbmask=0.0, sidelobethreshold=3.0, noisethreshold=5.0, lownoisethreshold=1.5, negativethreshold=0.0, smoothfactor=1.0, minbeamfrac=0.3, cutthreshold=0.01, growiterations=75, dogrowprune=True, minpercentchange=-1.0, verbose=False, fastnoise=True, restart=True, savemodel='none', calcres=False, calcpsf=False, parallel=False) },
        }
        script_pars_vals_3 = {
            '2':  { 'IQUV': None },#tstobj.get_params_as_dict(vis='J1927_12fields.ms', selectdata=True, field='', spw='2', timerange='', uvrange='<12km', antenna='', scan='', observation='', intent='OBSERVE_TARGET#UNSPECIFIED', datacolumn='corrected', imagename='J1927_iter2_2_IQUV', imsize=4000, cell='0.6arcsec', phasecenter='19:27:30.443 +61.17.32.898', stokes='IQUV', projection='SIN', startmodel='', specmode='mfs', reffreq='2.028GHz', nchan=-1, start='', width='', outframe='LSRK', veltype='radio', restfreq=[], interpolation='linear', perchanweightdensity=True, gridder='mosaic', facets=1, psfphasecenter='', chanchunks=1, wprojplanes=1, vptable='', mosweight=False, aterm=True, psterm=False, wbawp=True, conjbeams=False, cfcache='', usepointing=False, computepastep=360.0, rotatepastep=5.0, pointingoffsetsigdev=[], pblimit=0.1, normtype='flatnoise', deconvolver='mtmfs', scales=[0, 5, 12], nterms=1, smallscalebias=0.4, restoration=True, restoringbeam=[], pbcor=False, outlierfile='', weighting='briggs', robust=1.0, noise='1.0Jy', npixels=0, uvtaper=[''], niter=20000, gain=0.1, threshold=0.0, nsigma=4.5, cycleniter=100, cyclefactor=3.0, minpsffraction=0.05, maxpsffraction=0.8, interactive=False, usemask='pb', mask='', pbmask=0.4, sidelobethreshold=3.0, noisethreshold=5.0, lownoisethreshold=1.5, negativethreshold=0.0, smoothfactor=1.0, minbeamfrac=0.3, cutthreshold=0.01, growiterations=75, dogrowprune=True, minpercentchange=-1.0, verbose=False, fastnoise=True, restart=True, savemodel='none', calcres=False, calcpsf=False, parallel=False) },
            '8':  { 'IQUV': None },#tstobj.get_params_as_dict(vis='J1927_12fields.ms', selectdata=True, field='', spw='8', timerange='', uvrange='<12km', antenna='', scan='', observation='', intent='OBSERVE_TARGET#UNSPECIFIED', datacolumn='corrected', imagename='J1927_iter2_8_IQUV', imsize=4000, cell='0.6arcsec', phasecenter='19:27:30.443 +61.17.32.898', stokes='IQUV', projection='SIN', startmodel='', specmode='mfs', reffreq='2.796GHz', nchan=-1, start='', width='', outframe='LSRK', veltype='radio', restfreq=[], interpolation='linear', perchanweightdensity=True, gridder='mosaic', facets=1, psfphasecenter='', chanchunks=1, wprojplanes=1, vptable='', mosweight=False, aterm=True, psterm=False, wbawp=True, conjbeams=False, cfcache='', usepointing=False, computepastep=360.0, rotatepastep=5.0, pointingoffsetsigdev=[], pblimit=0.1, normtype='flatnoise', deconvolver='mtmfs', scales=[0, 5, 12], nterms=1, smallscalebias=0.4, restoration=True, restoringbeam=[], pbcor=False, outlierfile='', weighting='briggs', robust=1.0, noise='1.0Jy', npixels=0, uvtaper=[''], niter=20000, gain=0.1, threshold=0.0, nsigma=4.5, cycleniter=100, cyclefactor=3.0, minpsffraction=0.05, maxpsffraction=0.8, interactive=False, usemask='pb', mask='', pbmask=0.4, sidelobethreshold=3.0, noisethreshold=5.0, lownoisethreshold=1.5, negativethreshold=0.0, smoothfactor=1.0, minbeamfrac=0.3, cutthreshold=0.01, growiterations=75, dogrowprune=True, minpercentchange=-1.0, verbose=False, fastnoise=True, restart=True, savemodel='none', calcres=False, calcpsf=False, parallel=False) },
            '14': { 'IQUV': None },#tstobj.get_params_as_dict(vis='J1927_12fields.ms', selectdata=True, field='', spw='14', timerange='', uvrange='<12km', antenna='', scan='', observation='', intent='OBSERVE_TARGET#UNSPECIFIED', datacolumn='corrected', imagename='J1927_iter2_14_IQUV', imsize=4000, cell='0.6arcsec', phasecenter='19:27:30.443 +61.17.32.898', stokes='IQUV', projection='SIN', startmodel='', specmode='mfs', reffreq='3.564GHz', nchan=-1, start='', width='', outframe='LSRK', veltype='radio', restfreq=[], interpolation='linear', perchanweightdensity=True, gridder='mosaic', facets=1, psfphasecenter='', chanchunks=1, wprojplanes=1, vptable='', mosweight=False, aterm=True, psterm=False, wbawp=True, conjbeams=False, cfcache='', usepointing=False, computepastep=360.0, rotatepastep=5.0, pointingoffsetsigdev=[], pblimit=0.1, normtype='flatnoise', deconvolver='mtmfs', scales=[0, 5, 12], nterms=1, smallscalebias=0.4, restoration=True, restoringbeam=[], pbcor=False, outlierfile='', weighting='briggs', robust=1.0, noise='1.0Jy', npixels=0, uvtaper=[''], niter=20000, gain=0.1, threshold=0.0, nsigma=4.5, cycleniter=100, cyclefactor=3.0, minpsffraction=0.05, maxpsffraction=0.8, interactive=False, usemask='pb', mask='', pbmask=0.4, sidelobethreshold=3.0, noisethreshold=5.0, lownoisethreshold=1.5, negativethreshold=0.0, smoothfactor=1.0, minbeamfrac=0.3, cutthreshold=0.01, growiterations=75, dogrowprune=True, minpercentchange=-1.0, verbose=False, fastnoise=True, restart=True, savemodel='none', calcres=False, calcpsf=False, parallel=False) },
        }

        for spw in spws:
           for stokes in stokesParams:
              # initialize iter2, no cleaning
              image_iter='iter2'
              imagename = iname(image_iter, spw, stokes)
              run_tclean( imagename=imagename, datacolumn='corrected', niter=0, spw=spw,
                          stokes=stokes, reffreq=refFreqDict[spw], compare_tclean_pars=script_pars_vals_0[spw][stokes] )

              # # resume iter2 with QL mask
              run_tclean( imagename=imagename, datacolumn='corrected', scales=[0,5,12], nsigma=3.0, niter=20000, cycleniter=500,   
                          mask="QLcatmask.mask", calcres=False, calcpsf=False, spw=spw,
                          stokes=stokes, reffreq=refFreqDict[spw], compare_tclean_pars=script_pars_vals_1[spw][stokes] )

              # resume iter2 with combined mask
              # os.system('rm -rf *.workdirectory')
              # os.system('rm -rf *iter2*.mask')
              run_tclean( imagename=imagename, datacolumn='corrected', scales=[0,5,12], nsigma=3.0, niter=20000, cycleniter=500,
                          mask="combined.mask", calcres=False, calcpsf=False, spw=spw,
                          stokes=stokes, reffreq=refFreqDict[spw], compare_tclean_pars=script_pars_vals_2[spw][stokes])

              # os.system('rm -rf iter2*.mask')
              run_tclean( imagename=imagename, datacolumn='corrected', scales=[0,5,12], nsigma=4.5, niter=20000, cycleniter=100,
                          mask="", calcres=False, calcpsf=False, usemask='pb', pbmask=0.4, spw=spw,
                          stokes=stokes, reffreq=refFreqDict[spw], compare_tclean_pars=script_pars_vals_3[spw][stokes] )

              tt0statsI=imstat(imagename=imagename+'.image.tt0',box='2000,2000,2000,2000',stokes='I')
              tt0statsQ=imstat(imagename=imagename+'.image.tt0',box='2000,2000,2000,2000',stokes='Q')
              tt0statsU=imstat(imagename=imagename+'.image.tt0',box='2000,2000,2000,2000',stokes='U')
              tt0statsV=imstat(imagename=imagename+'.image.tt0',box='2000,2000,2000,2000',stokes='V')
              spwstats[spw]['IQUV']=np.array([tt0statsI['max'],tt0statsQ['max'],tt0statsU['max'],tt0statsV['max']])
              header=imhead(imagename+'.psf.tt0')['perplanebeams']['beams']['*0']['*0']
              beamstats=np.array([header['major']['value'],header['minor']['value'],header['positionangle']['value']])
              spwstats[spw]['beam']=beamstats

        ####################################################
        # %% Run tclean [test_j1927_mosaic_cube] end       @
        # %% Math stuff [test_j1927_mosaic_cube] start     @
        ####################################################

        spwlist=list(spwstats.keys())
        nspws=len(spwlist)
        freqs=np.zeros(nspws)
        fluxes=np.zeros(nspws)
        for i in range(nspws):
           freqs[i]=spwstats[spwlist[i]]['freq']
           fluxes[i]=spwstats[spwlist[i]]['IQUV'][0]

        logfreqs=np.log10(freqs)
        logfluxes=np.log10(fluxes)

        from scipy.optimize import curve_fit

        def func(x, a, b):
            nu_0=3.0
            return a*(x-np.log10(nu_0))+b

        popt, pcov = curve_fit(func, logfreqs, logfluxes)
        perr = np.sqrt(np.diag(pcov))

        #############################################################
        # %% Math stuff [test_j1927_mosaic_cube] end                @
        # %% Compare Expected Values [test_j1927_mosaic_cube] start @
        #############################################################

        # (l) Ensure intermediate products exist, pbcor images, RMS image (made by imdev), and cutouts (.subim) from imsubimage
        # N/A: no pbcore, rms, or subim images are created for this test

        # (g) Fit F_nu0 and Alpha from three cube planes and compare: 6.1.3, on-axis
        # compare to alpha (ground truth), and 6.1.3 (fitted for spws 2, 8, 14 from mosaic gridder in CASA 6.1.3)
        alpha = popt[0]
        f_nu0 = 10**popt[1]
        curr_stats        = np.squeeze(np.array([f_nu0,   alpha])) # [flux density, alpha]
        onaxis_stats      = np.array([           0.3337, -0.0476])
        casa613_stats     = np.array([           0.88879, 0.4127])
        success0, report0 = tstobj.check_fracdiff(curr_stats[0], onaxis_stats[0],  valname="Frac Diff F_nu, alpha (on-axis)")
        success1, report1 = tstobj.check_fracdiff(curr_stats[0], casa613_stats[0], valname="Frac Diff F_nu, alpha (6.1.3 image)")
        success2, report2 = tstobj.check_fracdiff(curr_stats[1], onaxis_stats[1],  valname="Frac Diff alpha (on-axis)")
        success3, report3 = tstobj.check_fracdiff(curr_stats[1], casa613_stats[1], valname="Frac Diff alpha (6.1.3 image)")

        spwstats_613= {
            '2': {
                'freq': 2.028,
                'IQUV': np.array([[ 0.75571942],
                                  [ 0.00576184],
                                  [ 0.00080162],
                                  [-0.00630026]]),
                'beam': np.array([  3.65375686,   3.0324676 , -22.7224865 ])},
            '8': {
                'freq': 2.796,
                'IQUV': np.array([[ 0.86450773],
                                  [ 0.00522543],
                                  [-0.00604387],
                                  [-0.00717182]]),
                'beam': np.array([  2.66741753,   2.23318672, -26.65664101])},
            '14': {
                'freq': 3.594,
                'IQUV': np.array([[ 0.95684659],
                                  [ 0.00543807],
                                  [-0.00958768],
                                  [-0.00369545]]),
                'beam': np.array([  2.0738709 ,   1.70000803, -27.73123169])
            }
        }

        success4 = []
        report4 = []
        for spw in spws:
            # (h) IQUV flux densities of all three spws:              6.1.3
            successN, reportN = tstobj.check_fracdiff(spwstats[spw]['IQUV'], spwstats_613[spw]['IQUV'], valname=f"Stokes Comparison (spw {spw}), Frac Diff IQUV vs 6.1.3")
            success4.append(successN)
            report4.append(reportN)
            # (i) IQUV flux densities of all three spws:              on-axis measurements
            # N/A: no no-axis measurements available in VLASS_mosaic_cube_stakeholder_test_script.py
            # (j) Beam of all three spws:                             6.1.3
            successN, reportN = tstobj.check_fracdiff(spwstats[spw]['beam'], spwstats_613[spw]['beam'], valname=f"Stokes Comparison (spw {spw}), Frac Diff Beam vs 6.1.3")
            success4.append(successN)
            report4.append(reportN)

        report  = "".join([report0, report1, report2, report3, *report4])
        success = success0 and success1 and success2 and success3 and all(success4) and th.check_final(report)
        casalog.post(f"{report}\nSuccess: {success}", "INFO")

        ###########################################################
        # %% Compare Expected Values [test_j1927_mosaic_cube] end @
        ###########################################################
        # not part of the jupyter scripts

        # (f) Runtimes not significantly different relative to previous runs
        # don't test this in jupyter notebooks - runtimes differ too much between machines
        success, report = tstobj.check_runtime(starttime, 1543, success, report)

        tstobj.assertTrue(success, msg=report)

    # Test 4
    @unittest.skipIf(ParallelTaskHelper.isMPIEnabled(), "Only run in serial, since John Tobin's only executed this test in serial (see 01/12/22 comment on CAS-12427).")
    def test_j1927_ql(self):
        """ [j1927] test_j1927_ql """
        ######################################################################################
        # Should match values for "QL" in the "Values to be compared"
        ######################################################################################
        # not part of the jupyter scripts
        tstobj = self # jupyter equivalent: "tstobj = test_j1927()"

        ###########################################
        # %% Set local vars [test_j1927_ql] start @
        ###########################################

        #previous steps in the pipeline would have created mask files from catalogs and images that were created as an
        #intermediate pipeline step.
        tstobj.data_path_dir  = 'J1927/J1927-stakeholdertest-mosaic-data'
        img0 = 'VLASS1.2.ql.T26t15.J1927.10.2048.v1.I.iter0'
        img1 = 'VLASS1.2.ql.T26t15.J1927.10.2048.v1.I.iter1'
        tstobj.prepData()
        # rundir = "/users/bbean/dev/CAS-12427/src/casalith/build-casalith/work/linux/test_vlass_j1927_QL_unittest"
        # os.system(f"mv {rundir}/run_results/VLASS* {rundir}/nosedir/test_vlass_1v2/")

        #########################################
        # %% Set local vars [test_j1927_ql] end @
        # .......................................

        # not part of the jupyter scripts
        starttime = datetime.now()

        # .......................................
        # %% Run tclean [test_j1927_ql] start   @
        #########################################

        def run_tclean(vis=tstobj.vis, intent='OBSERVE_TARGET#UNSPECIFIED', datacolumn='data',
                       imagename=None, niter=None, restoration=None, compare_tclean_pars=None,
                       phasecenter=tstobj.phasecenter, reffreq='3.0GHz', deconvolver='mtmfs', nsigma=0.0, cycleniter=-1, cyclefactor=1.0,
                       cell='1.0arcsec', imsize=7290, gridder='mosaic', restoringbeam='common',
                       perchanweightdensity=False, mosweight=False, scales=[0], calcres=True, calcpsf=True,
                       weighting='briggs', robust=1.0, interactive=0):
            params = locals()
            params = {k:params[k] for k in filter(lambda x: x not in ['tstobj'], params.keys())}
            tstobj.run_tclean(**params)

        script_pars_vals_0 = None#tstobj.get_params_as_dict(vis='J1927_12fields.ms', selectdata=True, field='', spw='', timerange='', uvrange='', antenna='', scan='', observation='', intent='OBSERVE_TARGET#UNSPECIFIED', datacolumn='data', imagename='VLASS1.2.ql.T26t15.J1927.10.2048.v1.I.iter0', imsize=[7290, 7290], cell='1.0arcsec', phasecenter='19:27:30.443 +61.17.32.898', stokes='I', projection='SIN', startmodel='', specmode='mfs', reffreq='3.0GHz', nchan=-1, start='', width='', outframe='LSRK', veltype='radio', restfreq=[], interpolation='linear', perchanweightdensity=False, gridder='mosaic', facets=1, psfphasecenter='', chanchunks=1, wprojplanes=1, vptable='', mosweight=False, aterm=True, psterm=False, wbawp=True, conjbeams=False, cfcache='', usepointing=False, computepastep=360.0, rotatepastep=360.0, pointingoffsetsigdev=[], pblimit=0.2, normtype='flatnoise', deconvolver='mtmfs', scales=[0], nterms=2, smallscalebias=0.0, restoration=False, restoringbeam='common', pbcor=False, outlierfile='', weighting='briggs', robust=1.0, noise='1.0Jy', npixels=0, uvtaper=[], niter=0, gain=0.1, threshold='0.0mJy', nsigma=0.0, cycleniter=-1, cyclefactor=1.0, minpsffraction=0.05, maxpsffraction=0.8, interactive=0, usemask='user', mask='', pbmask=0.0, sidelobethreshold=3.0, noisethreshold=5.0, lownoisethreshold=1.5, negativethreshold=0.0, smoothfactor=1.0, minbeamfrac=0.3, cutthreshold=0.01, growiterations=75, dogrowprune=True, minpercentchange=-1.0, verbose=False, fastnoise=True, restart=True, savemodel='none', calcres=True, calcpsf=True, parallel=False)
        script_pars_vals_1 = None#tstobj.get_params_as_dict(vis='J1927_12fields.ms', selectdata=True, field='', spw='', timerange='', uvrange='', antenna='', scan='', observation='', intent='OBSERVE_TARGET#UNSPECIFIED', datacolumn='data', imagename='VLASS1.2.ql.T26t15.J1927.10.2048.v1.I.iter1', imsize=[7290, 7290], cell='1.0arcsec', phasecenter='19:27:30.443 +61.17.32.898', stokes='I', projection='SIN', startmodel='', specmode='mfs', reffreq='3.0GHz', nchan=-1, start='', width='', outframe='LSRK', veltype='radio', restfreq=[], interpolation='linear', perchanweightdensity=False, gridder='mosaic', facets=1, psfphasecenter='', chanchunks=1, wprojplanes=1, vptable='', mosweight=False, aterm=True, psterm=False, wbawp=True, conjbeams=False, cfcache='', usepointing=False, computepastep=360.0, rotatepastep=360.0, pointingoffsetsigdev=[], pblimit=0.2, normtype='flatnoise', deconvolver='mtmfs', scales=[0], nterms=2, smallscalebias=0.0, restoration=True, restoringbeam='common', pbcor=False, outlierfile='', weighting='briggs', robust=1.0, noise='1.0Jy', npixels=0, uvtaper=[], niter=20000, gain=0.1, threshold=0.0, nsigma=4.5, cycleniter=500, cyclefactor=2.0, minpsffraction=0.05, maxpsffraction=0.8, interactive=0, usemask='user', mask='', pbmask=0.0, sidelobethreshold=3.0, noisethreshold=5.0, lownoisethreshold=1.5, negativethreshold=0.0, smoothfactor=1.0, minbeamfrac=0.3, cutthreshold=0.01, growiterations=75, dogrowprune=True, minpercentchange=-1.0, verbose=False, fastnoise=True, restart=True, savemodel='none', calcres=False, calcpsf=False, parallel=False)
        run_tclean(imagename=img0, niter=0,     restoration=False,                                                                           compare_tclean_pars=script_pars_vals_0)
        for ext in ['.weight.tt2', '.weight.tt0', '.psf.tt0', '.residual.tt0', '.weight.tt1', '.sumwt.tt2', '.psf.tt1', '.residual.tt1', '.psf.tt2', '.sumwt.tt1', '.model.tt0', '.pb.tt0', '.model.tt1', '.sumwt.tt0']:
            shutil.copytree(src=img0+ext, dst=img1+ext)
        run_tclean(imagename=img1, niter=20000, restoration=True, nsigma=4.5, cycleniter=500, cyclefactor=2.0, calcres=False, calcpsf=False, compare_tclean_pars=script_pars_vals_1)

        ###########################################
        # %% Run tclean [test_j1927_ql] end       @
        # %% Prepare Images [test_j1927_ql] start @
        ###########################################

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
        # %% Prepare Images [test_j1927_ql] end            @
        # %% Compare Expected Values [test_j1927_ql] start @
        ####################################################

        # (l) Ensure intermediate products exist, pbcor images, RMS image (made by imdev), and cutouts (.subim) from imsubimage
        success0, report0 = tstobj.get_imgs_exist_results()

        # (a) tt0 vs 6.1.3, on-axis
        imstat_vals       = imstat(imagename=img1+'.image.pbcor.tt0.subim',box='1860,1860,1860,1860')
        curr_stats        = np.squeeze(np.array([imstat_vals['max']]))
        onaxis_stats      = np.array([           0.9509])
        casa613_stats     = np.array([           0.90649462])
        success1, report1 = tstobj.check_fracdiff(curr_stats, onaxis_stats, valname="Frac Diff F_nu vs. on-axis")
        success2, report2 = tstobj.check_fracdiff(curr_stats, casa613_stats, valname="Frac Diff F_nu vs. 6.1.3 image")

        # (b) tt1 vs 6.1.3, on-axis
        # no tt1 images for this test, skip

        # (c) alpha images
        # TODO
        # success3, report3 = ...

        # (d) beamsize comparison vs 6.1.3
        restbeam          = imhead(img1+'.image.pbcor.tt0.subim')['restoringbeam']
        beamstats_curr    = np.array([restbeam['major']['value'], restbeam['minor']['value'], restbeam['positionangle']['value']])
        beamstats_613     = np.array([2.5034000873565674,         2.0568439960479736,         -27.06390953063965])
        success4, report4 = tstobj.check_fracdiff(beamstats_curr, beamstats_613, valname="Frac Diff Maj, Min, PA vs 6.1.3")

        # (e) Confirm presence of model column in resultant MS
        success5, report5 = tstobj.check_column_exists("MODEL_DATA")

        report  = "".join([report0, report1, report2, report4, report5])
        success = success1 and success2 and success4 and success5 and th.check_final(report)
        casalog.post(f"{report}\nSuccess: {success}", "INFO")

        ##################################################
        # %% Compare Expected Values [test_j1927_ql] end @
        ##################################################
        # not part of the jupyter scripts

        # (f) Runtimes not significantly different relative to previous runs
        # don't test this in jupyter notebooks - runtimes differ too much between machines
        success, report = tstobj.check_runtime(starttime, 1543, success, report)

        tstobj.assertTrue(success, msg=report)

##############################################
##############################################

## List to be run
def suite():
    return [test_j1302, test_j1927]

if __name__ == '__main__':
    unittest.main()
