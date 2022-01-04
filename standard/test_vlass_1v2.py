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
# Values to current versions against:
#  1. known ground truth     --- from what? fluxscale task?
#  2. CASA 6.1.3 values      --- pipeline approved version of casa
#  3. on-axis values         --- do either John or Urvashi know what this means? is this another version of casa maybe?
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
# Processing methods:
#  mosaic      - "stokes I?"
#  awproject   - "stokes I?"
#  mosaic cube - "cube?"
#  mosaic QL   - "QL?"       --- what is VLASS QL? maybe it's a pipeline thing?
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
import copy
import unittest
import numpy as np
from collections import OrderedDict
import shutil

from casatasks import casalog, impbcor, imdev, imhead, imsubimage, imstat

from baseclass.vlass_base_class import test_vlass_base

##############################################
##############################################
class test_j1302(test_vlass_base):
	
    def setUp(self):
        super().setUp()

        ######################################################################
        # This format is an attempt to maintain compatibility with the way
        # that these values were presented to us in the VLASS-provided tests
        # as written by John for CAS-12427.
        
        # values specific to J1302
        vis                 =  'J1302-12fields.ms'
        phasecenter         =  '13:03:13.874 -10.51.16.73'
        reffreq             =  '3.0GHz'
        intent              =  'OBSERVE_TARGET#UNSPECIFIED'
        deconvolver         = 'mtmfs'
        scales              = [0]
        weighting           = 'briggs'
        robust              = 1.0
        mosweight           = False
        interactive         = 0
        self.common_script_pars = OrderedDict()
        for k in ['vis', 'phasecenter', 'reffreq', 'intent', 'deconvolver', 'scales', 'weighting', 'robust', 'mosweight', 'interactive']:
            self.common_script_pars[k] = locals()[k]

        # non-QL J1302 script common defaults
        #   ^-- these scripts: https://open-confluence.nrao.edu/pages/viewpage.action?spaceKey=CASA&title=Requirements+for+VLASS+Imaging+Pipeline+Stakeholders+Tests
        cell           =  '0.6arcsec'
        uvrange        =  '<12km'
        wprojplanes    =  32
        usepointing    =  True
        datacolumn     = 'corrected'
        conjbeams      = True
        rotatepastep   = 5.0
        smallscalebias = 0.4
        nsigma         = 2.0
        cycleniter     = 5000
        cyclefactor    = 3
        self.default_script_pars = OrderedDict()
        for k in ['cell', 'uvrange', 'wprojplanes', 'usepointing', 'datacolumn', 'conjbeams', 'rotatepastep', 'smallscalebias', 'nsigma', 'cycleniter', 'cyclefactor']:
            self.default_script_pars[k] = locals()[k]

        # Merge into one default dict.
        # Include the tclean defaults, as currently described in documentation, so that even if the
        # defaults for the task change in the future, the parameters don't change for these tests.
        self.default_tclean_pars = {}
        self.default_tclean_pars_ql = {}
        pars_dicts = [self.tclean_task_defaults, self.common_script_pars, self.default_script_pars]
        for i in range(3):
            d = pars_dicts[i]
            for k in d:
                self.default_tclean_pars[k] = d[k]
                if i != 2: # skip non-QL common values
                    self.default_tclean_pars_ql[k] = d[k]

        ######################################################################
        # For maintaining some semblance of compatibility between:
        # - the VLASS format in CAS-12427, above, and
        # - the jupyter notebooks format, from github.com/casangi/stakeholder/
        self.msfile = vis

    def _run_tclean(self, def_pars, **wargs):
        def_pars['imsize'] = self.imsize
        tclean_pars = self.get_merged_pars(in_pars=wargs, def_pars=def_pars)
        self.print_dev_task_call('run_tclean', in_pars=wargs, def_pars=def_pars)
        super().run_tclean(**tclean_pars)

    def run_tclean(self, **wargs):
        default_tclean_pars = copy.deepcopy(self.default_tclean_pars)
        self._run_tclean(default_tclean_pars, **wargs)

    def run_tclean_ql(self, **wargs):
        default_tclean_pars_ql = copy.deepcopy(self.default_tclean_pars_ql)
        self._run_tclean(default_tclean_pars_ql, **wargs)

    def replace_psf(old, new):
        """ Replaces [old] PSF image with [new] image. Clears parallel working directories."""
        for this_tt in ['tt0', 'tt1', 'tt2']:
            shutil.rmtree(self.imagename_base+old+'.psf.'+this_tt)
            shutil.copytree(self.imagename_base+new+'.psf.'+this_tt, self.imagename_base+old+'.psf.'+this_tt)

    # Test 1
    def test_j1302_mosaic(self):
        """ [j1302] test_j1302_mosaic """
        ######################################################################################
        # Should match values for "Stokes I" in the "Values to be compared"
        ######################################################################################
        self.imsize = 4000
        img = self.imagename_base+'iter2'
        # self.prepData(nocube=True)
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
        self.imsize = 5250
        # img0 = self.imagename_base+'iter0d'
        # img2 = self.imagename_base+'iter2'
        # self.prepData(nocube=True)

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
        self.imsize = 4000
        # self.prepData(nocube=False)

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
        #previous steps in the pipeline would have created mask files from catalogs and images that were created as an 
        #intermediate pipeline step.
        self.data_path_dir  = 'J1302/Stakeholder-test-mosaic-data'
        self.imagename_base = 'J1302_'
        self.imsize = 7290
        img0 = 'VLASS1.2.ql.T08t20.J1302.10.2048.v1.I.iter0'
        img1 = 'VLASS1.2.ql.T08t20.J1302.10.2048.v1.I.iter1'
        self.prepData(nocube=True)

        ##############
        # Run tclean #
        ##############

        self.run_tclean_ql(imagename=img0, niter=0,     cell='1.0arcsec', datacolumn='data', perchanweightdensity=False, gridder='mosaic', restoration=False, restoringbeam='common', threshold='0.0mJy')
        for ext in ['.weight.tt2', '.weight.tt0', '.psf.tt0', '.residual.tt0', '.weight.tt1', '.sumwt.tt2', '.psf.tt1', '.residual.tt1', '.psf.tt2', '.sumwt.tt1', '.model.tt0', '.pb.tt0', '.model.tt1', '.sumwt.tt0']:
            shutil.copytree(src=img0+ext, dst=img1+ext)
        self.run_tclean_ql(imagename=img1, niter=20000, cell='1.0arcsec', datacolumn='data', perchanweightdensity=False, gridder='mosaic', restoringbeam='common', nsigma=4.5, cycleniter=500, cyclefactor=2.0, calcres=False, calcpsf=False)

        ##################
        # Prepare Images #
        ##################

        # hifv_pbcor(pipelinemode="automatic")
        for fromext,toext in [('.image.tt0','.image.pbcor.tt0'), ('.residual.tt0','.image.residual.pbcor.tt0')]:
            impbcor(imagename=img1+fromext, pbimage=img1+'.pb.tt0', outfile=img1+toext, mode='divide', cutoff=-1.0, stretch=False)
            self.assertTrue(os.path.exists(img1+toext), msg=f"os.path.exists('{img1+toext}')")

        # hif_makermsimages(pipelinemode="automatic")
        imdev(imagename=img1+'.image.pbcor.tt0',
              outfile=img1+'.image.pbcor.tt0.rms',
              overwrite=True, stretch=False, grid=[10, 10], anchor='ref',
              xlength='60arcsec', ylength='60arcsec', interp='cubic', stattype='xmadm',
              statalg='chauvenet', zscore=-1, maxiter=-1)
        self.assertTrue(os.path.exists(img1+'.image.pbcor.tt0.rms'), msg=f"os.path.exists('{img1+'.image.pbcor.tt0.rms'}')")

        # hif_makecutoutimages(pipelinemode="automatic")
        for ext in ['.image.tt0', '.residual.tt0', '.image.pbcor.tt0', '.image.pbcor.tt0.rms', '.psf.tt0', '.image.residual.pbcor.tt0', '.pb.tt0']:
            imhead(imagename=img1+ext)
            imsubimage(imagename=img1+ext, outfile=img1+ext+'.subim', box='1785.0,1785.0,5506.0,5506.0')
            self.assertTrue(os.path.exists(img1+ext+'.subim'), msg=f"os.path.exists('{img1+ext+'.subim'}')")

        ###########################
        # Compare Expected Values #
        ###########################

        tt0stats=imstat(imagename=img1+'.image.pbcor.tt0.subim',box='1860,1860,1860,1860')
        onaxis_stats=np.array([0.9509])
        casa613_stats=np.array([0.888])
        currentstats=np.squeeze(np.array([tt0stats['max']]))

        print('F_nu (current image): ',currentstats)
        fracdiff=(currentstats-casa613_stats)/casa613_stats
        print('Frac Diff F_nu vs. 6.1.3 image: ',fracdiff)

        fracdiff=(currentstats-onaxis_stats)/onaxis_stats
        print('Frac Diff F_nu vs. on-axis: ',fracdiff)

        header=imhead(img1+'.image.pbcor.tt0.subim')
        beamstats_613=np.array([3.1565470695495605,                        2.58677792549133,                         11.282347679138184])
        beamstats=np.array([    header['restoringbeam']['major']['value'], header['restoringbeam']['minor']['value'],header['restoringbeam']['positionangle']['value']])

        fracdiff=(beamstats-beamstats_613)/beamstats_613
        print('Frac DiffMaj, Min, PA vs 6.1.3: ',fracdiff)


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