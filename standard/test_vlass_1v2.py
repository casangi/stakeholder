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

from casatasks import casalog

from baseclass.vlass_base_class import test_vlass_base

##############################################
##############################################
class test_j1302(test_vlass_base):
	
    def setUp(self):
        super(test_j1302, self).setUp()
        # variables to be set per test method
        for local_var in ['imsize', 'img']:
            if hasattr(self, local_var):
                delattr(self, local_var)

        ######################################################################
        # This format is an attempt to maintain compatibility with the way
        # that these values were presented to us in the VLASS-provided tests
        # as written by John for CAS-12427.
        
        # values specific to J1302
        self.imagename_base =  'J1302_'
        vis                =  'J1302-12fields.ms'
        phasecenter        =  '13:03:13.874 -10.51.16.73'
        field              =  ''
        spw                =  ''
        antenna            =  ''
        scan               =  ''
        cell               =  '0.6arcsec'
        reffreq            =  '3.0GHz'
        uvrange            =  '<12km'
        intent             =  'OBSERVE_TARGET#UNSPECIFIED'
        wprojplanes        =  32
        usepointing        =  True
        self.common_script_pars = {
            "vis": vis, "phasecenter": phasecenter, "field": field, "spw": spw, "antenna": antenna,
            "scan": scan, "cell": cell, "reffreq": reffreq, "uvrange": uvrange, "intent": intent,
            "wprojplanes": wprojplanes, "usepointing": usepointing
        }

        # "run_tclean" script common default values for J1302
        #   ^-- these scripts: https://open-confluence.nrao.edu/pages/viewpage.action?spaceKey=CASA&title=Requirements+for+VLASS+Imaging+Pipeline+Stakeholders+Tests
        datacolumn     = 'data'
        mosweight      = False
        conjbeams      = True
        rotatepastep   = 5.0
        deconvolver    = 'mtmfs'
        scales         = [0]
        smallscalebias = 0.4
        weighting      = 'briggs'
        robust         = 1.0
        uvtaper        = ''
        niter          = 0
        gain           = 0.1
        threshold      = 0.0
        nsigma         = 2.0
        cycleniter     = 5000
        cyclefactor    = 3
        usemask        = 'user'
        mask           = ''
        pbmask         = 0.0
        restart        = True
        savemodel      = 'none'
        calcres        = True
        calcpsf        = True
        self.default_run_tclean_pars = {
            "datacolumn": datacolumn, "mosweight": mosweight, "conjbeams": conjbeams,
            "rotatepastep": rotatepastep, "deconvolver": deconvolver, "scales": scales,
            "smallscalebias": smallscalebias, "weighting": weighting, "robust": robust,
            "uvtaper": uvtaper, "niter": niter, "gain": gain, "threshold": threshold,
            "nsigma": nsigma, "cycleniter": cycleniter, "cyclefactor": cyclefactor,
            "usemask": usemask, "mask": mask, "pbmask": pbmask, "restart": restart,
            "savemodel": savemodel, "calcres": calcres, "calcpsf": calcpsf
        }

        # merge into one default dict
        self.default_tclean_pars = {}
        for d in [self.common_script_pars, self.default_run_tclean_pars]:
            for k in d:
                self.default_tclean_pars[k] = d[k]

        ######################################################################
        # For maintaining some semblance of compatibility between:
        # - the VLASS format in CAS-12427, above, and
        # - the jupyter notebooks format, from github.com/casangi/stakeholder/
        self.msfile = vis

    def run_tclean(self, **wargs):
        default_tclean_pars = copy.deepcopy(self.default_tclean_pars)
        default_tclean_pars['imsize'] = self.imsize
        tclean_pars = self.get_merged_pars(in_pars=wargs, def_pars=default_tclean_pars)
        self.print_dev_task_call('run_tclean', in_pars=wargs, def_pars=default_tclean_pars)
        # tclean(**tclean_pars)

    # Test 1
    def test_j1302_mosaic(self):
        """ [j1302] test_j1302_mosaic """
        ######################################################################################
        # Should match values for "Stokes I" in the "Values to be compared"
        ######################################################################################
        # TODO self.prepData(...)
        self.imsize = 4000
        self.img = self.imagename_base+'iter2'
        # combine first and 2nd order masks
        # immath(imagename=['secondmask.mask','QLcatmask.mask'],expr='IM0+IM1',outfile='sum_of_masks.mask')
        # im.mask(image='sum_of_masks.mask',mask='combined.mask',threshold=0.5)
        # TODO results = tclean(vis=...)

        # initialize iter2, no cleaning
        self.run_tclean( imagename=self.img, imsize=4000, datacolumn='corrected' )
        # resume iter2 with QL mask
        self.run_tclean( imagename=self.img, datacolumn='corrected', scales=[0,5,12], nsigma=3.0, niter=20000, cycleniter=500, mask="QLcatmask.mask", calcres=False, calcpsf=False  )
        # save model column, doesn't happen here in acutal VLASS pipeline, but makes sure functionality works.
        self.run_tclean( imagename=self.img, calcres=False, calcpsf=False, savemodel='modelcolumn', parallel=False  )
        # resume iter2 with combined mask, remove old mask first, pass new mask as parameter
        os.system('rm -rf '+self.img+'.mask')
        self.run_tclean( imagename=self.img, datacolumn='corrected', scales=[0,5,12], nsigma=3.0, niter=20000, cycleniter=500, mask="combined.mask", calcres=False, calcpsf=False  )
        # resume iter2 with pbmask, remove old mask first then specify pbmask in resumption of tclean
        os.system('rm -rf '+self.img+'.mask')
        self.run_tclean( imagename=self.img, datacolumn='corrected', scales=[0,5,12], nsigma=4.5, niter=20000, cycleniter=100, mask="", calcres=False, calcpsf=False,usemask='pb',pbmask=0.4,parallel=self.parallel)

        # TODO report=th.checkall(...)
        # TODO self.checkfinal(pstr=report)
        pass

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