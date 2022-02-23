import os
import copy
import shutil
from datetime import datetime
from collections import OrderedDict, Iterable
import numpy as np

from casatools import ctsys, table, imager
from casatasks import casalog, immath, tclean
from casatasks.private.parallel.parallel_task_helper import ParallelTaskHelper
from casatestutils.imagerhelpers import TestHelpers

from baseclass.tclean_base_class import test_tclean_base

class test_vlass_base(test_tclean_base):
    """Adds some VLASS test specific extensions to the general stakeholder test class from github.com/casangi/stakeholder/"""
    
    def setUp(self):
        super().setUp()
        self.img = 'delete_me' # here so that test_tclean_base.delData() doesn't error out
        self.imgs = []
        self.th = TestHelpers()
        self.tb = table()
        self.im = imager()

        # TODO remove these hardcoded paths
        ben1 = "/users/bbean/dev/CAS-12427"
        ben2 = "/export/home/figs/bbean/dev/CAS-12427"
        ctsys.setpath(ctsys.getpath() + [
            ben1+"/data",
            ben2+"/data",
        ])

    def tearDown(self):
        super().tearDown()
        self.delData()

        # remove these variables - should be set per test method
        for local_var in ['data_path_dir', 'imagename_base']:
            if hasattr(self, local_var):
                setattr(self, local_var, None)

    def copy_file_or_dir(self, src, dst):
        if (os.path.isdir(src)):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

    def prepData(self, *copyargs):
        msname = self.vis
        super().prepData(msname)

        if os.getenv("USE_CACHED_TCLEAN_VALS") != "true":
            # clean run
            data_path_dir = ctsys.resolve(self.data_path_dir)
            mssrc = os.path.join(data_path_dir, msname)
            casalog.post(f"{mssrc} => {msname}", "SEVERE")
            shutil.copytree(mssrc, msname)

            for copydir in copyargs:
                copysrc = os.path.join(data_path_dir, copydir)
                casalog.post(f"{copysrc} => {copydir}", "SEVERE")
                shutil.copytree(copysrc, copydir)
        else:
            # continue running with partially computed results (eg, ran tclean last time, now check the values)
            from os.path import dirname, join
            import sys
            fromdir = join( dirname(dirname(os.getcwd())), "partial_results" )
            skipfiles = ["__pycache__"]
            files = list(filter(lambda x: x not in skipfiles, os.listdir(fromdir)))
            casalog.post(f"Restorting partial results [{len(files)}]", "SEVERE")
            for i in range(len(files)):
                casalog.post(f"{i}: {files[i]}", "SEVERE")
                self.copy_file_or_dir(join(fromdir, files[i]), files[i])

    def _get_enable_parallel(self):
        """ Returns True if (a) mpi is enabled, and (b) we're not running in a jupyter notebook """
        if 'ipynb' not in self.get_exec_env(): # ipynb ~= notebook
            if ParallelTaskHelper.isMPIEnabled():
                return True
        return False

    def check_img_exists(self, img):
        """ Returns true if the image exists. A report is collected internally, to be returned as a group report in get_imgs_exist_results(...).

        See also: _clean_imgs_exist_dict(...)
        """
        exists = self.th.image_exists(img)
        success, report = self.th.check_val(exists, True, valname=f"image_exists('{img}')", exact=True, testname=self._testMethodName)
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

    def _clean_imgs_exist_dict(self):
        """ Clean the arrays that hold onto the listo fimages to check for existance.

        See also: check_img_exists(...)
        """
        self.imgs_exist = { 'successes':[], 'reports':[] }

    def nparray_to_list(self, val):
        if isinstance(val, np.ndarray) and val.ndim > 0:
            val = list(val)
            for i in range(len(val)):
                val[i] = self.nparray_to_list(val[i])
        return val

    def check_diff(self, actual, expected, diff, valname, desired_diff, max_diff):
        """ Logs a warning if outside of desired bounds, returns False if outside maximum bounds """

        # only worry about comparing the maximum value
        val = diff
        if isinstance(diff, Iterable):
            val = max(diff)
        
        # convert numpy arrays to lists so that the logs get printed on a single line
        actual = self.nparray_to_list(actual)
        expected = self.nparray_to_list(expected)
        diff = self.nparray_to_list(diff)

        # get some values
        out = (val <= max_diff)
        testname = self._testMethodName
        correctval = f"< {max_diff}"

        # generate the report
        if (val > desired_diff):
            casalog.post(f"Warning, {valname}: {diff} vs desired {desired_diff}, (actual: {actual}, expected: {expected})", "WARN")
        report = "[ {} ] {} is {} ( {} : should be {})\n".format(testname, valname, str(diff), self.th.verdict(out), str(correctval) )
        report = report.rstrip() + f" (raw actual/expected values: {actual}/{expected})\n"
        return out, report

    def check_fracdiff(self, actual, expected, valname, desired_diff=0.05, max_diff=0.1):
        """ Logs a warning if outside of desired bounds, returns False if outside required bounds
        
        5% desired, 10% required, as from https://drive.google.com/file/d/1zw6UeDEoXoxM05oFg3rir0hrCMEJMxkH/view and https://open-confluence.nrao.edu/display/VLASS/Updated+VLASS+survey+science+requirements+and+parameters
        """
        fracdiff=abs(actual-expected)/abs(expected)
        return self.check_diff(actual, expected, fracdiff, valname, desired_diff, max_diff)

    def check_absdiff(self, actual, expected, valname, desired_diff=0.1, max_diff=0.2):
        """ Logs a warning if outside of desired bounds, returns False if outside required bounds
        
        0.1 desired, 0.2 required, as from https://drive.google.com/file/d/1zw6UeDEoXoxM05oFg3rir0hrCMEJMxkH/view and https://open-confluence.nrao.edu/display/VLASS/Updated+VLASS+survey+science+requirements+and+parameters
        """
        absdiff=abs(actual-expected)
        return self.check_diff(actual, expected, absdiff, valname, desired_diff, max_diff)

    def check_column_exists(self, colname):
        """ Verifies that the given column exists in the self.vis measurement set. """
        self.tb.open(self.vis)
        cnt = self.tb.colnames().count(colname)
        self.tb.done()
        self.tb.close()
        return self.th.check_val(cnt, 1, valname=f"count('{colname}')", exact=True, testname=self._testMethodName)

    def check_runtime(self, starttime, runtime613, success, report):
        """ Verifies that the runtime is within 10% of the expected runtime613.

        Probably only valid when running on the same hardware as was used to measure the previous runtime.
        """
        endtime           = datetime.now()
        runtime           = (endtime-starttime).total_seconds()
        successt, reportt = self.th.check_val(runtime, runtime613, valname="6.1.3 runtime", exact=False, epsilon=0.1, testname=self._testMethodName)

        report += reportt
        success = success and successt and self.th.check_final(report)
        if not success:
            casalog.post(report, "SEVERE") # easier to read this way than in an assert statement
        else:
            casalog.post(reportt)

        return success, report

    # def get_merged_pars(self, in_pars, def_pars):
    #     ret = copy.deepcopy(def_pars)
    #     for pname in in_pars:
    #         ret[pname] = in_pars[pname]
    #     return ret

    def get_params_as_dict(self, **wargs):
        return dict(wargs)

    def print_task_diff_params(self, fname, act_pars, exp_pars):
        """ Compare the parameter values for the "act_pars" actual parameters
        and the "exp_pars" expected parameters. Print the parameters
        that are different and what their actual/expected values are. """
        same_par_vals = []
        diff_par_vals = []
        diff_par_strs = []
        new_par_vals = []
        new_par_strs = []

        for pname in act_pars:
            par_found = False
            aval_differs = True
            aval = act_pars[pname]
            aval_str = f"'{aval}'" if (type(aval) == str) else str(aval)
            xval = None if pname not in exp_pars else exp_pars[pname]
            xval_str = f"'{xval}'" if (type(xval) == str) else str(xval)
            if pname in exp_pars:
                par_found = True
                if aval == exp_pars[pname]:
                    same_par_vals.append(pname)
                    aval_differs = False
            if not par_found:
                new_par_vals.append(pname)
                new_par_strs.append(f"{pname}={aval_str}")
            elif aval_differs:
                diff_par_vals.append(pname)
                diff_par_strs.append(f"{pname}={aval_str}/{xval_str}")

        diff_pars_str = ", ".join(diff_par_strs)
        new_pars_str = ", ".join(new_par_strs)

        casalog.post(    f"These parameters are different/new: {diff_par_vals+new_par_vals}", "SEVERE")
        if len(diff_pars_str) > 0:
            casalog.post(f"                 (actual/expected): {diff_pars_str}", "SEVERE")
        if len(new_par_vals) > 0:
            casalog.post(f"                          new pars: {new_pars_str}", "SEVERE")

    # def print_tclean(self, **wargs):
    #     diff_par_strs = []
    #     for pname in self.tclean_task_defaults:
    #         if (pname in wargs) and (wargs[pname] != self.tclean_task_defaults[pname]):
    #             sval = f"'{wargs[pname]}'" if (type(wargs[pname]) == str) else str(wargs[pname])
    #             diff_par_strs.append(f"{pname}={sval}")
    #     diff_pars_call = 'tclean(' + ", ".join(diff_par_strs) + ')'

    #     casalog.post(f"Offending tclean call:\n{diff_pars_call}", "SEVERE")

    def _run_tclean(self, **wargs):
        """ Tracks the "imagename" in self.imgs (for cleanup), checks for mask existance, and runs tclean. """
        if ('imagename' in wargs):
            img = wargs['imagename']
            if (img not in self.imgs):
                self.imgs.append(img)
        if ('mask' in wargs) and (wargs['mask'] != ''):
            if not os.path.exists(wargs['mask']):
                raise RuntimeError(f"Error: trying to run tclean with nonexistant mask {wargs['mask']}")
        try:
            if os.getenv("USE_CACHED_TCLEAN_VALS") != "true":
                tclean(**wargs)
                pass
        except:
            # self.print_tclean(**wargs)
            raise

    def run_tclean(self, vis='', selectdata=True, field='', spw='', timerange='', uvrange='', antenna='',
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
                    calcres=True, calcpsf=True, psfcutoff=0.35, parallel=None, compare_tclean_pars=None):
        """ Runs tclean with the default parameters from v6.4.0
        If the 'compare_tclean_pars' dict is provided, then compare these values to the other parameters of this function.

        See also: self.run_tclean(...)
        """
        parallel = (self.parallel) if (parallel == None) else (parallel)
        run_tclean_pars = locals()
        run_tclean_pars = {k:run_tclean_pars[k] for k in filter(lambda x: x not in ['self', 'compare_tclean_pars', 'psfcutoff'] and '__' not in x, run_tclean_pars.keys())}
        if (compare_tclean_pars != None):
            self.print_task_diff_params('run_tclean', act_pars=run_tclean_pars, exp_pars=compare_tclean_pars)
        self._run_tclean(**run_tclean_pars)

    def resize_mask(self, maskname, outputmaskname, shape=[2000,2000,1,1]):
        """Resizes a .image mask from its current shape to the given shape"""
        from casatools import image as _ia
        from skimage.transform import resize # from package scikit-image
        ia = _ia()

        # scrub the input
        # a trailing '/' on the image name causes calc to give weird results
        maskname = maskname.rstrip(" \t\n\r/\\")
        if (len(shape) < 4): # ra, dec, chan, pol
            shape += [1 for i in range(4-len(shape))]
        if shape[2] != 1 or shape[3] != 1:
            raise RuntimeError("Error: image must have length 1 in the third (chan) and fourth (pol) dimensions")

        # get the shape
        ia.open(maskname)
        try:
            inshape = ia.shape()
            pixeltype = ia.pixeltype()
            inpixels = ia.getregion()
        finally:
            ia.close()
            ia.done()

        # populate some pixels
        pixels = resize(inpixels, shape)
        for r in range(shape[0]):
            for d in range(shape[1]):
                pixels[r][d][0][0] = 0 if (pixels[r][d][0][0] < 0.5) else 1

        # create the new outputmask
        if (pixeltype == 'dcomplex'):
            pixeltype = 'cd'
        else:
            pixeltype = pixeltype[0] # for 'f'loat, 'd'ouble, or 'c'omplex
        if os.path.exists(outputmaskname):
            shutil.rmtree(outputmaskname)
        ia.fromarray(outputmaskname, pixels=pixels, type=pixeltype)
        ia.close()
        ia.done()