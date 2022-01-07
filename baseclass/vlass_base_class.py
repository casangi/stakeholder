import os
import copy
import shutil
from collections import OrderedDict

from casatasks import casalog, immath, tclean

from baseclass.tclean_base_class import test_tclean_base

class test_vlass_base(test_tclean_base):
    """Adds some VLASS test specific extensions to the general stakeholder test class from github.com/casangi/stakeholder/"""
    
    def setUp(self):
        super().setUp()
        self.img = 'delete_me' # here so that test_tclean_base.delData() doesn't error out
        self.imgs = []
        self.data_path = "/users/bbean/dev/CAS-12427/data"

        # # actual tclean defaults, copy-pasted from https://casadocs.readthedocs.io/en/stable/api/tt/casatasks.imaging.tclean.html#casatasks.imaging.tclean
        # lst_tclean_task_defaults = [('selectdata', True), ('field', ''), ('spw', ''), ('timerange', ''), ('uvrange', ''), ('antenna', ''), ('scan', ''), ('observation', ''), ('intent', ''), ('datacolumn', 'corrected'), ('imagename', ''), ('imsize', [100]), ('cell', ['1arcsec']), ('phasecenter', ''), ('stokes', 'I'), ('projection', 'SIN'), ('startmodel', ''), ('specmode', 'mfs'), ('reffreq', ''), ('nchan', - 1), ('start', ''), ('width', ''), ('outframe', 'LSRK'), ('veltype', 'radio'), ('restfreq', ''), ('interpolation', 'linear'), ('perchanweightdensity', True), ('gridder', 'standard'), ('facets', 1), ('psfphasecenter', ''), ('wprojplanes', 1), ('vptable', ''), ('mosweight', True), ('aterm', True), ('psterm', False), ('wbawp', True), ('conjbeams', False), ('cfcache', ''), ('usepointing', False), ('computepastep', 360.0), ('rotatepastep', 360.0), ('pointingoffsetsigdev', ''), ('pblimit', 0.2), ('normtype', 'flatnoise'), ('deconvolver', 'hogbom'), ('scales', ''), ('nterms', 2), ('smallscalebias', 0.0), ('restoration', True), ('restoringbeam', ''), ('pbcor', False), ('outlierfile', ''), ('weighting', 'natural'), ('robust', 0.5), ('noise', '1.0Jy'), ('npixels', 0), ('uvtaper', ['']), ('niter', 0), ('gain', 0.1), ('threshold', 0.0), ('nsigma', 0.0), ('cycleniter', - 1), ('cyclefactor', 1.0), ('minpsffraction', 0.05), ('maxpsffraction', 0.8), ('interactive', False), ('usemask', 'user'), ('mask', ''), ('pbmask', 0.0), ('sidelobethreshold', 3.0), ('noisethreshold', 5.0), ('lownoisethreshold', 1.5), ('negativethreshold', 0.0), ('smoothfactor', 1.0), ('minbeamfrac', 0.3), ('cutthreshold', 0.01), ('growiterations', 75), ('dogrowprune', True), ('minpercentchange', - 1.0), ('verbose', False), ('fastnoise', True), ('restart', True), ('savemodel', 'none'), ('calcres', True), ('calcpsf', True), ('psfcutoff', 0.35), ('parallel', False)]
        # # some of the defaults as listed online are not what they claim to be
        # for replacement in [('pointingoffsetsigdev',[]), ('restfreq',[]), ('uvtaper',[])]:
        #     for i in range(len(lst_tclean_task_defaults)):
        #         if lst_tclean_task_defaults[i][0] == replacement[0]:
        #             lst_tclean_task_defaults[i] = (lst_tclean_task_defaults[i][0], replacement[1])
        # # put these values into a dict
        # self.tclean_task_defaults = OrderedDict()
        # for kv in lst_tclean_task_defaults:
        #     self.tclean_task_defaults[kv[0]] = kv[1]

    def tearDown(self):
        super().tearDown()
        self.delData()

        # remove these variables - should be set per test method
        for local_var in ['data_path_dir', 'imagename_base']:
            if hasattr(self, local_var):
                setattr(self, local_var, None)

    def prepData(self):
        msname = self.vis
        super().prepData(msname)

        data_path_dir = self.data_path_dir
        mssrc = os.path.join(self.data_path, data_path_dir, msname)

        shutil.copytree(mssrc, msname)

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
            if aval_differs or not par_found:
                diff_par_vals.append(pname)
                diff_par_strs.append(f"{pname}={aval_str}/{xval_str}")

        diff_pars_str = ", ".join(diff_par_strs)

        casalog.post(f"These parameters are different: {diff_par_vals}", "SEVERE")
        casalog.post(f"             (actual/expected): {diff_pars_str}", "SEVERE")

    # def print_tclean(self, **wargs):
    #     diff_par_strs = []
    #     for pname in self.tclean_task_defaults:
    #         if (pname in wargs) and (wargs[pname] != self.tclean_task_defaults[pname]):
    #             sval = f"'{wargs[pname]}'" if (type(wargs[pname]) == str) else str(wargs[pname])
    #             diff_par_strs.append(f"{pname}={sval}")
    #     diff_pars_call = 'tclean(' + ", ".join(diff_par_strs) + ')'

    #     casalog.post(f"Offending tclean call:\n{diff_pars_call}", "SEVERE")

    def run_tclean(self, **wargs):
        if ('imagename' in wargs):
            img = wargs['imagename']
            if (img not in self.imgs):
                self.imgs.append(img)
        try:
            tclean(**wargs)
            pass
        except:
            # self.print_tclean(**wargs)
            raise