import copy
from casatasks import casalog, immath

from baseclass.tclean_base_class import test_tclean_base

class test_vlass_base(test_tclean_base):
    """Adds some VLASS test specific extensions to the general stakeholder test class from github.com/casangi/stakeholder/"""

    def tearDown(self):
        super(test_vlass_base, self).tearDown()
        self.delData()

    def get_merged_pars(self, in_pars, def_pars):
        ret = copy.deepcopy(def_pars)
        for k in in_pars:
            ret[k] = in_pars[k]
        return ret

    def print_dev_task_call(self, fname, in_pars, def_pars):
        same_par_vals = []
        diff_par_strs = []

        for k in in_pars:
            par_found = False
            parval_differs = True
            v = in_pars[k]
            if k in def_pars:
                par_found = True
                if v == def_pars[k]:
                    same_par_vals.append(k)
                    parval_differs = False
            if parval_differs or not par_found:
                sval = f"'{v}'" if (type(v) == str) else str(v)
                diff_par_strs.append(f"{k}={sval}")

        diff_pars_call = fname + '(' + ",".join(diff_par_strs) + ')'

        casalog.post(f"These parameters are the same: {same_par_vals}", "SEVERE")
        casalog.post(f"Differing parameters only call: {diff_pars_call}", "SEVERE")