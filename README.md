# Installation
## JupyterLab
JupyterLab notebook can be installed easily using either `conda` or `pip`. This has been confirmed on Mint Linux 20 and RedHat 8. Both python 3.6 and python 3.8 have been tested.

### Installation with conda
To install `using conda` using `conda-forge`:

```
conda install -c conda-forge jupyterlab
```

### Installation with pip: JupyterLab
When using `pip` to install `jupyterlab` the following can be used:

```
python3 -m pip install jupyterlab
```

When installing using using the `--user` tag, you must add the user-level `bin` to your path. This can be done using `export $PATH:$HOME/.local/bin:$PATH` in bash.

### Installation with pip: notebook
If you would prefer to use the classic notebook you can instead use the following:

```
python3 -m pip install notebook
```

Rules concerning the `--user` tag are the same as in the `jupyterlab` case.

Once installation is complete, if the installation was successful, the notebook can be run from the command-line using, 

```
jupyter-lab
```
or
```
jupyter notebook
```

depending on your choice of installation.

### Setup
For testing purposes the expected metric values dictionary and relevant measurement file is available for download. For simplicity, a setup script is provided to pull the files and build the data directory structure.
`python3 setup.py`. If this is successful, you should see the measurement file and expected values dictionary within `stakeholder/data/`.

```
(env) [username@machine notebook]$ ls -lrt data
total 716
-rw-r--r--  1 username nraocv 726297 Dec 10 09:46 test_stk_alma_pipeline_imaging_exp_dicts.json
drwxr-xr-x 28 username nraocv   4096 Jan  6 10:29 E2E6.1.00034.S_tclean.ms
``` 

# Usage

The stakeholder tests can be used either from the command-line as in the case of unit testing and in the form of a `jupyter-notebook`.  Each test case is divided into a separate directory and a list of test cases is provided below. 

### Installation

You can install the required libraries using `pip` in either your standard envirnoment or by creating a new environment.

```
python3 -m pip install -r requirements.txt
```

You will also need to install the stakeholder tools library from the `pypi` test server (This is a stakehodler tool package I built while a better solution is identified)

```
python3 -m pip install --index-url https://test.pypi.org/simple/ --no-deps stktools
```

You can now run the test case scripts either using `python` locally  or using your choice of `casa` standalone. In order to run the first test case, from the command-line using `python`, 

```
cd stakeholder/
python3 -m scripts.test_standard_cube_briggsbwtaper
```

You can also run your choice of test using a given `casa` version using 

```
casa -c scripts/stakeholder_test.py --stakeholder-test <test-id>
```

The list of available tests is currently,

| Test-ID                       | Stakeholder Test                   |
| :---                          |    :----:                          |
| standard_cube_briggsbwtaper   | test_standard_cube_briggsbwtaper   |
| mosaic_cube_briggsbwtaper     | test_mosaic_cube_briggsbwtaper     |


The code should run to completion and an html testing report should be created in the `stakeholder/` directory.

```
test_tclean_alma_pipeline_weblog.html
```

The `jupyter notebook` test cases are available in the `stakeholder/` directory. Notebooks can be run by simply running the the notebook and opening the desired test case (`.ipynb`). 

**As a warning, it is advised that the user restart the kernel and run the notebook after making changes to avoid issues due the hidden states in Jupyter notebooks.**

## Syncing notebook <--> script

The `scripts/nbsync.py` script allows the user to synchronize changes between the notebook and the testing script. The script makes changes based on code packaged between header (footer). For example in the standard_cube code:

```
# %% test_standard_cube_briggsbwtaper_tclean_1 start @

Modifiable code placed here.

# %% test_standard_cube_briggsbwtaper_tclean_1 end @

```

Multiple header (footers) can be used in the code but they must be unique pairs. Currently the script only works on with the `standard_cube_briggsbwtaper` script and notebook. The synchronization can be done as follows:

**notebook --> script**
`python3 nbsync.py --tout`

**script --> notebook**
`python3 nbsync.py --tonb`