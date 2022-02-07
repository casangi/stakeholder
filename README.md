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

# Usage

The stakeholder test code has two modes of operation: running in a functional manner using `jupyter-notebook` or running as a script via the command-line; this includes both running as a script using `python3` or using 
casalith as `casa -c`.
 

### Installation

If running locally using `juptyer-notebook` or `python3` you can install the required libraries using `pip` in either your standard environment or by creating a new environment. 

```
python3 -m pip install -r requirements.txt
```

You will also need to install the stakeholder tools library from the `pypi` test server (This is a stakeholder tool package I built while a better solution is identified). This library is currently required even when 
running the stakeholder tests using casalith. Once the tools are added to `casatestutils`, this will no longer be necessary.  

```
python3 -m pip install --index-url https://test.pypi.org/simple/ --no-deps stktools
```

### Execution from Command-line

You can now run the test case scripts either using `python` locally  or using your choice of `casa` standalone. In order to run the first test case, from the command-line using `python3`, 

```
cd stakeholder/
python3 -m scripts.test_standard_cube_briggsbwtaper
```

You can also run your choice of test(s) using a given `casa` version using (test_standard_cube_briggsbwtaper.py for instance)

```
casa -c <directory to script>/test_standard_cube_briggsbwtaper.py
```

The code should run to completion and a html testing report should be created in the `stakeholder/` directory.

```
test_tclean_alma_pipeline_weblog.html
```

### Execution using Jupyter Notebook

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