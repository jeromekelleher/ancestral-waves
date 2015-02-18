# Ancestral Waves
Simulations of ancestral waves for the continuum, Wright-Fisher and Moran models.

# Running the simulations

All of the simulations are run and analysed using the `src/simulations.py` Python
script. This script has a user interface with online help; for example,
```sh
$ python src/simulations.py -h
usage: simulations.py [-h] {simulate,process,plot,list} ...

Run simulations, process data files and generate plots.

positional arguments:
  {simulate,process,plot,list}

optional arguments:
  -h, --help            show this help message and exit
```

The script has three basic commands: `simulate`, `process` and `plot`. The fourth
command `list` just lists the available plots that we can generate:
```sh
$ python src/simulations.py list
Available plots:
         1d_pedigree_cline_shape_mean
         1d_ancestral_material_linear_genome
         1d_genetic_wave_parameters
         1d_pedigree_cline_shape_replicates
         2d_genetic_wave_parameters
         1d_pedigree_integral_equation
         1d_pedigree_wave_parameters
         1d_pedigree_numerics_comparison
         1d_pedigree_genetic_comparison
         1d_ancestral_material_free_recombination
         1d_2d_pedigree_parameters_comparison
```
A special plot ID `all` is recognised, which sequentially runs the 
command in question for all plots.

The `simulate` command takes a plot ID as a parameter, and optionally the number of 
replicates and the number of processors to use. For example,
```sh
$ python src/simulations.py simulate 1d_pedigree_wave_parameters -n 10 -p 4
```
This runs 10 replicates for the ``1d_pedigree_wave_parameters`` simulation
using 4 processes. By default a large number of replicates will be simulated
using all the available cores on the machine. Replicates are stored in the 
``data/replicates__NOBACKUP__`` directory.

The `process` command takes the raw replicate data in the ``data/replicates__NOBACKUP__``
directory, derives summaries and writes the results to files in the ``data``
directory. For example,
```sh
$ python src/simulations.py process 1d_pedigree_wave_parameters
Read 10 n replicates at g=0 for rho_e = 75.0 d = 1
Read 10 n replicates at g=5 for rho_e = 75.0 d = 1
[snip]
```

Finally, the `plot` command takes the summarised data generated by the 
`process` command and generates a plot. Plots are written in the 
`figures` directory.
```sh
$ python src/simulations.py process 1d_pedigree_wave_parameters
```

# Requirements


To compile the C programs, we need [libconfig](http://www.hyperrealm.com/libconfig/)
and [GSL](http://www.gnu.org/software/gsl/). Both of these are commonly available
in package managers.

On Debian/Ubuntu, for example, we can use
```sh
$ sudo apt-get install libgsl0-dev libconfig-dev
```


# Running from a new Debian wheezy install

To run the simulations on a fresh Debian wheezy install, do the following:
```sh
$ sudo apt-get install build-essential git libgsl0-dev libconfig-dev \
python-dev python-scipy python-pip
$ git clone https://github.com/jeromekelleher/ancestral-waves.git
$ cd ancestral-waves
$ pip install -r requirements.txt --user
$ make
```
