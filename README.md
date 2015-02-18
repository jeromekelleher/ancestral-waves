# ancestral-waves
Simulations of ancestral waves for the continuum, Wright-Fisher and Moran models.


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
$ sudo apt-get install build-essential git libgsl0-dev libconfig-dev python-dev python-scipy
$ git clone https://github.com/jeromekelleher/ancestral-waves.git
$ cd ancestral-waves
$ pip install -r requirements.txt --user
$ make
```
