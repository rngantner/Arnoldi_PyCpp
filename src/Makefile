CC=g++
CFLAGS=-Wall -O3 -fopenmp

PYTHONVERSION=2.7

# Generic and Debian GNU/Linux
PYHEADERS=/usr/include/python$(PYTHONVERSION)
NUMPYHEADERS=/usr/lib/python$(PYTHONVERSION)/site-packages/numpy/core/include
EIGEN3=/usr/include/eigen3
PYLIBS=/usr/lib # location of libboost_python.so
BOOSTPYLIBS=/usr/lib

# Brutus
#module load python
#module load boost
# get NUMPYHEADERS location with command numpy.get_include() 
#NUMPYHEADERS=/cluster/apps/python/2.7.2/x86_64/lib64/python2.7/site-packages/numpy/core/include
#PYHEADERS=/cluster/apps/python/2.7.2/x86_64/include/python2.7/
#EIGEN3=eigen3 # local installation
#BOOSTPYLIBS=/cluster/apps/boost/1_47_0_nompi/x86_64/gcc_4.1.2/lib64
#PYLIBS=/cluster/apps/python/2.7.2/x86_64/lib64

# OS X
#NUMPYHEADERS=/Library/Frameworks/Python.framework/Versions/6.3/lib/python2.6/site-packages/numpy/core/include
#PYHEADERS=/usr/include/python$(PYTHONVERSION)
#PYHEADERS2=/Library/Frameworks/Python.framework/Versions/6.3/include/python2.6
#BOOSTPYLIBS=/usr/lib # location of libboost_python.so
#PYLIBS=/usr/lib # ?? not needed ??

modulename=arnoldi

all: python prog
 
python:
	$(CC) $(CFLAGS) -c -fPIC -DPYTHONMODULE -I$(PYHEADERS) -I$(NUMPYHEADERS) -I$(EIGEN3) ${modulename}.cpp
	$(CC) $(CFLAGS) -shared -o ${modulename}.so ${modulename}.o $(BOOSTPYLIBS)/libboost_python.so -I$(EIGEN3) -L$(PYLIBS) -lpython$(PYTHONVERSION)

prog:
	$(CC) $(CFLAGS) -I$(PYHEADERS) -I$(NUMPYHEADERS) -I$(EIGEN3) ${modulename}.cpp -o ${modulename}

clean:
	$(RM) ${modulename} ${modulename}.so *.o
	$(RM) *.pyc a.out
