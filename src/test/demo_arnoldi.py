#!/usr/bin/env python2
# coding: utf-8
import time
from numpy import *
from numpy.linalg import norm
import matplotlib.pyplot as plt
import sys, os

sys.path.append('..')
import carnoldi as a
from pyarnoldi import arnoldi

if __name__ == '__main__':
    basedir = 'results'
    if not os.path.exists(basedir):
        os.mkdir(basedir)
    k = 7
    n = 20
    Creps = 10
    Preps = 6
    #Nvals = logspace(1, 4) # 5, ..., 10^2 logarithmically
    #Nvals = logspace(1, log(5000)/log(10))
    Nmin = 10
    lNmin = log(Nmin)/log(10)
    #Nmax = 20000
    Nmax = 1500
    lNmax = log(Nmax)/log(10)
    Nvals = array(logspace(lNmin, lNmax),dtype=int)

    type_to_name = {float32:'float32', float64:'float64', complex64:'complex64', complex128:'complex128'}
    #for T in [float32, float64, complex64, complex128]:
    for T in [float32, float64, complex64, complex128]:
        print "current type:", type_to_name[T]
        Tc = []
        Tp = []
        for n in Nvals:
            print "n:",n
            #A = array(diag(2*ones(n)) - diag(ones(n-1),1) - diag(ones(n-1),-1), dtype=t)
            A = array(random.random((n,n)), dtype=T)
            v = ones((n,1), dtype=T)
            ctime = inf; ptime = inf
            for r in range(Creps):
                # time C++ version
                V = zeros((n,k), dtype=T)
                H = zeros((k+1,k), dtype=T)
                t = time.time()
                a.arnoldi(A,v,k,V,H)
                ctime = min(ctime, time.time()-t)
            for r in range(Preps):# time python version
                t = time.time()
                Vloc,Hloc = arnoldi(A,v,k)
                ptime = min(ptime, time.time()-t)
            Tc.append(ctime)
            Tp.append(ptime)
            
            normHH = norm(H-Hloc)/norm(H) # "relative" norm
            epsilon = 1e-5
            if normHH > epsilon:
                print "H-Hloc is not small (%f>%f)!"%(normHH,epsilon)
            del A, v, V, H, Vloc, Hloc

        plt.loglog(Nvals, Tp, '-o', label="Python")
        plt.loglog(Nvals, Tc, '-^', label="C++")
        plt.legend(loc='best')
        plt.xlabel('N', fontsize=16)
        plt.ylabel('Time', fontsize=16)
        plt.title('Timing of Arnoldi Algorithm with $k=%d$. Type: %s'%(k,type_to_name[T]))
        ax = plt.axis()
        plt.axis((Nmin,Nmax,ax[2],ax[3]))
        plt.draw()
        plt.savefig(os.path.join(basedir,'arnolditimings_'+type_to_name[T]+'.pdf'))
        #plt.show()
        plt.close()


