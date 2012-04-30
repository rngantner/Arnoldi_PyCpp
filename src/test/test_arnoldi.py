#!/usr/bin/env python2
# coding: utf-8
import time
from numpy import *
from numpy.linalg import norm
import arnoldi as a
from pyarnoldi import arnoldi
import matplotlib.pyplot as plt

from numpy.testing import assert_almost_equal
class TestTypes:
    def helper(self,T,prec=10):
        k = 7
        Nvals = [10, 50, 100, 1000, 5000]
        for n in Nvals:
            A = array(random.random((n,n)), dtype=T)
            v = ones((n,1), dtype=T)
            # C++ version
            V = zeros((n,k), dtype=T)
            H = zeros((k+1,k), dtype=T)
            a.arnoldi(A,v,k,V,H)
            # Python version
            Vloc,Hloc = arnoldi(A,v,k)
            # assertion
            assert_almost_equal(norm(H)/n,norm(Hloc)/n,prec)

    def test_float32(self):
        self.helper(float32,6)
    
    def test_float64(self):
        self.helper(float64)
    
    def test_complex64(self):
        self.helper(complex64,6)
    
    def test_complex128(self):
        self.helper(complex128)


