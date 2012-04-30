/* 
 * File:   arnoldi.cpp
 * Author: Robert Gantner
 *
 * Created on November 25, 2011
 */

#ifndef ARNOLDI_HPP
#define ARNOLDI_HPP

#include "../inc/globalinc.hpp"
#include <boost/shared_ptr.hpp>
#include <unsupported/Eigen/MatrixFunctions>

#ifdef PYTHONMODULE
#include <Python.h>
#include <boost/python.hpp>
#include <numpy/arrayobject.h>
#endif

#include <Eigen/Core>
using namespace Eigen;

// delete this when done testing
#include <iostream>
using namespace std;

/**
 * Arnoldi method to compute Krylov approximation of a matrix
 * @param A nxn matrix to approximate
 * @param v nx1 initial vector
 * @param k number of Krylov steps (size of resulting basis)
 * @param V output matrix (n x k) of orthogonal vectors
 * @param H output matrix (k+1 x k) containing Krylov approximation of A
 */
template<class DerivedMatrix, class DerivedVector>
void arnoldi(const MatrixBase<DerivedMatrix>& A, const MatrixBase<DerivedVector>& v, 
               size_t k, MatrixBase<DerivedMatrix>& V, MatrixBase<DerivedMatrix>& H){
    int n = A.cols();
    typename Types<typename DerivedVector::Scalar>::Vector vt(n);
    V.col(0) = v/v.norm();
    for (unsigned int m=0; m<k; m++) {
        vt = A*V.col(m);
        for (unsigned int j=0; j<m+1; j++) {
            H(j,m) = vt.dot(V.col(j));
            vt = vt - H(j,m)*V.col(j);
        }
        H(m+1,m) = vt.norm();
        if (m != k-1)
            V.col(m+1) = vt/H(m+1,m);
	}
}

#ifdef PYTHONMODULE
/**
 * wrapper function that creates Eigen matrices of the correct type and
 * calls the arnoldi function of the corresponding type.
 * @param A_in numpy array of dimension nxn
 * @param v_in numpy array of dimension nx1
 * @param k number of iterations
 * @param V_out numpy array of dimension nxk
 * @param H_out numpy array of dimension kxk
 */
template<class T>
void call_arnoldi(PyObject* A_in, PyObject* v_in, size_t k, PyObject* V_out, PyObject* H_out) {
	npy_intp* shape = PyArray_DIMS(A_in);
	int n = shape[0];
	Map< Eigen::Matrix< T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor > > _A_in((T *) PyArray_DATA(A_in), n,n);
	Map< Eigen::Matrix< T,Eigen::Dynamic,1 > > _v_in((T *) PyArray_DATA(v_in), n);
	Map< Eigen::Matrix< T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor > > _V_out((T *) PyArray_DATA(V_out), n, k);
	Map< Eigen::Matrix< T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor > > _H_out((T *) PyArray_DATA(H_out), k+1, k);
	// call arnoldi algorithm. This function is also templated!
	arnoldi(_A_in, _v_in, k, _V_out, _H_out);
}

void arnoldi_py(PyObject* A_in, PyObject* v_in, size_t k, PyObject* V_out, PyObject* H_out) {
	// sanity checks
	bool error = false;
	npy_intp* shape;
	int d;
	// A_in
	d = PyArray_NDIM(A_in);
	shape = PyArray_DIMS(A_in);
	if (d != 2 || shape[0] != shape[1]){
		cout << "arnoldi_py error: A is not a matrix in R^(n x n)" << endl;
		error = true;
	}
	int n = shape[0];
	// v_in
	d = PyArray_NDIM(v_in);
	shape = PyArray_DIMS(v_in);
	if (d == 0 || shape[0] != n){
		cout << "arnoldi_py error: v is not a vector in R^n" << endl;
		error = true;
	}
	// V_out
	d = PyArray_NDIM(V_out);
	shape = PyArray_DIMS(V_out);
	if (d != 2 || shape[0] != n || (unsigned int)shape[1] != k){
		cout << "arnoldi_py error: V is not a matrix in R^(n x k)" << endl;
		error = true;
	}
	// H_out
	d = PyArray_NDIM(H_out);
	shape = PyArray_DIMS(H_out);
	if (d != 2 || (unsigned int)shape[0] != k+1 || (unsigned int)shape[1] != k){
		cout << "arnoldi_py error: H is not a matrix in R^(k+1 x k)" << endl;
		error = true;
	}
	
	if (error) return;
	
	// determine numpy data type
	PyArray_Descr* descr = PyArray_DESCR(A_in);
	switch (descr->type) {
	// float32 (single precision)
	case 'f':
		call_arnoldi<float>(A_in, v_in, k, V_out, H_out);
	break;
	
	// float64 (double precision)
	case 'd':
		call_arnoldi<double>(A_in, v_in, k, V_out, H_out);
	break;
	
	// complex64 (2x single precision)
	case 'F':
		call_arnoldi<std::complex<float> >(A_in, v_in, k, V_out, H_out);
	break;
	
	// complex128 (2x double precision)
	case 'D':
		call_arnoldi<std::complex<double> >(A_in, v_in, k, V_out, H_out);
	break;
		
//// unsupported types
//	case 'e': // float 16
//	call_arnoldi<float16>(A_in, v_in, k, V_out, H_out);
//	break;
//	case 'g': // floa128 (quadruple precision)
//		call_arnoldi<float>(A_in, v_in, k, V_out, H_out);
//	break;
//	case 'G': // complex256 (2x quadruple precision)
//		call_arnoldi<complex256>(A_in, v_in, k, V_out, H_out);
//	break;
		
		// default
	default:
		cout << "arnoldi_py error: unknown type: " << descr->type << endl;
		return;
	}
}

namespace bp = boost::python;
#include "boost/python.hpp"
BOOST_PYTHON_MODULE(arnoldi) {
	bp::numeric::array::set_module_and_type("numpy", "ndarray");
	
	boost::python::def("arnoldi", arnoldi_py);
}
#endif

int main (int argc, char const *argv[])
{
	size_t n = 8;
	
	// initialize A
	Types<double>::Matrix A(n,n);
	for (unsigned int i=0; i<n; ++i) {
		A(i,i) = 2;
	}
	for (unsigned int i=1; i<n; ++i) {
		A(i,i-1) = -1;
		A(i-1,i) = -1;
	}
	cout << "A:\n" << A << endl;
	
	// initialize v
	Types<double>::Vector v(n);
	for (unsigned int i=0; i<n; ++i)
		v(i) = 1;
	
	// initialize H and V
	int m = 4;
	Types<double>::Matrix V(n,m),H(m+1,m);
	
	// call arnoldi
	arnoldi(A, v, m, V, H);
	
//	cout << "A:\n" << A << endl;
//	cout << "V:\n" << V << endl;
	cout << "H:\n" << H << endl;
	//cout << "V^T * A * V:\n" << V.transpose()*A*V << endl;
	
	// compute exponential
	Types<double>::Matrix expA(n,n);
	Types<double>::Matrix expH(m,m);
	
	Eigen::MatrixExponential<Types<double>::Matrix> expa(A);
	Eigen::MatrixExponential<Types<double>::Matrix> exph(H.block(0,0,m,m));
	expa.compute(expA); // stores result in expM
	exph.compute(expH); // stores result in expM
	//cout << "expA:\n" << expA << endl;
	//cout << "expH:\n" << expH << endl;
	
	//cout << "V^T * H * V:\n" << expA << endl;
	cout << "V^T * A * V:\n" << V.transpose()*A*V << endl;
	//cout << "V^T * expH * V:\n" << V*expH*V.transpose() << endl;
	
	return 0;
}


#endif	/* ARNOLDI_HPP */
