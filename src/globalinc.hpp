/* 
 * File:   globalinc.hpp
 * Author: Robert Gantner
 *
 * Created on November 20, 2011, 4:10 PM
 */

#ifndef GLOBALINC_HPP
#define	GLOBALINC_HPP

#include <vector>
#include <Eigen/Core>

//typedef typename std::vector<int> Vector;
//typedef typename std::vector<double> Matrix;
typedef double Real;

/**
 * Template classes for storing matrix and vector types.
 */
template<class T>
class Types{
public:
    // typedefs to keep code clean
    typedef typename Eigen::Matrix<T, Eigen::Dynamic, 1> Vector;
    typedef typename Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    
    // just overwrite this if you have your own matrix and vector types
    // other examples include:
    
    //typedef std::vector<T> Vector;
    //typedef boost::ndarray<T> Matrix;
};


#endif	/* GLOBALINC_HPP */

