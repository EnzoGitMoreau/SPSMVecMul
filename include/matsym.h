// Cytosim was created by Francois Nedelec. Copyright 2020 Cambridge University.

#ifndef MATSYM_H
#define MATSYM_H
#define NB_THREADS 2

//Nb thread = 6 doesnt work
//Nb thread = 4 doesnt work
#include "real.h"

#include <cstdio>

#include <string>


class MatrixSymmetric final
{
private:
    
    /// leading dimension of array
    size_t dimension_;
    
    /// size of matrix
    size_t size_;

    /// size of memory which has been allocated
    size_t allocated_;
    
    // full upper triangle:
    real* val;
    
    // if 'false', destructor will not call delete[] val;
    bool in_charge;
    
public:
    
    /// return the size of the matrix
    size_t size() const { return size_; }
    
    /// change the size of the matrix
    void resize(size_t s) { allocate(s); size_=s; }

    /// base for destructor
    void deallocate();
    
    /// default constructor
    MatrixSymmetric();
    
    
    /// constructor from an existing array
    MatrixSymmetric(size_t s)
    {
        val = nullptr;
        resize(s);
        dimension_ = s;
        val = new_real(s*s);
        zero_real(s*s, val);
        in_charge = true;
    }

    /// constructor from an existing array
    MatrixSymmetric(size_t s, real* array, size_t ldd)
    {
        free_real(val);
        size_ = s;
        dimension_ = ldd;
        val = array;
        in_charge = false;
    }
    
    /// default destructor

    /// set to zero
    void reset();
    
    /// allocate the matrix to hold ( sz * sz )
    void allocate(size_t alc);
    
    /// returns address of data array
    real* data() const { return val; }

    /// returns the address of element at (x, y), no allocation is done
    real* addr(size_t x, size_t y) const;
    
    /// returns the address of element at (x, y), allocating if necessary
    real& operator()(size_t i, size_t j);
    
    /// scale the matrix by a scalar factor
    void scale(real a);
    
    /// multiplication of a vector: Y = Y + M * X, dim(X) = dim(M)
    void vecMulAdd(const real* X, real* Y) const;
    
    /// 2D isotropic multiplication of a vector: Y = Y + M * X
    void vecMulAddIso2D(const real* X, real* Y) const;
    
    /// 3D isotropic multiplication of a vector: Y = Y + M * X
    void vecMulAddIso3D(const real* X, real* Y) const;
    void vecMulAddBlock(const real* X, real*Y, int index_x, int index_j, int blocksize, int matsize) const;
    void transVecMulAddBlock(const real* X, real*Y, int index_x, int index_j, int blocksize, int matsize) const;
    void vecMulAddBlock2(const real * __restrict__ X, real *__restrict__ Y, int index_x, int index_j, int blocksize, int matsize) const;
    void transVecMulAddBlock2(const real* X, real*Y1, real*Y2, int index_x1, int index_y1, int blocksize, int matsize) const;
    void transVecMulAddBlock3(const real* __restrict__ X1, const real* __restrict__ X2,real* __restrict__ Y1,  real* __restrict__ Y2, int blocksize, int matsize, const real* __restrict__ valptr )const;
    /// true if matrix is non-zero
    void transVecMulAddBlock4(const real* __restrict__ X1, const real* __restrict__ X2,real* __restrict__ Y1,  real* __restrict__ Y2, int blocksize, int matsize, const real* __restrict__ valptr )const;
   

#endif
