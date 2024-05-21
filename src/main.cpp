
#include <stdio.h>
#include <stdlib.h>

#include "matsym.h"
#include <iostream>
#include "sparmatsymblk.h"
#include <random>
#include "matrix.h"
#include "blockMatrix.hpp"
#include <time.h>
#include "blockInstance.hpp"
#include "omp.h"
#include <deque> 
#include <tuple>
#include "real.h"
#include "GraphBLAS.h"
#include "cblas.h"

#define NUMBER 1*2*3*4*5*6*7*2

typedef std::tuple<int,int,int,Matrix44*> Tuple2;
typedef std::deque<Tuple2> Queue2;



void printMatrixValues(MatrixSymmetric* matrix, bool only_int = true)
{
    char str[32];
    real* val = matrix->data();
    std::cout << "\nMatrix of size : "<< matrix->size()<<"\n";

    for(int i=0; i<matrix->size(); i++)
    {
        
        
        for(int j = 0; j<matrix->size();j++)
        {
        if(!only_int){
        snprintf(str, sizeof(str), "%9.2f", *matrix->addr(i,j));
        }
        else
        {
            std::snprintf(str,sizeof(str), "%d ", (int)*matrix->addr(i,j));
        }
        std::cout << str <<"" ;
        }
        if(i!=matrix->size()-1)
        {
            std::cout<<"\n";
        }
       
    }
}
void setMatrixRandomValues(MatrixSymmetric matrix)
{
    #ifdef VERBOSE
    std::cout << "Setting random values";
    #endif
    try{

    real* val = matrix.data();
        
#ifdef VERBOSE
    std::cout << "Matrix of size : "<< matrix.size()<<"\n";
#endif
    for(int i=0; i<matrix.size(); i++)
    {
    
        for(int j = 0; j<=i;j++)
        {
                ///real value = dis(gen);
                if(i==j)
                {
                    val[i*matrix.size()+j] = 0;
                }
                else
                {
                    if(true)
                    {
                        val[i*matrix.size()+j] =(i+j)+1/(i+j);
                        val[j*matrix.size()+i] =(i+j)+1/(i+j);
                    }
                    else
                    { val[i*matrix.size()+j] = 0;}
                    
                }
                //val[i*matrix.size() + j] = (i+j)%100;
                //val[j*matrix.size() + i] = (i+j)%100;
         
        }
    }
    }
    catch(std::exception){
        std::cout << "Error loading RNG";
    }
    
}


void fillSMSB(int nbBlocks, int matsize,int blocksize, SparMatSymBlk* matrix)
{
    
    for(int i =0; i<matsize/blocksize;i++)
    {
      
        for(int j=0; j<=i; j++)
        {
            Matrix44* blockTest = new Matrix44(1,1,3,4,5,6,7,8,9,1,2,12,13,14,15,16);
            matrix->block(i,j).add_full(*blockTest);
        }
    }
}

int main(int argc, char* argv[])
{

    
    int blocksize = 4;
    size_t size = blocksize * NUMBER;
    size = 4*8*2*2*2*2*2;
    real* Vec = (real*)malloc(size * sizeof(real));//Defining vector to do MX
    real* Y_res = (real*)malloc(size * sizeof(real));//Y
    real* Y_true = (real*)malloc(size * sizeof(real));//Y_true to compare
    real* Y_third = (real*)malloc(size * sizeof(real));//Y_true to compare
    real* Y_dif = (real*)malloc(size * sizeof(real));//Y_diff that will store differences
    real* Y_diff = (real*)malloc(size * sizeof(real));//Y_diff that will store differences
    for(int i=0; i<size;i++)
    {
        Vec[i]=i;//Init
        Y_res[i] = 0;
    }



  
    SparMatSymBlk testMatrix = SparMatSymBlk();
    testMatrix.allocate(size);
    testMatrix.resize(size);
    testMatrix.reset();
    
    //testMatrix.element(1,2) += bl ockTest.value(1,2);
    fillSMSB(5, size, 4, &testMatrix);
    //testMatrix.printSummary(std::cout, 0, 25);
    
    
    omp_set_num_threads(1);
    
    GrB_init(GrB_BLOCKING);

    std::cout<<"OpenMP with" << omp_get_max_threads() <<"cores";
    std::cout<<"GraphBlas started";
    bool have_openmp ;
   
    GrB_Matrix A;
    GrB_Matrix_new(&A, GrB_FP64, size, size); // n_rows and n_cols are the dimensions of your matrix
  
    GrB_Vector v;
    GrB_Vector_new(&v, GrB_FP64, size);
    GrB_Index* I = (GrB_Index*) malloc(sizeof(GrB_Index)*size*size);
    GrB_Index* J = (GrB_Index*) malloc(sizeof(GrB_Index)*size*size);
    double* values = (double*) malloc(sizeof(GrB_Index)*size*size);
    for(int i=0;i<size;i++)
    {
        
        GrB_Vector_setElement_FP64(v, (double)(i*size)/4, (GrB_Index) i);
        for(int j=0; j<size;j++)
        {
            I[i*size +j] = i;
            J[i*size +j] = j;
            values[i*size+j] = (i+j*size)/4;
            
        }

    }
    std::cout<<GrB_Matrix_build_FP64(A,I,J,values,(GrB_Index)size*size,GxB_IGNORE_DUP);
    GrB_Vector result;
    GrB_Vector_new(&result, GrB_FP64, size);
    
    
    std::vector<double> result_values(size);
    

   
    int nMatrix = 10000;
    int nThreads = 8;
    testMatrix.prepareForMultiply(1);
    
   
    testMatrix.vecMulMt(nThreads, Vec, Y_res,nMatrix);
    for(int i=0; i<nMatrix;i++)
    {
        GrB_mxv(v, GrB_NULL, GrB_NULL, GrB_PLUS_TIMES_SEMIRING_FP64, A, v, GrB_NULL);
        //GxB_Vector_fprint(v, NULL, GxB_COMPLETE,NULL);
    }

    int nbDiff = 0;
    
    for(int i=0; i<size;i++)
    {
        Y_dif[i] = Y_true[i] - Y_res[i];
        if(Y_dif[i]!=0)
        {
            nbDiff++;
        }
        //Y_dift[i] = Y_true[i] - Y_third[i];
        
    }

if(nbDiff !=0)
{
    std::cout<<"Resultat computation originelle\n";
    for(int i =0; i< size; i++)
    {
        std::cout<<Y_true[i]<<" ";
    }
    std::cout<<"Resultat computation maison\n";
    for(int i =0; i< size; i++)
    {
        std::cout<<Y_res[i]<<" ";
    }
    std::cout<<"\n\nDifference of true_computation\n";
    for(int i=0; i<size;i++)
    {
        std::cout<<Y_dif[i]<<" ";
    }
}
else
{
        std::cout<<"\nComputation went well";
}
    
    
    GrB_Matrix_free(&A);
    GrB_Vector_free(&v);
    GrB_Vector_free(&result);
    GrB_finalize();

}