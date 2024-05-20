
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
    testMatrix.prepareForMultiply(1);
    
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
    
    std::cout << "Result of matrix-vector multiplication:" << std::endl;
   
    int nMatrix = 10000;
    int nThreads = 8;
    testMatrix.prepareForMultiply(1);
    
   
    testMatrix.testFullCalcMtNtime2(nThreads, Vec, Y_res,nMatrix);
    
    //testMatrix.testFullCalcMt(8, Vec, Y_res);
 
    //GxB_Matrix_fprint(A,NULL,GxB_COMPLETE, NULL);
   
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

    // Finalize GraphBLAS
    GrB_finalize();

#if 0

    //blockTest.print(std::cout);

    int nbrMatrix = 1;
    for(int i=0; i<nbrMatrix; i++)
    {
   
        MatrixSymmetric test(size);
        
        setMatrixRandomValues(test);
        //printMatrixValues(&test);
        
      
        
        for(int i=0; i<size;i++)
        {
            Y_true[i] = 0;//init
        }
        //clock_t start_1 = clock();
        //Y_third =  multi(&test,Vec,blocksize,true);
        
        //clock_t end_1 = clock();
        
        //clock_t start_2 = clock();
        test.vecMulAdd(Vec, Y_true);//calculate using boths techniques
        //clock_t end_2 = clock();
        
        //clock_t start_3 = clock();
        //test.vecMulAdd(Vec, Y_true);//calculate using boths techniques
        //cblas_dsymv(CblasRowMajor, CblasLower, 4, 1.0, test.data(), size, Vec, 1, 1.0, Y_true, 1);
        
        //cblas_dsymv(CblasRowMajor, CblasLower, 4, 1.0, test.data(), size, Vec, 1, 1.0, Y_true, 1);
        int t1 = std::time(NULL);
        
        test.vecMulPerBlock(Vec, Y_res, 4, 8);
        std::cout<<"Matrix calculation performed time : "<< t1 - std::time(NULL);
        //Y_res= multi(&test,Vec,blocksize,false);//calculate using both techniques
        //clock_t end_3 = clock();
        //const real* val = test.data();
       // matrix_multiply_4x4_neonMain(Vec, Vec, Y_thidrd, Y_diff, 4, size, val);
       //matrix_multiply_4x4_neonMain(Vec, Vec, Y_third, Y_diff, 4, size, val);
      
    }
    


 


    std::cout<<"\n\nResult of block_computation\n";
    for(int i=0; i<size;i++)
    {
        std::cout<<Y_res[i]<<" ";
    }

  

    cout<<"\n\nResult of third_computation\n";
    for(int i=0; i<size;i++)
    {
        cout<<Y_third[i]<<" ";
    }
    cout<<"\n\nDifference of true_computation with third\n";
    for(int i=0; i<size;i++)
    {
        cout<<Y_dift[i]<<" ";
    }

    auto t_1 = 1000*(end_1 - start_1)/CLOCKS_PER_SEC;
    auto t_2 = 1000*(end_2 - start_2)/CLOCKS_PER_SEC;
    auto t_3 = 1000*(end_3 - start_3)/CLOCKS_PER_SEC;
    std::cout<<"\n\nElapses times\nBlock Algorithm:"<<t_1<<"\nNative algorithm:"<<t_2<<"\n"<<"fullvecmut"<<t_3;
    


    using namespace std::chrono;
    auto start = high_resolution_clock::now();
    std::cout<<"Perf test for bothvecmul\n";
    
    real R1[4] = {1,2,3,4};
    real R2[4] = {1,2,3,4};
    real Y1[4] = {0,0,0,0};
    real Y2[4] = {0,0,0,0};
    int c1 =0;

    real Y12[4] = {0,0,0,0};
    real Y22[4] = {0,0,0,0};
    
    int c2 = 0;
    Matrix44* blockTest = new Matrix44(1,1,3,4,5,6,7,8,9,1,2,12,13,14,15,16);
  
        
    blockTest -> bothvecmulopti(R1, R2, Y12, Y22);
    
    blockTest-> bothvecmul(R1, R2, Y1, Y2);
    
    std::cout<<"Results:";
    for(int i=0; i<4;i++)
    {
        std::cout<<Y1[i]<<" ";
    }
    std::cout<<" \n";
    for(int i=0; i<4;i++)
    {
        std::cout<<Y2[i]<<" ";
    }
    std::cout<<" \n\n";
    for(int i=0; i<4;i++)
    {
        std::cout<<Y12[i]<<" ";
    }
    std::cout<<" \n";
    for(int i=0; i<4;i++)
    {
        std::cout<<Y22[i]<<" ";
    }

    std::cout<<"\nNumber executed for 1:"<<c1<<"\n";
    std::cout<<"Number executed for 2:"<<c2<<"\n";

    
#endif
    //ding of parallel region
}
