
#include <stdio.h>
#include <stdlib.h>

#include "matsym.h"
#include <iostream>
#include "sparmatsymblk.h"
#include <random>
#include "matrix.h"
#include <time.h>
#include "omp.h"
#include "armpl.h"
#include <deque> 
#include <tuple>
#include "real.h"


#include <chrono>
#include <fstream>
#define NUMBER 1*2*3*4*5*6*7*2


#include "custom_alg.h"


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

double* amd_matrix_vecmul(int size, int nTests, std::vector<std::pair<int, int>> pairs)
{
    double* values = (double*) malloc(sizeof(double)*size*size);
    add_block_to_pos(values,pairs,size);
    
    
    armpl_spmat_t armpl_mat;
    armpl_int_t creation_flags = 0;
    armpl_status_t info = armpl_spmat_create_dense_d(&armpl_mat,ARMPL_COL_MAJOR,size, size, size, values,creation_flags);
    if (info!=ARMPL_STATUS_SUCCESS)
          printf("ERROR: armpl_spmat_create_csr_d returned %d\n", info);

    /* 3a. Supply any pertinent information that is known about the matrix */
    info = armpl_spmat_hint(armpl_mat, ARMPL_SPARSE_HINT_STRUCTURE,
                          ARMPL_SPARSE_STRUCTURE_UNSTRUCTURED);
    if (info!=ARMPL_STATUS_SUCCESS)
          printf("ERROR: armpl_spmat_hint returned %d\n", info);

    /* 3b. Supply any hints that are about the SpMV calculations
         to be performed */
    info = armpl_spmat_hint(armpl_mat, ARMPL_SPARSE_HINT_SPMV_OPERATION,
                          ARMPL_SPARSE_OPERATION_NOTRANS);
    if (info!=ARMPL_STATUS_SUCCESS)
          printf("ERROR: armpl_spmat_hint returned %d\n", info);

    info = armpl_spmat_hint(armpl_mat, ARMPL_SPARSE_HINT_SPMV_INVOCATIONS,
                          ARMPL_SPARSE_INVOCATIONS_MANY);
    if (info!=ARMPL_STATUS_SUCCESS)
          printf("ERROR: armpl_spmat_hint returned %d\n", info);

    /* 4. Call an optimization process that will learn from the hints you
        have previously supplied */
    info = armpl_spmv_optimize(armpl_mat);
    if (info!=ARMPL_STATUS_SUCCESS)
          printf("ERROR: armpl_spmv_optimize returned %d\n", info);
    double *x = (double *)malloc(size*sizeof(double));
    for (int i=0; i<size; i++) {
            x[i] = i;
    }
    double *y = (double *)malloc(size*sizeof(double));
    for (int i=0; i<nTests; i++) {
            info = armpl_spmv_exec_d(ARMPL_SPARSE_OPERATION_NOTRANS, 1.0,
                             armpl_mat, x, 1.0, y);
            if (info!=ARMPL_STATUS_SUCCESS)
            printf("ERROR: armpl_spmv_exec_d returned %d\n", info);
    }

   
    armpl_spmat_destroy(armpl_mat);
    return y;
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

    using milli = std::chrono::milliseconds;
    int blocksize = 4;
    int size;
    int nb_threads;
    int nMatrix = 1;
    double block_percentage = 0.10;
    if (argc < 4) {
        std::cerr << "Usage: tests nb_threads matSize" << std::endl;
        
        return 1;
        }
        else
    {
        try
        {
            size = std::stoi(argv[2]);
            nb_threads = std::stoi(argv[1]); 

            
            nMatrix = std::stoi(argv[3]);
        }
        catch(std::exception e)
        {
            std::cerr << "Usage: tests nb_threads matSize nMatrix [block_percentage]" << std::endl;
            return 1;
        }
        if(size%(nb_threads*4)!= 0)
        {
            size = (int)(size / nb_threads) *nb_threads;
           
        }
        if(argc >=5)
        {
        try
        {
            block_percentage = std::stod(argv[4]);
        }
        catch(std::exception e)
        {
            std::cerr << "block percentage not recognized as a double: usage x.f";
            return 1;
        }
        }
    }
   
    real* Vec = (real*)malloc(size * sizeof(real));//Defining vector to do MX
    real* Y_res = (real*)malloc(size * sizeof(real));//Y
    real* Y_true = (real*)malloc(size * sizeof(real));//Y_true to compare
    real* y_arm = (real*)malloc(size * sizeof(real));//Y_true to compare
    real* Y_dif = (real*)malloc(size * sizeof(real));//Y_diff that will store differences
    for(int i=0; i<size;i++)
    {
        Vec[i]=i;//Init
        Y_res[i] = 0;
        Y_true[i] = 0;
    }
    

    
    //Selecting blocks
    std::vector<std::pair<int, int>> pairs = select_random_points(size/4,(int) size*size/16 * block_percentage);
    //Init SPSM
    SparMatSymBlk testMatrix = SparMatSymBlk();
    testMatrix.allocate(size);
    testMatrix.resize(size);
    testMatrix.reset();
    add_block_to_pos_std(&testMatrix, pairs, size);
    testMatrix.prepareForMultiply(1);
    //End of SPSM Init
    std::cout<<"Constructed matrix of size "<<size<<" with "<<(int) size*size/16 * block_percentage <<" blocks of size 4, preparing to do "<<nMatrix<<" multiplications";
    
    omp_set_num_threads(nb_threads);

    std::cout<<"\n[STARTUP] OpenMP is enabled with " << omp_get_max_threads() <<" threads\n";
    std::cout<<"[STARTUP] ARM PL is working\n";
    std::ofstream outfile1;
    outfile1.open("res/armpl.csv", std::ios::app);
    std::cout<<"[INFO] Starting ARMPL matrix-vector multiplications\n";
    auto start = std::chrono::high_resolution_clock::now();
    y_arm = amd_matrix_vecmul(size, nMatrix, pairs);
    auto stop= std::chrono::high_resolution_clock::now();
    std::cout<<"[INFO] ARMPL multiplications done in "<<std::chrono::duration_cast<milli>(stop - start).count()<<" ms\n";
    outfile1 << std::chrono::duration_cast<milli>(stop - start).count()<<",";
    outfile1.close();

    std::cout<<"[INFO] Starting Cytosim matrix-vector multiplications\n";
    
   
    outfile1.open("res/standard.csv", std::ios::app);
    start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<nMatrix;i++)
    {
        testMatrix.vecMulAdd(Vec, Y_true);
    }
    
    stop= std::chrono::high_resolution_clock::now();
    
    outfile1 << std::chrono::duration_cast<milli>(stop - start).count()<<",";
    outfile1.close();
    std::cout<<"[INFO] Cytosim multiplications done in "<<std::chrono::duration_cast<milli>(stop - start).count()<<" ms\n";

    
    std::ofstream outfile2;
    outfile2.open("res/newImpl.csv", std::ios::app);
    std::cout<<"[INFO] Starting new-impl matrix-vector multiplications\n";
    using milli = std::chrono::milliseconds;
    start = std::chrono::high_resolution_clock::now();
    testMatrix.vecMulMt2(nb_threads, Vec, Y_res,nMatrix);
    stop = std::chrono::high_resolution_clock::now();
    outfile2 << std::chrono::duration_cast<milli>(stop - start).count()<<",";
    outfile2.close();
    std::cout<<"[INFO] new-impl multiplications done in "<<std::chrono::duration_cast<milli>(stop - start).count()<<" ms";

    
    
   


    //Arm sparse linear algebra lib
     


    int nbDiff = 0;
    
    for(int i=0; i<size;i++)
    {
        Y_dif[i] = Y_res[i] - Y_true[i];
        if(Y_dif[i]!=0)
        {
            nbDiff++;
        }
       
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
    
    
 
}