
#include <stdio.h>
#include <stdlib.h>
#include "matsym.h"
#include <iostream>
#include "sparmatsymblk.h"
#include <random>
#include "matrix.h"
#include <time.h>
#include "omp.h"
#ifdef MACOS
#include "armpl.h"
#endif
#include <deque> 
#include <tuple>
#include "real.h"


#include <chrono>
#include <fstream>
#define NUMBER 1*2*3*4*5*6*7*2


#include "custom_alg.h"


typedef std::tuple<int,int,int,Matrix44*> Tuple2;
typedef std::deque<Tuple2> Queue2;


#ifdef MACOS
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
#endif
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
    int size = 0;
    int nb_threads = 1;
    int nRepeat = 1;
    int nBlocks = 0;
    
    if (argc < 4) {
        std::cerr << "Usage: tests nb_threads matSize nRepeat" << std::endl;
        return 1;
    }
    else
    {
        try
        {
            nb_threads = std::stoi(argv[1]);
            size = std::stoi(argv[2]);
            nRepeat = std::stoi(argv[3]);
        }
        catch(std::exception e)
        {
            std::cerr << "Usage: tests nb_threads matSize nRepeat [block_percentage]" << std::endl;
            return 1;
        }
        if(size%(nb_threads*4)!= 0)
        {
            size = (int)(size / nb_threads) *nb_threads;
        }
        nBlocks = (int) size*size*0.1/16;
        if(argc >=5)
        {
            try
            {
                nBlocks = (int) size*size/16 * std::stod(argv[4]);
            }
            catch(std::exception e)
            {
                std::cerr << "block percentage not recognized as a double: usage x.f\n";
                return 1;
            }
        }
    }
   
    real* Vec = new_real(size);//Defining vector to do MX
    real* Y_res = new_real(size);//Y
    real* Y_true = new_real(size);//Y_true to compare
    real* y_arm = new_real(size);//Y_true to compare
    real* Y_dif = new_real(size);//Y_diff that will store differences
    for(int i=0; i<size;i++)
    {
        Vec[i]=i;//Init
        Y_res[i] = 0;
        Y_true[i] = 0;
    }
    

    
    //Selecting blocks
    std::vector<std::pair<int, int>> pairs = select_random_points(size/4, nBlocks);
    //Init SPSM
    SparMatSymBlk testMatrix = SparMatSymBlk();
    testMatrix.allocate(size);
    testMatrix.resize(size);
    testMatrix.reset();
    add_block_to_pos_std(&testMatrix, pairs, size);
    testMatrix.prepareForMultiply(1);
    //End of SPSM Init
    std::cout<<"Constructed matrix of size "<<size<<" with "<< nBlocks <<" blocks of size 4, preparing to do "<<nRepeat<<" multiplications";
    
    omp_set_num_threads(nb_threads);

    std::cout<<"\n[STARTUP] OpenMP is enabled with " << omp_get_max_threads() <<" threads\n";
    std::ofstream outfile1;
    
    #ifdef MACOS
    std::cout<<"[STARTUP] ARM PL is working\n";
    
    outfile1.open("res/armpl.csv", std::ios::app);
    std::cout<<"[INFO] ARMPL... ";
    auto start = std::chrono::high_resolution_clock::now();
    //y_arm = amd_matrix_vecmul(size, nRepeat, pairs);
    auto stop= std::chrono::high_resolution_clock::now();
    std::cout<<"   "<<std::chrono::duration_cast<milli>(stop - start).count()<<" ms\n";
    //outfile1 << std::chrono::duration_cast<milli>(stop - start).count()<<",";
    outfile1.close();
    #endif

    std::cout<<"[INFO] " << testMatrix.what() << " ";

    outfile1.open("res/standard.csv", std::ios::app);
    auto start = std::chrono::high_resolution_clock::now();
  
    auto stop= std::chrono::high_resolution_clock::now();
    
    outfile1 << std::chrono::duration_cast<milli>(stop - start).count()<<",";
    outfile1.close();
    std::cout<<" "<<std::chrono::duration_cast<milli>(stop - start).count()<<" ms\n";

    
    std::ofstream outfile2;
    outfile2.open("res/newImpl.csv", std::ios::app);
    std::cout<<"[INFO] MULTI...";
    start = std::chrono::high_resolution_clock::now();
    testMatrix.vecMulMt2(nb_threads, Vec, Y_res,nRepeat);
    stop = std::chrono::high_resolution_clock::now();
    outfile2 << std::chrono::duration_cast<milli>(stop - start).count()<<",";
    outfile2.close();
    std::cout<<"   "<<std::chrono::duration_cast<milli>(stop - start).count()<<" ms";

    
    
   


    //Arm sparse linear algebra lib
     


    int nbDiff = 0;
    
    for(int i=0; i<size;i++)
    {
        Y_dif[i] = Y_res[i] - Y_true[i];
        nbDiff += ( Y_dif[i] != 0 );
    }
    
    if(nbDiff !=0)
    {
        std::cout<<"Resultat computation originelle\n";
        for(int i =0; i< size; i++)
            std::cout<<Y_true[i]<<" ";
        std::cout<<"Resultat computation maison\n";
        for(int i =0; i< size; i++)
            std::cout<<Y_res[i]<<" ";
        std::cout<<"\n\nDifference of true_computation\n";
        for(int i=0; i<size;i++)
            std::cout<<Y_dif[i]<<" ";
    }
    else
    {
        std::cout<<"\nComputation went well\n";
    }
}
