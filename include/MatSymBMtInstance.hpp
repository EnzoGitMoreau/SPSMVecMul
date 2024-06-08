//
//  MatSymBMtInstance.hpp
//  MatrixCalculation
//
//  Created by Moreau Enzo on 17/04/2024.
//

#ifndef MatSymBMtInstance_hpp
#define MatSymBMtInstance_hpp

#include <stdio.h>
#include <arm_neon.h>
#include "sparmatsymblk.h"
#include "real.h"
#include "boost/thread.hpp"
#include <boost/thread/barrier.hpp>


class MatSymBMtInstance final
{
    const static int BLOCKSIZE = 4;

private:
    int phase_number;
    int* phase_tab;
    int nbThreads;
    boost::mutex _mutex;
    int big_mat_size;
    real* Y1;
    real* Y2;
    const real* X;
    SparMatSymBlk* matrix;
    int thNb = 0;
    real*** workingPhases;
    int*** workingIndexes;
    int** firstBlocks;
    unsigned short*** workingIndexes7;
    int** work_lengths;
    
public:
    
    MatSymBMtInstance(SparMatSymBlk* mat, int nbThreadsE)
    {
        nbThreads = nbThreadsE;
        this->matrix = mat;
        big_mat_size = mat->size();
        real* big_Y = (real*) malloc(big_mat_size*2*sizeof(real));
        Y1 = big_Y;
        Y2 = big_Y + big_mat_size;
        for(int i =0; i<big_mat_size;i++)
        {
            Y1[i] = 0;
            Y2[i] = 0;
        }
    }
    
    /// set phase_tab, which is ???
    void setPhaseTabs(int nbt)
    {
        for ( int i = 0; i <= nbt/2; ++i )
        {
            phase_tab[nbt-i] = i;
            phase_tab[i] = i;
        }
    }
    
    void generateBlocks2()
    {
        phase_tab = (int*) malloc(sizeof(int)*nbThreads);
        phase_tab[0] = 0;
        if(nbThreads %2 == 0)
        {
            phase_number = nbThreads /2 +1;
            for(int i =1; i<=nbThreads/2;i++)
            {
                phase_tab[i] = i;
                //std::cout << phase_tab[i] << " ";
            }
            for(int i=1 ; i<nbThreads/2;i++)
            {
                phase_tab[nbThreads/2 + i] = nbThreads/2 - i;
                //std::cout << phase_tab[nbThreads/2 + i] << " ";
            }
        }
        else
        {
          
            phase_number  = (nbThreads +1) / 2;
            if(nbThreads == 3)
            {
                phase_tab[0] = 0;
                phase_tab[1] = 1;
                phase_tab[2] = 1;
            }
            if(nbThreads == 5)
            {
                phase_tab[0] = 0;
                phase_tab[1] = 1;
                phase_tab[2] = 2;
                phase_tab[3] = 2;
                phase_tab[4] = 1;

            }
            if(nbThreads == 7)
            {
                phase_tab[0] = 0;
                phase_tab[1] = 1;
                phase_tab[2] = 2;
                phase_tab[3] = 3;
                phase_tab[4] = 3;
                phase_tab[5] = 2;
                phase_tab[6] = 1;

            }
             if(nbThreads == 9)
            {
                phase_tab[0] = 0;
                phase_tab[1] = 1;
                phase_tab[2] = 2;
                phase_tab[3] = 3;
                phase_tab[4] = 4;
                phase_tab[5] = 4;
                phase_tab[6] = 3;
                phase_tab[7] = 2;
                phase_tab[8] = 1;

            }
             if(nbThreads == 11)
            {
                phase_tab[0] = 0;
                phase_tab[1] = 1;
                phase_tab[2] = 2;
                phase_tab[3] = 3;
                phase_tab[4] = 4;
                phase_tab[5] = 5;
                phase_tab[6] = 5;
                phase_tab[7] = 4;
                phase_tab[8] = 3;
                phase_tab[9] = 2;
                phase_tab[10] = 1;

            }
        }
        
        workingPhases = (real***) malloc(sizeof(real**)*phase_number);
        workingIndexes = (int***)malloc(sizeof(int**)*phase_number);
        
        QueueBlock*** phasesTemp = (QueueBlock***) malloc(sizeof(QueueBlock**) * phase_number);
        int threadBlockSize = (int) matrix->size() / (S_BLOCK_SIZE * nbThreads);
        int* phase_counter = (int*) malloc(sizeof(int)*phase_number);
        work_lengths = (int**) malloc(sizeof(int*)*phase_number);
        for(int i=0; i<phase_number;i++)
        {
            work_lengths[i] = (int*) malloc(sizeof(int)*nbThreads);
        }
        //Init
        
        for(int i =0; i<phase_number; i++)
        {
            workingPhases[i] = (real**) malloc(sizeof(real*) * nbThreads);
            workingIndexes[i] = (int**) malloc(sizeof(int*) * nbThreads);
            
            phasesTemp[i] = (QueueBlock**) malloc(sizeof(QueueBlock*)*nbThreads);
            for(int j=0; j<nbThreads; j++)
            {
                phasesTemp[i][j] = new QueueBlock();
            }
        }
        
        int i =0;
        while(true)//Fetching blocks, assigned them to a phase each
        {
            int indX, indY;
            SparMatSymBlk::Column* col = &matrix->column_[i];
            if(col->nbb_>0)
            {
                //std::cout<<"Column number "<<matrix->colidx_[i]<<"->ind_YB : "<<(int) matrix->colidx_[i]/threadBlockSize<<"\n";
                indY = matrix->colidx_[i]/threadBlockSize;
                int indice_col = matrix->colidx_[i];
                
                for(int j=0; j<col->nbb_; j++)
                {
                    int indice_ligne = col->inx_[j];
                    //std::cout<<"Block position :"<<col->inx_[j]<<"-> ind_XB : "<<(int) col->inx_[j]/threadBlockSize<<"\n";
                    indX =col->inx_[j]/threadBlockSize;
                    //std::cout<<"Put in phase: "<<phase_tab[indX-indY]<<"\n";
                    int swap = 0;
                    if(indX - indY >= 1+ (int)(nbThreads/2))
                    {
                        swap =1;
                    }
                    //swap = 0; // Le swap ne fonctionne pas encore.
                    int phase =phase_tab[indX-indY];
                    if(phase>0)
                    {
                        if(swap ==1)
                        {
                            // std::cout<<"\nWARNINGINININING\n";
                        }
                    }
                    //phasesTemp[phase][phase_counter[phase]%nbThreads]->push_front(Tuple2( indice_ligne, indice_col,swap, &matrix->block(col->inx_[j],matrix->colidx_[i])));
                    Block data;
                    Matrix44 mat= col->blk_[j];
                    data.a0 = mat.val[0x0];
                    data.a1 = mat.val[0x1];
                    data.a2 = mat.val[0x2];
                    data.a3 = mat.val[0x3];
                    data.a4 = mat.val[0x4];
                    data.a5 = mat.val[0x5];
                    data.a6 = mat.val[0x6];
                    data.a7 = mat.val[0x7];
                    data.a8 = mat.val[0x8];
                    data.a9 = mat.val[0x9];
                    data.aA = mat.val[0xA];
                    data.aB = mat.val[0xB];
                    data.aC = mat.val[0xC];
                    data.aD = mat.val[0xD];
                    data.aE = mat.val[0xE];
                    data.aF = mat.val[0xF];
                    
                    data.index_x = indice_ligne * BLOCKSIZE;
                    data.index_y = indice_col * BLOCKSIZE;
                    
                    if(swap==0)
                    {
                        
                        phasesTemp[phase][indX]->push_front(data);
                    }
                    else
                    {
                        int temp = data.index_x;
                        data.index_x = data.index_y;
                        data.index_y = temp;
                        phasesTemp[phase][indY]->push_front(data);
                    }
                    
                    phase_counter[phase]++;
                    
                }
                //std::cout<<"\n\n";
            }
            
            i = matrix->colidx_[i+1];
            if(i>=matrix->rsize_)
            {
                
                break;
            }
            
        }
        
        for(int i=0; i<phase_number; i++)//Putting in order for threads
        {
            for(int j=0; j<nbThreads;j++)
            {
                
                
                int t_work_length = (int)phasesTemp[i][j]->size();
                
                work_lengths[i][j] = t_work_length;
                
                workingPhases[i][j] = (real*) malloc(sizeof(real)*t_work_length*16);
                workingIndexes[i][j] = (int*) malloc(sizeof(int)*t_work_length*2);
                
                
                for(int m=0; m<t_work_length; m++)
                {
                    
                   
                    Block first= phasesTemp[i][j]->front();
                    phasesTemp[i][j]->pop_front();
                    
                    workingPhases[i][j][16*m] = first.a0;
                    workingPhases[i][j][16*m+1] = first.a1;
                    workingPhases[i][j][16*m+2] = first.a2;
                    workingPhases[i][j][16*m+3] = first.a3;
                    workingPhases[i][j][16*m+4] = first.a4;
                    workingPhases[i][j][16*m+5] = first.a5;
                    workingPhases[i][j][16*m+6] = first.a6;
                    workingPhases[i][j][16*m+7] = first.a7;
                    workingPhases[i][j][16*m+8] = first.a8;
                    workingPhases[i][j][16*m+9] = first.a9;
                    workingPhases[i][j][16*m+10] = first.aA;
                    workingPhases[i][j][16*m+11] = first.aB;
                    workingPhases[i][j][16*m+12] = first.aC;
                    workingPhases[i][j][16*m+13] = first.aD;
                    workingPhases[i][j][16*m+14] = first.aE;
                    workingPhases[i][j][16*m+15] = first.aF;
                    workingIndexes[i][j][2*m] = first.index_x;
                    workingIndexes[i][j][2*m+1] = first.index_y;
                   // std::cout<<"Block sent : ("<<first.index_x<<", "<<first.index_y<<")"<<"\n";
                    
                    
                }
                
                
            }
        }
    }
    
    
    void generateBlocks()
    {
        phase_tab = (int*) malloc(sizeof(int)*nbThreads);
        phase_tab[0] = 0;
        if(nbThreads %2 == 0)
        {

            phase_number = nbThreads /2 +1;
            for(int i =1; i<=nbThreads/2;i++)
            {
                phase_tab[i] = i;
            
            }
            for(int i=1 ; i<nbThreads/2;i++)
            {
                phase_tab[nbThreads/2 + i] = nbThreads/2 - i;
              
            }
        }
        else
        {
       
            phase_number  = (nbThreads +1) / 2;
            for(int i =1; i<= (nbThreads - 1) /2; i++)
            {
                phase_tab[i] = i;

               
            }
            for(int i=0; i< (nbThreads -1 )/ 2; i++)
            {
                phase_tab[(nbThreads+1)/2 +i] = (nbThreads-1)/2 -i;
               
            }
        }
        
        workingPhases = (real***) malloc(sizeof(real**)*phase_number);
        workingIndexes7 = (unsigned short***)malloc(sizeof(unsigned short**)*phase_number);
        firstBlocks = (int**)malloc(sizeof(int*)*phase_number);
        QueueBlock*** phasesTemp = (QueueBlock***) malloc(sizeof(QueueBlock**) * phase_number);
        QueueBlock*** phasesTempSwap = (QueueBlock***) malloc(sizeof(QueueBlock**) * phase_number);
        int threadBlockSize = (int) matrix->size() / (S_BLOCK_SIZE * nbThreads);
        int* phase_counter = (int*) malloc(sizeof(int)*phase_number);
        work_lengths = (int**) malloc(sizeof(int*)*phase_number);
        for(int i=0; i<phase_number;i++)
        {
            work_lengths[i] = (int*) malloc(sizeof(int)*nbThreads);
        }
        //Init
        
        for(int i =0; i<phase_number; i++)
        {
            workingPhases[i] = (real**) malloc(sizeof(real*) * nbThreads);
            workingIndexes7[i] = (unsigned short**) malloc(sizeof(unsigned short*) * nbThreads);
            firstBlocks[i] = (int*)malloc(sizeof(int)*nbThreads);
            phasesTemp[i] = (QueueBlock**) malloc(sizeof(QueueBlock*)*nbThreads);
            phasesTempSwap[i] = (QueueBlock**) malloc(sizeof(QueueBlock*)*nbThreads);
            for(int j=0; j<nbThreads; j++)
            {
                phasesTemp[i][j] = new QueueBlock();
                phasesTempSwap[i][j] = new QueueBlock();
            }
        }
        
        int i =0;
        while(true)//Fetching blocks, assigned them to a phase each
        {
            int indX, indY;
            SparMatSymBlk::Column* col = &matrix->column_[i];
            if(col->nbb_>0)
            {
                //std::cout<<"Column number "<<matrix->colidx_[i]<<"->ind_YB : "<<(int) matrix->colidx_[i]/threadBlockSize<<"\n";
                indY = matrix->colidx_[i]/threadBlockSize;
                int indice_col = matrix->colidx_[i];
                
                for(int j=0; j<col->nbb_; j++)
                {
                    int indice_ligne = col->inx_[j];
                    //std::cout<<"Block position :"<<col->inx_[j]<<"-> ind_XB : "<<(int) col->inx_[j]/threadBlockSize<<"\n";
                    indX =col->inx_[j]/threadBlockSize;
                    //std::cout<<"Put in phase: "<<phase_tab[indX-indY]<<"\n";
                    int swap = 0;
                    if(indX - indY >= 1+ (int)(nbThreads/2))
                    {
                        swap =1;
                    }
                    //swap = 0; // Le swap ne fonctionne pas encore.
                    int phase =phase_tab[indX-indY];
                    if(phase>0)
                    {
                        if(swap ==1)
                        {
                            // std::cout<<"\nWARNINGINININING\n";
                        }
                    }
                    //phasesTemp[phase][phase_counter[phase]%nbThreads]->push_front(Tuple2( indice_ligne, indice_col,swap, &matrix->block(col->inx_[j],matrix->colidx_[i])));
                    Block data;
                    Matrix44 mat= col->blk_[j];
                    data.a0 = mat.val[0x0];
                    data.a1 = mat.val[0x1];
                    data.a2 = mat.val[0x2];
                    data.a3 = mat.val[0x3];
                    data.a4 = mat.val[0x4];
                    data.a5 = mat.val[0x5];
                    data.a6 = mat.val[0x6];
                    data.a7 = mat.val[0x7];
                    data.a8 = mat.val[0x8];
                    data.a9 = mat.val[0x9];
                    data.aA = mat.val[0xA];
                    data.aB = mat.val[0xB];
                    data.aC = mat.val[0xC];
                    data.aD = mat.val[0xD];
                    data.aE = mat.val[0xE];
                    data.aF = mat.val[0xF];
                    
                    data.index_x = indice_ligne * BLOCKSIZE;
                    data.index_y = indice_col * BLOCKSIZE;
                    
                    if(swap==0)
                    {
                        //std::cout<<data.index_x<<","<<data.index_y<<"\n";
                        phasesTemp[phase][indX]->push_front(data);
                    }
                    else
                    {
                        //std::cout<<data.index_x<<","<<data.index_y<<"\n";
                        int temp = data.index_x;
                        data.index_x = data.index_y;
                        data.index_y = temp;
                        
                        phasesTemp[phase][indY]->push_front(data);
                    }
                    
                    phase_counter[phase]++;
                    
                }
                //std::cout<<"\n\n";
            }
            
            i = matrix->colidx_[i+1];
            if(i>=matrix->rsize_)
            {
                
                break;
            }
            
        }
        
        int relevantSize = (int) matrix->size() / nbThreads;
        for(int i=0; i<phase_number; i++)//Putting in order for threads
        {
            
            for(int j=0; j<nbThreads;j++)
            {
                
                
                int t_work_length = (int)phasesTemp[i][j]->size();
                
                
                work_lengths[i][j] = t_work_length;
                
                //std::cout<<"wl1:"<<t_work_length1<<" wl2:"<<t_work_length2<<"\n";
                workingPhases[i][j] = (real*) malloc(sizeof(real)*t_work_length*16);
                workingIndexes7[i][j] = (unsigned short*) malloc(sizeof(unsigned short)*t_work_length*2);
                
                //threadBlockSize
                
                //xi + yi*(thread_blocksize * BLOCKSIZE) = ni
                //xi = ni%(thead_blocksize * BLOCKSIZE)
                //yi = ni/(thread_blocksize * BLOCKSIZE)
                //xi+1 - xi + (yi+1 - yi)*thread_blocksize = ni+1 - ni
                //
                
                int last_x = 0;
                int last_y =0;
                for(int m=0; m<t_work_length; m++)//ajouter les blocks swappes
                {
                    Block first= phasesTemp[i][j]->back();
                    phasesTemp[i][j]->pop_back();
                    
                    workingPhases[i][j][16*m] = first.a0;
                    workingPhases[i][j][16*m+1] = first.a1;
                    workingPhases[i][j][16*m+2] = first.a2;
                    workingPhases[i][j][16*m+3] = first.a3;
                    workingPhases[i][j][16*m+4] = first.a4;
                    workingPhases[i][j][16*m+5] = first.a5;
                    workingPhases[i][j][16*m+6] = first.a6;
                    workingPhases[i][j][16*m+7] = first.a7;
                    workingPhases[i][j][16*m+8] = first.a8;
                    workingPhases[i][j][16*m+9] = first.a9;
                    workingPhases[i][j][16*m+10] = first.aA;
                    workingPhases[i][j][16*m+11] = first.aB;
                    workingPhases[i][j][16*m+12] = first.aC;
                    workingPhases[i][j][16*m+13] = first.aD;
                    workingPhases[i][j][16*m+14] = first.aE;
                    workingPhases[i][j][16*m+15] = first.aF;
                    
                    if(m !=0)
                    {
                        if(first.index_y <= first.index_x)
                        {
                            workingIndexes7[i][j][m] = first.index_x-last_x + (first.index_y-last_y) * relevantSize;
                        }
                        else
                        {
                            workingIndexes7[i][j][m] = first.index_y-last_y + (first.index_x-last_x) * relevantSize;
                        }
                    }
                else
                    {
                        workingIndexes7[i][j][m] = 0;
                        firstBlocks[i][j] = first.index_x + 65535*first.index_y;
                    }
                if(j>100)
                    {
                        
                            // std::cout<<first.index_x-last_x + (first.index_y-last_y) * relevantSize<<"\n";
                            std::cout<<"Block sent: ("<<first.index_x<<","<<first.index_y<<")"<<"pad:"<< workingIndexes7[i][j][m]<<"th:"<<j<<"\n";
                            
                            if(m ==0)
                            {
                                std::cout<<"value sent :"<<first.index_x + 65535*first.index_y<<"\n";
                            }
                        
                    }
                  //  std::cout<<"lastix:"<<last_x<<"lastiy:"<<last_y<<"\n";
                    
                   // std::cout<<"ix:"<<first.index_x<<"iy:"<<first.index_y<<"\n\n";
                    last_x = first.index_x;
                    last_y = first.index_y;
                    
                }
            }
        }
    }
    
    int thread_number()
    {
        return thNb++;
    }
   
    
    void work(boost::barrier& barrier,boost::barrier& barrier2, int n_work)
    {
        _mutex.lock();
        int thread_nb = thread_number();
        //std::cout<<"Thread number:  "<<thread_nb<<"\n";
        _mutex.unlock();
        
        for(int m = 0; m<n_work;m++)
        {
            workThread(barrier2, thread_nb);
            barrier.wait();
        }
    }
    void work2(boost::barrier& barrier,boost::barrier& barrier2, int n_work)
    {
        _mutex.lock();
        int thread_nb = thread_number();
        //std::cout<<"Thread number:  "<<thread_nb<<"\n";
        _mutex.unlock();
        
        for(int m = 0; m<n_work;m++)
        {
            workThread2(barrier2, thread_nb);
            barrier.wait();
        }
     }
    
    void workThread(boost::barrier& barrier2, int thID)
    {
        
        
        int k = 0;
        real* work;
        int work_nb;
        int relevantSize = (int) matrix->size() / nbThreads;
        const real* X1;
        const real* X2;
        real* Y1_;
        real* Y2_;
        real y20 = 0;
        real y21 = 0;
        real y22 = 0;
        real y23=0;
        real y10 = 0;
        real y12=0;
        real y13=0;
        real y11 = 0;
        real r10 = 0;
        real r11 = 0;
        real r12 = 0;
        real r13 = 0;
        real r20 = 0;
        real r21 = 0;
        real r22 = 0;
        real r23 = 0;
        while(k<phase_number)
        {
            //std::cout<<"\nStart of phase"<<k<<"\n";
            
            _mutex.lock();
            work =  workingPhases[k][thID];
            work_nb = work_lengths[k][thID];
            unsigned short* index = workingIndexes7[k][thID];
            
            int ix_f = firstBlocks[k][thID]%65535;
            int iy_f = firstBlocks[k][thID]/65535;
            _mutex.unlock();
            
            int pad = 0;
            short dist = ix_f - iy_f;
            bool change;
            int ix = 0;
            int iy =0;
           
            for(int m = 0; m<work_nb; m++)
            {
                pad += index[m];
                
                if(dist <0)
                {
                    iy = iy_f+ pad%relevantSize;
                    change = (iy+index[m]-iy_f >= relevantSize) || m ==0;
                    ix = ix_f+ (int) pad/relevantSize;
                    
                }
                else
                {
                    change = (ix+index[m]-ix_f >= relevantSize) || m ==0;
                    ix = ix_f+ pad%relevantSize;
                    iy = iy_f+ (int) pad/relevantSize;
                }
                
                real a0 = work[16*m+0];
                real a1 = work[16*m+1];
                real a2 = work[16*m+2];
                real a3 = work[16*m+3];
                real a4 = work[16*m+4];
                real a5 = work[16*m+5];
                real a6 = work[16*m+6];
                real a7 = work[16*m+7];
                real a8 = work[16*m+8];
                real a9 = work[16*m+9];
                real aA = work[16*m+10];
                real aB = work[16*m+11];
                real aC = work[16*m+12];
                real aD = work[16*m+13];
                real aE = work[16*m+14];
                real aF = work[16*m+15];
               
                if(m!=0)
                    {
                        if(dist>=0)//Cas de base, on it�re sur x
                        {
                            if(!change)
                                {
                                    
                                }
                                else//Je change Y2 et X1, j'applique donc les r�sultats des calculs pr�cendents
                                {
                                    Y2_[0] += y20;
                                    Y2_[1] += y21;
                                    Y2_[2] += y22;
                                    Y2_[3] += y23;
                                    Y2_= Y2 + iy;
                                    X1 = X  + iy;
                                    y20 = 0;
                                    y21 = 0;
                                    y22 = 0;
                                    y23=0;
                                    r10 = X1[0];
                                    r11 = X1[1];
                                    r12 = X1[2];
                                    r13 = X1[3];
                                }
                                Y1_[0] += y10;
                                Y1_[1] += y11;
                                Y1_[2] += y12;
                                Y1_[3] += y13;
                                X2 = X  + ix;
                                Y1_= Y1 + ix;
                                r20 = X2[0];
                                r21 = X2[1];
                                r22 = X2[2];
                                r23 = X2[3];
                                y10 =0;
                                y11 =0;
                                y12=0;
                                y13=0;
                            }
                        else
                        {
                            if(!change)//Je change Y2 et X1, j'applique donc les r�sultats des calculs pr�cendents
                                {}
                                else
                                {
                                    Y2_[0] += y20;
                                    Y2_[1] += y21;
                                    Y2_[2] += y22;
                                    Y2_[3] += y23;
                                    
                                    X1 = X  + ix;
                                    Y2_= Y1 + ix;
                                    r10 = X1[0];
                                    r11 = X1[1];
                                    r12 = X1[2];
                                    r13 = X1[3];
                                    y20 = 0;
                                    y21 = 0;
                                    y22 = 0;
                                    y23=0;
                                    
                                }
                                Y1_[0] += y10;
                                Y1_[1] += y11;
                                Y1_[2] += y12;
                                Y1_[3] += y13;
                                X2 = X  + iy;
                                Y1_= Y2 + iy;
                                r20 = X2[0];
                                r21 = X2[1];
                                r22 = X2[2];
                                r23 = X2[3];
                                y10 = 0;
                                y12=0;
                                y13=0;
                                y11 = 0;
                            }
                    }
                else
                    {
                        if(dist>=0)//Cas de base, on it�re sur x
                        {
                            
                            Y2_= Y2 + iy;
                            X1 = X  + iy;
                            X2 = X  + ix;
                            Y1_= Y1 + ix;
                            y10 =0;
                            y11 =0;
                            y12=0;
                            y13=0;
                            y20 = 0;
                            y21 = 0;
                            y22 = 0;
                            y23=0;
                            r10 = X1[0];
                            r11 = X1[1];
                            r12 = X1[2];
                            r13 = X1[3];
                            r20 = X2[0];
                            r21 = X2[1];
                            r22 = X2[2];
                            r23 = X2[3];
                        }
                        else
                        {
                            X2 = X  + iy;
                            Y1_= Y2 + iy;
                            X1 = X  + ix;
                            Y2_= Y1 + ix;
                            y10 = 0;
                            y12=0;
                            y13=0;
                            y11 = 0;
                            y20 = 0;
                            y21 = 0;
                            y22 = 0;
                            y23=0;
                            r10 = X1[0];
                            r11 = X1[1];
                            r12 = X1[2];
                            r13 = X1[3];
                            r20 = X2[0];
                            r21 = X2[1];
                            r22 = X2[2];
                            r23 = X2[3];
                        }
                    }
                if(ix-iy !=0)
                {
                    {
                        real & c = a0;
                        y10 += c * r10;
                        y20 += c * r20;
                    }
                    {
                        real & c = a1;
                        y11 += c * r10;
                        y20 += c * r21;
                    }
                    {
                        real & c = a4;
                        y10 += c * r11;
                        y21 += c * r20;
                    }
                    {
                        real & c = a2;
                        y12 += c * r10;
                        y20 += c * r22;
                    }
                    {
                        real& c = a8;
                        y10 += c * r12;
                        y22 += c* r20;
                    }
                    {
                        real & c = a3;
                        y13 += c * r10;
                        y20 += c * r23;
                    }
                    {
                        real & c = aC;
                        y10 += c *r13;
                        y23 += c * r20;
                    }
                    {
                        real & c = a5;
                        y11+= c*r11;
                        y21 += c *r21;
                    }
                    {
                        real & c = a6;
                        y12 += c*r11;
                        y21 += c*r22;
                    }
                    {
                        real & c = a9;
                        y11 += c * r12;
                        y22 += c * r21;
                    }
                    {
                        real & c = a7;
                        y13 += c * r11;
                        y21 += c * r23;
                    }
                    {
                        real & c  = aD;
                        y11 += c * r13;
                        y23 += c * r21;
                    }
                    {
                        real & c  = aA;
                        y12 += c * r12;
                        y22 += c * r22;
                    }
                    {
                        real & c = aE;
                        y12 += c * r13;
                        y23 += c * r22;
                    }
                    {
                        real & c = aB;
                        y13 += c * r12;
                        y22 += c *r23;
                    }
                    {
                        real & c = aF;
                        y13 += c * r13;
                        y23 += c *r23;
                    }
                    
                }
                else
                {
                    Matrix44* restored = new Matrix44(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,aA,aB,aC,aD,aE,aF);
                    Vector4 res1=restored->vecmul(X1);
                    for(int i=0;i<4;i++)
                    {
                        Y1_[i] += res1[i];
                        
                    }
                }
                if(m!= work_nb-1)
                {}
                else
                {
                    Y1_[0] += y10;
                    Y1_[1] += y11;
                    Y1_[2] += y12;
                    Y1_[3] += y13;
                    Y2_[0] += y20;
                    Y2_[1] += y21;
                    Y2_[2] += y22;
                    Y2_[3] += y23;
                }
            }
            
            barrier2.wait();
            k++;
        }
}
    void workThread2(boost::barrier& barrier2, int thID)
    {
        
        
        int k = 0;
        real* work;
        int work_nb;
        
        const real* X1;
        const real* X2;
        real* Y1_;
        real* Y2_;
        while(k<phase_number)
        {
            //std::cout<<"\nStart of phase"<<k<<"\n";
            
            _mutex.lock();
            work =  workingPhases[k][thID];
            work_nb = work_lengths[k][thID];
            int* index = workingIndexes[k][thID];
            _mutex.unlock();
            
            for(int m = 0; m<work_nb; m++)
            {
                int ix = index[m*2+0];
                int iy = index[m*2+1];
              
                real a0 = work[16*m+0];
                real a1 = work[16*m+1];
                real a2 = work[16*m+2];
                real a3 = work[16*m+3];
                real a4 = work[16*m+4];
                real a5 = work[16*m+5];
                real a6 = work[16*m+6];
                real a7 = work[16*m+7];
                real a8 = work[16*m+8];
                real a9 = work[16*m+9];
                real aA = work[16*m+10];
                real aB = work[16*m+11];
                real aC = work[16*m+12];
                real aD = work[16*m+13];
                real aE = work[16*m+14];
                real aF = work[16*m+15];
                
                if(ix-iy >=0)
                {
                    X1 = X  + iy;
                    X2 = X  + ix;
                    Y1_= Y1 + ix;
                    Y2_= Y2 + iy;
                    
                }
                else
                {
                    
                    X1 = X  + ix;
                    X2 = X  + iy;
                    Y2_= Y1 + ix;
                    Y1_= Y2 + iy;
                    
                    
                }
                
                if(ix-iy !=0)
                {
                    
                    real y10 = 0;
                    real y11 = 0;
                    real y12=0;
                    real y13=0;
                    real y20 = 0;
                    real y21 = 0;
                    real y22 = 0;
                    real y23=0;
                    real r10 = X1[0];
                    real r11 = X1[1];
                    real r12 = X1[2];
                    real r13 = X1[3];
                    real r20 = X2[0];
                    real r21 = X2[1];
                    real r22 = X2[2];
                    real r23 = X2[3];
                    {real & c = a0;
                        
                        y10 += c*r10;
                        y20 += c*r20;
                    }
                    {
                        real & c = a1;
                        y11 += c * r10;
                        y20 += c * r21;
                    }
                    {
                        real & c = a4;
                        y10 += c * r11;
                        y21 += c * r20;
                    }
                    {
                        real & c = a2;
                        y12 += c * r10;
                        y20 += c * r22;
                    }
                    {
                        real& c = a8;
                        y10 += c * r12;
                        y22 += c* r20;
                    }
                    {
                        real & c = a3;
                        y13 += c * r10;
                        y20 += c * r23;
                    }
                    {
                        real & c  = aC;
                        y10 += c *r13;
                        y23 += c * r20;
                    }
                    {
                        real & c = a5;
                        y11+= c*r11;
                        y21 += c *r21;
                    }
                    {
                        real & c=  a6;
                        y12 += c*r11;
                        y21 += c*r22;
                    }
                    {
                        real & c = a9;
                        y11 += c *r12;
                        y22 += c *r21;
                    }
                    {
                        real &c  = a7;
                        y13 += c *r11;
                        y21 += c * r23;
                    }
                    {
                        real & c  = aD;
                        y11 +=c*r13;
                        y23 += c *r21;
                    }
                    {
                        real & c  = aA;
                        y12 += c * r12;
                        y22 += c* r22;
                    }
                    {
                        real & c = aE;
                        y12 += c * r13;
                        y23 += c * r22;
                    }
                    {
                        real & c = aB;
                        y13 += c * r12;
                        y22 += c *r23;
                    }
                    {
                        real & c = aF;
                        y13 += c * r13;
                        y23 += c *r23;
                    }
                    
                    Y1_[0] += y10;
                    Y1_[1] += y11;
                    Y1_[2] += y12;
                    Y1_[3] += y13;
                    Y2_[0] += y20;
                    Y2_[1] += y21;
                    Y2_[2] += y22;
                    Y2_[3] += y23;
                    
                    
                   
                }
                else
                {
                    Matrix44* restored = new Matrix44(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,aA,aB,aC,aD,aE,aF);
                    Vector4 res1=restored->vecmul(X1);
                    for(int i=0;i<4;i++)
                    {
                        Y1_[i] += res1[i];
                        
                    }
                }
            }
            
            barrier2.wait();
            k++;
        }
}

    void vecMulAddnTimes(const real*X_calc, real*Y, int n_time)
    {

        X= X_calc;
        generateBlocks();
     
        try
        {
            boost::thread* threads = (boost::thread*) malloc(sizeof(boost::thread)*nbThreads);
            boost::barrier bar(nbThreads);
            boost::barrier bar2(nbThreads);
            for(int i=0;i<nbThreads; i++)
            {
                threads[i] = boost::thread(boost::bind(&MatSymBMtInstance::work, this,boost::ref(bar),boost::ref(bar2), n_time));;
            }
            
            
            
           
            
            for(int i=0;i<nbThreads; i++)
            {
                threads[i].join();
            }
            for(int i=0; i<nbThreads;i++)
            {
                threads[i].interrupt();
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << "Exception occurred: " << e.what() << std::endl;
            // Handle exception here if needed
            
        }
        for(int i=0;i<big_mat_size;i++)
        {
            Y[i] += Y1[i] + Y2[i];
        }
    }
    
    void vecMulAddnTimes2(const real*X_calc, real*Y, int n_time)
    {
        //Start thread,
        //Calculate, wait, calculate, wait..
        X= X_calc;
        generateBlocks2();
        //generateBlocks4thWay();
        try
        {
            boost::thread* threads = (boost::thread*) malloc(sizeof(boost::thread)*nbThreads);
            boost::barrier bar(nbThreads);
            boost::barrier bar2(nbThreads);
            for(int i=0;i<nbThreads; i++)
            {
                threads[i] = boost::thread(boost::bind(&MatSymBMtInstance::work2, this,boost::ref(bar),boost::ref(bar2), n_time));;
            }
            
            for(int i=0;i<nbThreads; i++)
            {
                threads[i].join();
            }
            for(int i=0; i<nbThreads;i++)
            {
                threads[i].interrupt();
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << "Exception occurred: " << e.what() << std::endl;
            // Handle exception here if needed
            
        }
        for(int i=0; i<big_mat_size; i++)
        {
                Y[i] += Y1[i] + Y2[i];
        }
    }
};

#endif /* MatSymBMtInstance_hpp */


