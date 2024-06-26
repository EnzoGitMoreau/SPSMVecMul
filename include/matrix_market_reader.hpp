#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include "matrix44.h"
#include "sparmatsymblk.h"
#define RSB
#ifdef RSB
#include <rsb.hpp>
#endif
class MatrixReader
{
private:
	std::vector<Matrix44*> blocks;
	short** isCreatedBlock;
	short** indexInVector;
	int matrixSize;
	int numberofEntries;
	int number_of_block_per_line;
public:
	std::vector<std::string> splitStringByWhitespace(const std::string& str) {
	std::vector<std::string> result;
	std::istringstream iss(str);
	std::string token;
	
	

	while (iss >> token) 
	{
			result.push_back(token);
	}

		return result;
	}
	
	
	MatrixReader(std::string file_name, SparMatSymBlk* matrix, int nb_threads)
	{
		std::ifstream inputFile(file_name);
		if(!inputFile)
		{
			std::cout<<"\nCouldn't read "<<file_name<<"\n";	
		}
		else
		{
			
			std::string ligne;
			int i =0;
			std::istringstream iss;
			
			std::vector<std::string> splitLine;
			
			while (getline(inputFile, ligne)) 
			{ 
				splitLine = splitStringByWhitespace(ligne);
				if(i==1)
				{
					matrixSize = std::stoi(splitLine[0]);
					numberofEntries = std::stoi(splitLine[2]);
					std::cout<<"\n[INFO] Matrix Market's Matrix informations: \n";
					std::cout << "[INFO] Matrix size: "<<matrixSize <<" Number of entries: "<<numberofEntries;
					std::cout <<"\n[INFO] Reading and constructing Matrix";
					if(matrixSize%(4*nb_threads) !=0)
					{
					matrix->resize((int(matrixSize/(4*nb_threads))+1)*(4*nb_threads));

					}
					else
					{
						matrix->resize(matrixSize);
					}
				}
				if(i>1)
				{
					std::istringstream iss(splitLine[2]);
					int i = std::stoi(splitLine[0]);
					int j = std::stoi(splitLine[1]);
					real value;
					iss>>value;
					real& value_in_matrix = matrix->element(i,j);
					value_in_matrix = value;
				
					
				}
				i++;
			}
		}
		

	}
	#ifdef RSB
	MatrixReader(std::string file_name, SparMatSymBlk* matrix,rsb::RsbMatrix<double>* mtx, int nb_threads)
	{
		std::ifstream inputFile(file_name);
		if(!inputFile)
		{
			std::cout<<"\nCouldn't read "<<file_name<<"\n";	
		}
		else
		{
			
			std::string ligne;
			int i =0;
			std::istringstream iss;
			
			std::vector<std::string> splitLine;
			
			while (getline(inputFile, ligne)) 
			{ 
				splitLine = splitStringByWhitespace(ligne);
				if(i==1)
				{
					matrixSize = std::stoi(splitLine[0]);
					numberofEntries = std::stoi(splitLine[2]);
					std::cout<<"\n[INFO] Matrix Market's Matrix informations: \n";
					std::cout << "[INFO] Matrix size: "<<matrixSize <<" Number of entries: "<<numberofEntries;
					std::cout <<"\n[INFO] Reading and constructing Matrix";
					if(matrixSize%(4*nb_threads) !=0)
					{
					matrix->resize((int(matrixSize/(4*nb_threads))+1)*(4*nb_threads));

					}
					else
					{
						matrix->resize(matrixSize);
					}
					
					//const rsb_coo_idx_t nrA { matrixSize }, ncA { matrixSize };
					//mtx = new rsb::RsbMatrix<double>(nrA, ncA);
				}
				if(i>1)
				{
					std::istringstream iss(splitLine[2]);
					int i = std::stoi(splitLine[0]);
					int j = std::stoi(splitLine[1]);
					real value;
					iss>>value;
					real& value_in_matrix = matrix->element(i,j);
					value_in_matrix = value;
					//mtx->set_val(value, i,j);
				
					
				}
				i++;
			}
			std::cout<<"[INFO] Done reading input Matrix \n";
		}
		

	}
	#endif
	
	int problemSize()
	{
		return matrixSize;
	}
	
	
};