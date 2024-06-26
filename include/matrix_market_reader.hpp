#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <fstream>

class MatrixReader
{
public:
	MatrixReader(std::string file_name)
	{
		std::ifstream inputFile(file_name);
		if(!inputFile)
		{
			std::cout<<"\nCouldn't read "<<file_name<<"\n";	
		}
		else
		{
			std::string ligne;
			while (getline(inputFile, ligne)) 
			{ 
				std::cout << ligne << std::endl;
			}
		}

	}
};