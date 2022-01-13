//----------------------------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cuda.h>
#include <vector>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//----------------------------------------------------------------------------
using namespace std;
//----------------------------------------------------------------------------
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }
//----------------------------------------------------------------------------


struct eDataList {
	bool* index;
	short depth;
};

struct eData {
	eDataList* data;
	bool* isPossible;
	short current;
};

struct NonoData {
	eData* row_data;
	eData* col_data;
};

struct InputHint {
	short num;
	short* hint;
};

struct NonoInput {
	short row_size;
	short col_size;
	InputHint* row_data;
	InputHint* col_data;
};

//Find gaps
__device__ void MakeCase(short index, short num, short sum, short* g, short* gg, short *depth)
{
	if (index >= num) return;
	else if (index == num-1 ) {
		index++;
		for (int i = 0; i < sum - num; i++)
		{
			MakeCase(index, num, sum, g, gg, depth);
			g[index]++;
		}
		short temp = 0;
		for (int j = 0; j < index; j++) temp += g[j];
		if (temp <= sum)
		{
			for (int j = 0; j < num; j++) gg[depth[0]*num+j]= g[j];
			depth[0]++;			
		}
		__syncthreads();
		temp = 0;
		g[index] = 1;
	}
	else {
		index++;
		for (int i = 0; i <= sum - num + 1; i++)
		{
			MakeCase(index, num, sum, g, gg, depth);
			g[index]++;
		}
		if (index != 0 && index!=num)g[index] = 1;
	}
}

__global__ void SolveNono(NonoInput* ni, NonoData* nd,short row_size, short col_size, short* row_hints, short* col_hints, short row_num, short col_num, short* pre_row, short* pre_col, bool* results)
{
	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	//Transform input data to NonoData
	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	short ty = threadIdx.y;
	short tx = threadIdx.x;
	
	extern __shared__ char* ds_Pout;

	//initialize element of ds_Pout as 2
	//0: blanked, 1: colored, 2: not determined
	ds_Pout = new char[row_size *(col_size)];
	ds_Pout[ty * (col_size) + tx] = 2;
	

	ni->row_size = row_size;
	ni->col_size = col_size;
	ni->row_data = new InputHint[row_size];
	ni->col_data = new InputHint[col_size];

	if(ty<(row_size)-1) ni->row_data[ty].num = pre_row[ty+1]-pre_row[ty];
	else ni->row_data[ty].num = row_num - pre_row[ty];

	ni->row_data[ty].hint = new short[ni->row_data[ty].num];
	__syncthreads();
	if(tx==0)	for(int i=0; i< ni->row_data[ty].num; i++) ni->row_data[ty].hint[i] = row_hints[pre_row[ty] + i];

	if (tx < (col_size)-1) ni->col_data[tx].num = pre_col[tx + 1] - pre_col[tx];
	else  ni->col_data[tx].num = col_num - pre_col[tx];

	ni->col_data[tx].hint = new short[ni->col_data[tx].num];
	__syncthreads();
	if (ty == 0) for (int i = 0; i < ni->col_data[tx].num; i++) ni->col_data[tx].hint[i] = col_hints[pre_col[tx] + i];
	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	//End of transform
	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	//Start to make all number of possible case
	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	//Initializing data structure for nonodata
	nd->row_data = new eData[row_size];
	nd->col_data = new eData[col_size];

	nd->row_data[ty].data = new eDataList[col_size];
	nd->row_data[ty].data[tx].depth = 0;

	nd->col_data[tx].data = new eDataList[row_size];
	nd->col_data[tx].data[tx].depth = 0;

	//starting to make row case
	short Hsum = 0; //sum of hint in row
	for (int i = 0; i < ni->row_data[ty].num; i++) Hsum += ni->row_data[ty].hint[i]; //calculate Hsum for row
	short* gap;
	gap = new short[(ni->row_data[ty].num)];
	
	short* depth;
	depth = new short[1];
	depth[0] = 0;
	
	gap[0] = 0;
	
	short* gg;
	gg = new short[10000];
	for (int i = 1; i < ni->row_data[ty].num; i++) gap[i] = 1;
	if (ni->row_data[ty].hint[0] == 0)//make every possible blank for row
	{
		depth[0] = 1;
		nd->row_data[ty].data[tx].depth = 1;
		nd->row_data[ty].data[tx].index = new bool[1];
		nd->row_data[ty].data[tx].index[0] = false;
	}
	else if (ni->row_data[ty].hint[0] == col_size)
	{
		depth[0] = 1;
		nd->row_data[ty].data[tx].depth = 1;
		nd->row_data[ty].data[tx].index = new bool[1];
		nd->row_data[ty].data[tx].index[0] = true;
	}
	else {
		for (int i = 0; i <= col_size - Hsum - ni->row_data[ty].num + 1; i++)
		{
			gap[0] = i;
			MakeCase(0, (ni->row_data[ty].num), col_size - Hsum, gap, gg, depth);
		}
	}
	__syncthreads();
	
	nd->row_data[ty].data[tx].depth = depth[0]; //updating depth information for each row
	nd->row_data[ty].data[tx].index = new bool[nd->row_data[ty].data[tx].depth]; //initializing index according to the depth
	nd->row_data[ty].current = depth[0];
	nd->row_data[ty].isPossible = new bool[nd->row_data[ty].data[tx].depth];

	for (int i = 0; i < nd->row_data[ty].data[tx].depth; i++) nd->row_data[ty].isPossible[i] = true;
	for (int i = 0; i < nd->row_data[ty].data[tx].depth; i++)
	{
		nd->row_data[ty].isPossible[i] = true;
		for (int j = 0; j < ni->row_data[ty].num; j++)
		{
			gap[j] = gg[i* ni->row_data[ty].num+j]; //make gap according to ith depth
		}
		
		__syncthreads();
		short temp1;
		short temp2 = 0;

		for (int k = 0; k < ni->row_data[ty].num; k++) //make row case
		{
			temp1 = temp2 + gap[k];
			temp2 = temp1 + ni->row_data[ty].hint[k];
			if (tx < temp1)
			{
				nd->row_data[ty].data[tx].index[i] = false;
				break;
			}
			else if (tx < temp2)
			{
				nd->row_data[ty].data[tx].index[i] = true;
				break;
			}
			else
			{
				if (k >= ((ni->row_data[ty].num) - 1)) {
					nd->row_data[ty].data[tx].index[i] = false;
					break;
				}
			}
		}
		__syncthreads();
	}
	__syncthreads();
	free(gg);
	free(gap);
	//short* ngap;
	gg = new short[10000];
	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@end of make row case, and start to make column case
	Hsum = 0;
	gap = new short[ni->col_data[tx].num];
	//__syncthreads();

	for (int i = 0; i < ni->col_data[tx].num; i++) Hsum += ni->col_data[tx].hint[i]; //calculate Hsum for column
	for (int i = 1; i < ni->col_data[tx].num; i++) gap[i] = 1;
	depth[0] = 0;
	gap[0] = 0;
	if (ni->col_data[tx].hint[0]==0)//make every possible blank for col
	{
		//printf("hint 0 case\n");
		depth[0] = 1;
		nd->col_data[tx].data[ty].depth = 1;
		nd->col_data[tx].data[ty].index = new bool[1];
		nd->col_data[tx].data[ty].index[0] = false;
	}
	else if (ni->col_data[tx].hint[0] == ni->row_size) 
	{
		//printf("hint full case\n");
		depth[0] = 1;
		nd->col_data[tx].data[ty].depth = 1;
		nd->col_data[tx].data[ty].index = new bool[1];
		nd->col_data[tx].data[ty].index[0] = true;
	}
	else 
	{
		for (int i = 0; i <= row_size - Hsum - ni->col_data[tx].num + 1; i++)
		{
			gap[0] = i;
			MakeCase(0, (ni->col_data[tx].num), row_size - Hsum, gap, gg, depth);
		}
	}
	__syncthreads();

	nd->col_data[tx].data[ty].depth = depth[0]; //updating depth information for each column
	nd->col_data[tx].data[ty].index = new bool[nd->col_data[tx].data[ty].depth]; //initializing index according to the depth
	nd->col_data[tx].current = depth[0];
	nd->col_data[tx].isPossible = new bool[depth[0]];
	for (int i = 0; i < nd->col_data[tx].data[ty].depth; i++) nd->col_data[tx].isPossible[i] = true;
	for (int i = 0; i < nd->col_data[tx].data[ty].depth; i++)
	{
		for (int j = 0; j < ni->col_data[tx].num; j++)
		{
			gap[j] = gg[i * ni->col_data[tx].num + j]; //make gap according to ith depth
		}

		__syncthreads();
		short temp1;
		short temp2 = 0;

		for (int k = 0; k < ni->col_data[tx].num; k++) //make row case
		{
			temp1 = temp2 + gap[k];
			temp2 = temp1 + ni->col_data[tx].hint[k];
			//	__syncthreads();
			if (ty < temp1)
			{
				nd->col_data[tx].data[ty].index[i] = false;
				break;
			}
			else if (ty < temp2)
			{
				nd->col_data[tx].data[ty].index[i] = true;
				break;
			}
			else
			{
				if (k >= ((ni->col_data[tx].num) - 1)) {
					nd->col_data[tx].data[ty].index[i] = false;
					break;
				}
			}
		}
		__syncthreads();
	}
	free(gg);
	__syncthreads();

	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	//End of making all number of possible case
	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

	ds_Pout[ty * ni->col_size + tx] = 2;
	bool isContinue = true;
	int count = 1;
	short row_index;
	short col_index;
	if (nd->row_data[ty].current == 1) ds_Pout[ty * ni->col_size + tx] = nd->row_data[ty].data[tx].index[0];
	if (nd->col_data[tx].current == 1) ds_Pout[ty * ni->col_size + tx] = nd->col_data[tx].data[ty].index[0];
	__syncthreads();
	
	while (isContinue && count <200) {

		//@@@@@@@@@@@@@@@@@@@@@@@@@@@
		//Find first possible index
		//@@@@@@@@@@@@@@@@@@@@@@@@@@@
		for (int i = 0; i < nd->row_data[ty].data[tx].depth; i++) {
			if (nd->row_data[ty].isPossible[i]) {
				row_index = i;
				break;
			}
		}
		for (int i = 0; i < nd->col_data[tx].data[ty].depth; i++) {
			if (nd->col_data[tx].isPossible[i]) {
				col_index = i;
				break;
			}
		}
		__syncthreads();

		//@@@@@@@@@@@@@@@@@@@@@@@@@@@
		//scan row
		//@@@@@@@@@@@@@@@@@@@@@@@@@@@
		if (nd->row_data[ty].current > 1 && ds_Pout[ty * ni->col_size + tx] == 2) {
			bool isIdentical = true;
			for (int i = 0; i < nd->row_data[ty].data[tx].depth; i++)
			{
				if ((nd->row_data[ty].data[tx].index[row_index] != nd->row_data[ty].data[tx].index[i])&&(nd->row_data[ty].isPossible[i]))
				{
					isIdentical = false;
					break;
				}
			}
			if (isIdentical) ds_Pout[ty * ni->col_size + tx] = nd->row_data[ty].data[tx].index[row_index];
		}
		__syncthreads();
		

		//@@@@@@@@@@@@@@@@@@@@@@@@@@@
		//scan column
		//@@@@@@@@@@@@@@@@@@@@@@@@@@@
		if (nd->col_data[tx].current>1 && ds_Pout[ty * ni->col_size + tx] == 2)
		{
			bool isIdentical = true;
			for (int i = 0; i < nd->col_data[tx].data[ty].depth; i++)
			{
				if ((nd->col_data[tx].data[ty].index[col_index] != nd->col_data[tx].data[ty].index[i])&&(nd->col_data[tx].isPossible[i]))
				{
					isIdentical = false;
					break;
				}
			}
			if (isIdentical) ds_Pout[ty * ni->col_size + tx] = nd->col_data[tx].data[ty].index[col_index];
		}
		
		__syncthreads();


		//@@@@@@@@@@@@@@@@@@@@@@@@@@@
		//reduce column
		//@@@@@@@@@@@@@@@@@@@@@@@@@@@
		if ((nd->col_data[tx].current != 1)&& (ds_Pout[ty * ni->col_size + tx] != 2))
		{
			for (int j = 0; j < nd->col_data[tx].data[ty].depth; j++)
			{
				if (ds_Pout[ty * ni->col_size + tx] != nd->col_data[tx].data[ty].index[j])
				{
					nd->col_data[tx].isPossible[j] = false;
				}
			}		
		}
		__syncthreads();
		int curcount = 0;
		for (int j = 0; j < nd->col_data[tx].data[ty].depth; j++)
		{
			if (nd->col_data[tx].isPossible[j]) curcount++;
		}
		__syncthreads();
		nd->col_data[tx].current = curcount;
		__syncthreads();

		//@@@@@@@@@@@@@@@@@@@@@@@@@@@
		//reduce row
		//@@@@@@@@@@@@@@@@@@@@@@@@@@@
		__syncthreads();

		if ((nd->row_data[ty].current != 1)&& (ds_Pout[ty * ni->col_size + tx] != 2))
		{
			for (int j = 0; j < nd->row_data[ty].data[tx].depth; j++)
			{
				if (ds_Pout[ty * ni->col_size + tx] != nd->row_data[ty].data[tx].index[j])
				{
					nd->row_data[ty].isPossible[j] = false;
				}
			}	
		}
		__syncthreads();

		curcount = 0;
		for (int j = 0; j < nd->row_data[ty].data[tx].depth; j++)
		{
			if (nd->row_data[ty].isPossible[j]) curcount++;
		}
		__syncthreads();
		nd->row_data[ty].current = curcount;
		__syncthreads();

		count++;
		if (nd->row_data[ty].current <= 1 && nd->col_data[tx].current <= 1) isContinue = false;

		//@@@@@@@@@@@@@@@@@@@@@@@@@@@
		//Check again
		//@@@@@@@@@@@@@@@@@@@@@@@@@@@
		if (nd->row_data[ty].current == 1)
		{
			for(int i=0; i<nd->row_data[ty].data[tx].depth;i++) 
				if (nd->row_data[ty].isPossible[i])
				{
					ds_Pout[ty * ni->col_size + tx] = nd->row_data[ty].data[tx].index[i];
					break;
				}
		}
		if (nd->col_data[tx].current == 1)
		{
			for(int i=0; i<nd->col_data[tx].data[ty].depth;i++)
				if (nd->col_data[tx].isPossible[i])
				{
					ds_Pout[ty * ni->col_size + tx] = nd->col_data[tx].data[ty].index[i];
					break;
				}
		}

		if (nd->row_data[ty].current == 0)
		{
			for (int i = 0; i < nd->row_data[ty].data[tx].depth; i++)nd->row_data[ty].isPossible[i] = true;
			nd->row_data[ty].current = nd->row_data[ty].data[tx].depth;
		}
		if (nd->col_data[tx].current == 0)
		{
			for (int i = 0; i < nd->col_data[tx].data[ty].depth; i++)nd->col_data[tx].isPossible[i] = true;
			nd->col_data[tx].current= nd->col_data[tx].data[ty].depth;
		}
	}
	

	if(ds_Pout[ty*col_size+tx]==1) results[ty * (col_size)+tx]=true;
	else results[ty * (col_size)+tx] = false;

}

//----------------------------------------------------------------------------
void writeOutput(string oName, bool* output, short col_width, short row_width)
{
	ofstream outputFile;
	outputFile.open(oName.c_str());
	if (outputFile.is_open())
	{
		for (size_t i = 0; i < row_width; i++)
		{
			for (size_t j = 0; j < col_width; j++)
			{
				if (output[i * col_width + j]) outputFile << "бс";
				else outputFile << "бр";
			}
			outputFile << endl;
		}
	}
}
int main(void) {
	cout << "check input file...";
	ifstream fin("./input3.txt");
	cout << endl;

	if (!fin.is_open()) cout << "File is not opend!!";
	else if(fin.is_open()) //start of data input
	{
		short row_size, col_size, * row_hints, * col_hints, row_num, col_num, * pre_row, * pre_col;
		short d_row_size, d_col_size, * d_row_hints, * d_col_hints, * d_row_num, * d_col_num, * d_pre_row, * d_pre_col;
		bool* results, * d_results;
		NonoData* nd;
		NonoInput* ni;
		short* ds_g;

		cout << "done. \n";
		cout << "open input file...";
		fin >> row_size;
		fin >> col_size;
		fin.ignore(1);

		pre_row = new short[row_size];
		pre_col = new short[col_size];
		row_num = 0;
		col_num = 0;
		short* temp_row_data;
		temp_row_data = new short[row_size * col_size / 2];
		short* temp_col_data;
		temp_col_data = new short[col_size * row_size / 2];

		for (int ri = 0; ri < row_size; ri++)
		{
			string line;
			getline(fin, line);

			stringstream ss(line); short temp;
			short num = 0;
			while (ss >> temp)
			{
				temp_row_data[row_num + num] = temp;
				num++;
			}
			pre_row[ri] = row_num;
			row_num += num;
		}
		row_hints = new short[row_num];
		for (int ri = 0; ri < row_num; ri++)	row_hints[ri] = temp_row_data[ri];


		for (int ci = 0; ci < col_size; ci++)
		{
			string line;
			getline(fin, line);

			stringstream ss(line); short temp;
			short num = 0;
			while (ss >> temp)
			{
				temp_col_data[col_num + num] = temp;
				num++;
			}
			pre_col[ci] = col_num;
			col_num += num;
		}
		col_hints = new short[col_num];
		for (int ci = 0; ci < col_num; ci++) col_hints[ci] = temp_col_data[ci];


		cout << "done. \n";
		cout << "Row size is : " << row_size << endl;
		cout << "Column size is : " << col_size << endl;
		//end of data input

		free(temp_row_data);
		free(temp_col_data);

		d_row_size = row_size;
		d_col_size = col_size;		




		results = (bool*)malloc(sizeof(bool) * row_size * col_size);


		cout << "memory allocation to device...";
		CUDA_CHECK_RETURN(cudaMalloc((void**)&ni, sizeof(NonoInput)));
		CUDA_CHECK_RETURN(cudaMalloc((void**)&nd, sizeof(NonoData)));
		CUDA_CHECK_RETURN(cudaMalloc((void**)&d_row_hints, sizeof(short) * row_num));
		CUDA_CHECK_RETURN(cudaMalloc((void**)&d_col_hints, sizeof(short) * col_num));
		CUDA_CHECK_RETURN(cudaMalloc((void**)&d_pre_row, sizeof(short) * row_size));
		CUDA_CHECK_RETURN(cudaMalloc((void**)&d_pre_col, sizeof(short) * col_size));
	    CUDA_CHECK_RETURN(cudaMalloc((void**)&d_results, sizeof(bool) * row_size * col_size));
		cout << "done. \n";

		cout << "memory copy from host to device...";
		CUDA_CHECK_RETURN(cudaMemcpy(d_row_hints, row_hints, sizeof(short) * row_num, cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN(cudaMemcpy(d_col_hints, col_hints, sizeof(short) * col_num, cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN(cudaMemcpy(d_pre_row, pre_row, sizeof(short) * row_size, cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN(cudaMemcpy(d_pre_col, pre_col, sizeof(short) * col_size, cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN(cudaMemcpy(d_results, results, sizeof(bool) * row_size * col_size, cudaMemcpyHostToDevice));

		cout << "done. \n";

		cout << "kernel launching...";
		//Time kernel launch
		cudaEvent_t start, stop;
		CUDA_CHECK_RETURN(cudaEventCreate(&start));
		CUDA_CHECK_RETURN(cudaEventCreate(&stop));
		float elapsedTime;

		CUDA_CHECK_RETURN(cudaEventRecord(start, 0));
		dim3 size = dim3(col_size, row_size, 1);
		//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
		// Calling the Kernel
		//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
		SolveNono << <1, size >> > (ni, nd, d_row_size, d_col_size, d_row_hints, d_col_hints, row_num, col_num, d_pre_row, d_pre_col, d_results);
		
		CUDA_CHECK_RETURN(cudaEventRecord(stop, 0));

		CUDA_CHECK_RETURN(cudaEventSynchronize(stop));
		CUDA_CHECK_RETURN(cudaEventElapsedTime(&elapsedTime, start, stop));
		CUDA_CHECK_RETURN(cudaThreadSynchronize());	// Wait for the GPU launched work to complete
		CUDA_CHECK_RETURN(cudaGetLastError());
		CUDA_CHECK_RETURN(cudaEventDestroy(start));
		CUDA_CHECK_RETURN(cudaEventDestroy(stop));
		cout << "done.\nElapsed kernel time: " << elapsedTime << " ms\n";
		cout << "Copying results back to host .... ";

		cout << "copying results from device to host...\n";
		//copy back output from GPU to host
		CUDA_CHECK_RETURN(cudaMemcpy(results, d_results, sizeof(bool) * row_size * col_size, cudaMemcpyDeviceToHost));
		cout << "done \n";

		cout << "printing output...\n";
		//print output
		for (int row = 0; row < row_size; row++)
		{
			for (int col = 0; col < col_size; col++)
			{
				if (results[row * col_size + col]) cout << "бс";
				else cout << "бр";
			}
			cout << endl;
		}
		cout << "done. \n";
		writeOutput("OuputNono.txt", results, col_size, row_size);
		cout << "deleting memory...";
		//free memories
		free(row_hints), free(col_hints), free(pre_row), free(pre_col) , free(results);
		cudaFree(ni), cudaFree(nd), cudaFree(d_row_hints), cudaFree(d_col_hints), cudaFree(d_pre_row), cudaFree(d_pre_col) , cudaFree(d_results);
		cout << "done. \n";
	}
	return 0;
}
//----------------------------------------------------------------------------
