#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL\cl.h>
#include <fstream>
#include <iostream>
#include <cstdio>
#include <stdio.h>
#include <string>
#include <ctime>
#include <Windows.h>
#pragma comment(lib, "OpenCL.lib")
using namespace std;

#define IND(x,y) (y * N + x)


const int N = 512;
const float Step = 1.0f / (N - 1);
const int M = N * N;
const double eps = 0.005;

int GetSizeFile(const char *);//возвращает кол-во char в кернел файле, т.е. кол-во символов
char *GetKernelSource(const char * , size_t);
void Init_data(float *data, float *exact, float *F)
{
	float x = 0.0f;
	float y = 0.0f;

	for (int i = 0; i < M; i++)
	{
		data[i] = 0.0f;
	}

	for (int i = 0; i < N; i++, x += Step)
	{
		data[i] = data[N * (N - 1) + i] = x * x - x + 1.0f;
	}

	for (int j = 0; j < N; j++, y += Step)
	{
		data[N * j] = data[N - 1 + N * j] = y * y - y + 1.0f;
	}

	x = 0.0;
	y = 0.0;
	for (int i = 0; i < N; i++)
	{
		x = i * Step;
		for (int j = 0; j < N; j++)
		{
			y = j * Step;
			exact[IND(i, j)] = (x * x - x + 1.0f) * (y * y - y + 1.0f);
			F[IND(i, j)] = 4.0f + 2.0f * x * x - 2.0f * x + 2.0f * y * y - 2.0f * y;
		}
	}




}


int main()
{
	cl_int ret=0;
	srand(time(NULL));
	int boolean=0;
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	const char FileName[] = "test.cl";
	const char KernelName[]= "test2";
	cl_platform_id PlatformID;
	cl_device_id DeviceID;
	cl_context Context;
	cl_command_queue queue;
	cl_program Program;
	cl_kernel Kernel,Kernel2,Kernel3,Kernel4;

	size_t FileSize=GetSizeFile(FileName);
	char *KernelSource=GetKernelSource(FileName,FileSize);
	clGetPlatformIDs(1,&PlatformID,NULL);

	clGetDeviceIDs(PlatformID,CL_DEVICE_TYPE_GPU,1,&DeviceID,NULL);

	Context = clCreateContext(NULL,1,&DeviceID,NULL,NULL,&ret);
	//cout<<ret<<endl;
	queue = clCreateCommandQueue(Context,DeviceID,0,NULL);
	Program = clCreateProgramWithSource(Context,1,(const char **)&KernelSource,(const size_t *)&FileSize,&ret);
	//cout<<ret<<endl;
	clBuildProgram(Program,1,&DeviceID,NULL,NULL,&ret);
	//cout<<ret<<endl;

	cl_kernel kernel = clCreateKernel(Program, "test" ,&ret);
	if(ret)
	{
		cout<<endl;
		cout<<ret<<endl<<"error kernel"<<endl;
		system("pause");
	}

	
	float *data=new float[M];
	float  *exact=new float[M];
	float *F=new float[M];
	float *result = new float[M];
	float error = 1;

	Init_data(data,exact,F);

	float h = 1.0f / N;
	cl_mem Bdata = clCreateBuffer(Context,CL_MEM_READ_WRITE, sizeof(float)*M , NULL , NULL );
	cl_mem BF = clCreateBuffer(Context, CL_MEM_READ_ONLY, sizeof(float)*M, NULL, NULL);
	cl_mem Bresult = clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(float)*M, NULL, NULL);
	cl_mem H = clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(float)*M, NULL, NULL);
	clEnqueueWriteBuffer(queue,Bdata,CL_TRUE,0, sizeof(float)*M , data , 0 , NULL , NULL );
	clEnqueueWriteBuffer(queue, BF, CL_TRUE, 0, sizeof(float)*M, F, 0, NULL, NULL);


	
	//clSetKernelArg(kernel, 3, sizeof(cl_mem), &h);
	size_t group;
	unsigned int count = M;

	int iteration = 0;
	clock_t t0 = clock();
	clSetKernelArg(kernel , 0 , sizeof (cl_mem ), &Bdata);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &Bresult);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &BF);
	system("pause");

	while (error > eps)
	{
		

		
		clGetKernelWorkGroupInfo(kernel, DeviceID, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &group, NULL);
		clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &count, &group, 0, NULL, NULL);
		clFinish(queue);
		if(iteration%2==0)
		{
			clSetKernelArg(kernel , 0 , sizeof (cl_mem ), &Bdata);
			clSetKernelArg(kernel, 1, sizeof(cl_mem), &Bresult);
		}
		else
		{
			clSetKernelArg(kernel , 1 , sizeof (cl_mem ), &Bdata);
			clSetKernelArg(kernel, 0, sizeof(cl_mem), &Bresult);
		}


		clEnqueueBarrier(queue);
		if (iteration % 1000 == 0){
			if(iteration%2==0)
			{
			clEnqueueReadBuffer(queue,Bresult,CL_TRUE,0, sizeof(float)*M , data , 0 , NULL , NULL );
			}
			clFinish(queue);
			error = 0;
			cout<<data[0]<<endl;
			for (int i = 0; i < M; i++){
				if (abs(data[i] - exact[i]) > error)
				{error = abs(data[i] - exact[i]);}

			}
			
			cout << "Error on " << iteration << " iterationation = " << error << endl;
		}
		iteration++;
		//cl_mem t = NULL;
		//t = Bdata;
		//Bdata = Bresult;
		//Bresult= t;
	}

	cout << "Done with " << iteration << " iterations for " << (double)(clock() - t0) / CLOCKS_PER_SEC << " seconds" << endl;

	system("pause");


	
	
	clReleaseMemObject(Bdata);
	clReleaseMemObject(Bresult);
	clReleaseProgram(Program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(queue);
	clReleaseContext(Context);
	return 0;
}
int GetSizeFile(const char *FileName)
{
	int count=0;
	ifstream file(FileName);	
	string a;
	while(!file.eof())
	{
		getline(file,a);
		count+=a.length()+1;
	}
	file.close();
	count--;
	return count;

}
char *GetKernelSource(const char *FileName, size_t FileSize)
{
	FILE *FileKernel=fopen(FileName,"r");;
	char *source_str=new char [FileSize];
	fread(source_str,sizeof(char),FileSize,FileKernel);
	fclose(FileKernel);
	return source_str;
}
