__kernel void test(__global float *data,__global float *result,__global float *F)
{
	int Idx=get_global_id(0);
	int N=512;
	float h=1.0/N;
	/*if ((Idx <= (N-1)) || ((Idx+1) % N ==0) || (Idx % N==0) || (Idx >= N*(N-1)))
	{
		result[Idx] = data[Idx];
	}
	else
	{
		result[Idx] = 0.25*(data[Idx-1] + data[Idx+1] + data[Idx - N] + data[Idx+N]-F[Idx]*h*h);
	}*/
	
}