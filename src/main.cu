#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <math.h>

#define THREADS_PER_BLOCK 256

typedef unsigned int u32;
typedef unsigned long long u64;

__host__ __device__ __inline__
float sdotfun(float s0, float g, float H, float I, float w)
{
    float a=270, b=108, d=0.154, tau=0.1, gamma=0.641, J=0.2609;
    float x = w*J*s0 + J*g*H + I;
    float R = (a*x-b)/(1-exp(-d*(a*x-b)));
    return -(s0/tau)+(1-s0)*R*gamma;
}

__host__ __device__ __inline__
float clamp(float x, float low, float up)
{
	if(x < low) return low;
	if(x > up) return up;
	return x;
}

__global__
void sim(u32 n_sims, u32 n_steps, u32 n_nodes, float * W, float * X)
{
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	if(gid >= n_sims) return;
    for (int t=0; t<n_steps; t++)
    {
        for (int i=0; i<n_nodes; i++)
        {
            float x = X[i*n_sims + gid], H=0.0;
            for (int j=0; j<n_nodes; j++)
                H += W[i*n_nodes+j] * X[j*n_sims + gid];
            X[i*n_sims + gid] = clamp(x + 0.1 * sdotfun(x, 0.053, H, 0.3, 1.0)*(1e-3), 0.0, 1.0);

        }
    }

}

u64 get_time()
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	u64 ret = tv.tv_usec/1000;

	/* Adds the seconds after converting them to millisecond (10^-3) */
	ret += (tv.tv_sec * 1000);
	return ret;
}


void rand_fill(float * x, u32 size){
	for(int i = 0; i < size; ++i){
		x[i] = (float)rand()/(float)RAND_MAX;
	}
}

u64 test_1(u32 n_steps, u32 n_skip, u32 n_sims, u32 n_nodes, float * h_weights, u32 n_weights, float * h_x, u32 n_x, float * times, float * dats){

	u32 n_block = n_sims / THREADS_PER_BLOCK + 1;

	u64 start_time = get_time();

	float * d_weights, * d_x;
	cudaMalloc((void**) &d_weights, sizeof(float) * n_weights);
	cudaMalloc((void**) &d_x, sizeof(float) * n_x);

	cudaMemcpy(d_weights, h_weights, sizeof(float) * n_weights, cudaMemcpyHostToDevice);
	cudaMemcpy(d_x, h_x, sizeof(float) * n_x, cudaMemcpyHostToDevice);

	for(u32 i = 0; i < n_steps; ++i)
	{
		sim<<<n_block, THREADS_PER_BLOCK>>>(n_sims, 1, n_nodes, d_weights, d_x);
		if(cudaDeviceSynchronize() != cudaSuccess){
			printf("test_1 runs into error at step %d!\n", i);
			exit(1);
		}
		if(i % n_skip == 0)
		{
			printf("Processing step %u / %u.\n", i, n_steps);
			times[i/n_skip] = i * 0.1;
			cudaMemcpy(dats + (i/n_skip) * n_x, d_x, sizeof(float) * n_x, cudaMemcpyDeviceToHost);
		}
	}

	cudaFree(d_weights);
	cudaFree(d_x);

	u64 end_time = get_time();
	return end_time - start_time;
}

u64 test_cpu(u32 n_steps, u32 n_skip, u32 n_sims, u32 n_nodes, float * h_weights, u32 n_weights, float * h_x, u32 n_x, float * times, float * dats)
{
	u64 start_time = get_time();

	for(u32 i = 0; i < n_steps; ++i)
	{
		for(int i_sim = 0; i_sim < n_sims; ++i_sim)
		{
			for (int i_node=0; i_node<n_nodes; i_node++)
			{
				float x = h_x[i_node*n_sims + i_sim], H=0.0;
				for (int j=0; j<n_nodes; j++)
					H += h_weights[i_node*n_nodes+j] * h_x[j*n_sims + i_sim];
				h_x[i_node*n_sims + i_sim] = clamp(x + 0.1 * sdotfun(x, 0.053, H, 0.3, 1.0)*(1e-3), 0.0, 1.0);
			}
		}
		printf("Processing step %u / %u.\n", i, n_steps);
		if(i % n_skip == 0)
		{
			times[i/n_skip] = i * 0.1;
			memcpy(dats + (i/n_skip) * n_x, h_x, sizeof(float) * n_x);
		}
	}
	u64 end_time = get_time();

	return end_time - start_time;
}

__inline__
bool float_equal(float a, float b)
{
	float precision = 0.00001;
	return (a - precision) < b && (a + precision) > b;
}

bool verify(float * x, float * y, u32 size)
{
	for(int i = 0; i < size; ++i)
	{
		if(!float_equal(x[i], y[i]))
			return false;
	}
	return true;
}

int main(int argc, char * argv[]){

	srand((u32)time(NULL));

	u32 n_steps = 1000; //10*60*1000*10 in opencl version
	u32 n_steps_cpu = 1000;
	u32 n_nodes = 426, n_sims = 512;
	u32 n_skip = 100;

	u32 n_weights = n_nodes * n_nodes;
	u32 n_x = n_nodes * n_sims;
	u32 n_time = n_steps/n_skip + 1;
	u32 n_dat = (n_steps/n_skip + 1) * n_x;

	float * h_weights = (float *) malloc(sizeof(float) * n_weights);
	float * h_x = (float *) malloc(sizeof(float) * n_x);

	rand_fill(h_weights, n_weights);
	rand_fill(h_x, n_x);

	float * times_cpu = (float *) malloc(sizeof(float) * n_time);
	float * dats_cpu = (float *) malloc(sizeof(float) * n_dat);
	float * h_x_cpu = (float *) malloc(sizeof(float) * n_x);
	memcpy(h_x_cpu, h_x, sizeof(float) * n_x);
//u64 test_cpu_run_time = test_cpu(n_steps_cpu, n_skip, n_sims, n_nodes, h_weights, n_weights, h_x_cpu, n_x, times_cpu,  dats_cpu);
	//printf("Test cpu takes %lu ms to complete %u step(s).\n", test_cpu_run_time, n_steps_cpu);

	float * times_1 = (float *) malloc(sizeof(float) * n_time);
	float * dats_1 = (float *) malloc(sizeof(float) * n_dat);
	float * h_x_1 = (float *) malloc(sizeof(float) * n_x);
	memcpy(h_x_1, h_x, sizeof(float) * n_x);
	u64 test_1_run_time = test_1(n_steps, n_skip, n_sims, n_nodes, h_weights, n_weights, h_x_1, n_x, times_1,  dats_1);
	printf("Test 1 takes %lu ms to complete %u step(s).\n", test_1_run_time, n_steps);

	printf("Verifying results....");
	bool result = verify(dats_cpu, dats_1, (min(n_steps, n_steps_cpu)/n_skip + 1) * n_x);
	printf("%s!\n", result?"success":"failure");


	free(h_weights);
	free(h_x);

	free(times_cpu);
	free(times_1);
	free(dats_cpu);
	free(dats_1);

	free(h_x_1);
	free(h_x_cpu);

	return 0;
}

/*
 * CPU - 100 steps - 16124 ms
 * test_1 - 100 steps - 1731 ms
 *
 * CPU - 1000 steps - 164836 ms
 * test_1 - 1000 steps - 15581 ms
 */
