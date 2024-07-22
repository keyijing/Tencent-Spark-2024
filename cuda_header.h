# include <cuda_runtime.h>
# include <chrono>
template<typename F>
float PERF_GPU(F &&func)
{
	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	func();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime,start,stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	return elapsedTime/1000;
}
template<typename F>
float PERF_CPU(F &&func)
{
	using namespace std::chrono;
	auto start=high_resolution_clock::now();
	func();
	auto stop=high_resolution_clock::now();
	return duration<float>(stop-start).count();
}