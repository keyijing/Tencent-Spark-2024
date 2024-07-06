# include <cuda_runtime.h>
# include <iostream>
# include <vector>
using namespace std;
constexpr int BLOCK_SIZE=16;
__global__ void kernel_init(int *a,size_t *b,size_t n)
{
	size_t i=blockIdx.x*blockDim.x+threadIdx.x;
	if(i<n) b[i]=(a[i]>0);
}
__global__ void kernel_acc(size_t *a,size_t n,size_t B)
{
	size_t i=blockIdx.x*blockDim.x+threadIdx.x,l=i*B,r=(i+1)*B;
	if(r>n) r=n;
	while(++l<r) a[l]+=a[l-1];
}
__global__ void kernel_acc_block(size_t *a,size_t *b,size_t B,size_t L)
{
	b[0]=0;
	for(size_t i=1,j=B-1;i<L;i++,j+=B) b[i]=b[i-1]+a[j];
}
__global__ void kernel_add_block(size_t *a,size_t *b,size_t n,size_t B)
{
	size_t i=blockIdx.x*blockDim.x+threadIdx.x,j=blockIdx.y*blockDim.y+threadIdx.y;
	if(j<B && i*B+j<n) a[i*B+j]+=b[i];
}
__global__ void kernel_debubble(int *a,size_t *b,int *c,size_t n)
{
	size_t i=blockIdx.x*blockDim.x+threadIdx.x;
	if(i<n && a[i]>0) c[b[i]-1]=a[i];
}
size_t debubble(vector<int> &data)
{
	size_t n=data.size(),B=sqrt(n)+0.5,L=(n-1)/B+1,m;
	int *d_a,*d_d;
	size_t *d_b,*d_c;
	cudaMalloc(&d_a,n*sizeof(int));
	cudaMalloc(&d_b,n*sizeof(size_t));
	cudaMalloc(&d_c,L*sizeof(size_t));
	cudaMalloc(&d_d,n*sizeof(int));
	cudaMemcpy(d_a,data.data(),n*sizeof(int),cudaMemcpyHostToDevice);
	kernel_init<<<(n-1)/(BLOCK_SIZE*BLOCK_SIZE)+1,BLOCK_SIZE*BLOCK_SIZE>>>(d_a,d_b,n);
	kernel_acc<<<(L-1)/(BLOCK_SIZE*BLOCK_SIZE)+1,BLOCK_SIZE*BLOCK_SIZE>>>(d_b,n,B);
	kernel_acc_block<<<1,1>>>(d_b,d_c,B,L);
	kernel_add_block<<<dim3((L-1)/BLOCK_SIZE+1,(B-1)/BLOCK_SIZE+1),dim3(BLOCK_SIZE,BLOCK_SIZE)>>>(d_b,d_c,n,B);
	cudaMemcpy(&m,d_b+(n-1),sizeof(size_t),cudaMemcpyDeviceToHost);
	kernel_debubble<<<(n-1)/(BLOCK_SIZE*BLOCK_SIZE)+1,BLOCK_SIZE*BLOCK_SIZE>>>(d_a,d_b,d_d,n);
	cudaMemset(d_d+m,0,(n-m)*sizeof(int));
	cudaMemcpy(data.data(),d_d,n*sizeof(int),cudaMemcpyDeviceToHost);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	cudaFree(d_d);
	return m;
}
int main()
{
	vector<int> vec={0,1,0,2,0,3};
	cout<<debubble(vec)<<"\n";
	for(int i:vec) cout<<i<<" ";
	cout<<"\n";
	return 0;
}