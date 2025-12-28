#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <assert.h>
#include "cudaGF.h"
//#include <nvml.h>

extern "C"{
void cuda_Devinfo(void);
double cudaKernel_call(int, int, int, Vec3, int, Constant,int);
void cudaTransfer_toDev(int,int,int,Vec3 *,PLIST *,Edge_list *,int, Vec3 *);
void cuda_dealloc(void);
void cuda_HostUpdate(int,Output*);
}

__device__ Vec3 getTriangleCenter(Vec3,Vec3*,const Vec3*,const PLIST*);
__device__ double getGravitationAtPoint(Vec3,Vec3*,double*,const Vec3*,const PLIST*,const Edge_list*);

Vec3 *dev_vlist;
Vec3 *dev_npoints;//
PLIST *dev_plist;
Edge_list *dev_elist;
Output *dev_res;
extern int GPUDeviceID;

//nvmlDevice_t device_nvml;

__device__ __constant__ int vnum, pnum, edge_num, POLYGON_SCALE,nnum;
__device__ __constant__ Vec3 omega;
__device__ __constant__ Constant inp;
__device__ __constant__ double rad;


__host__ void cuda_Devinfo(void){
   cudaDeviceProp prop;
   int count,i;

	cudaSetDevice(GPUDeviceID);
	cudaGetDeviceCount(&count);
	for(i=0;i<count;i++){
       cudaGetDeviceProperties( &prop, i );
        fprintf( stderr,"   --- General Information for device %d ---\n", i );
        fprintf( stderr,"Name:  %s\n", prop.name );
        fprintf( stderr,"Compute capability:  %d.%d\n", prop.major, prop.minor );
        fprintf( stderr,"Clock rate:  %d\n", prop.clockRate );
        fprintf( stderr,"Device copy overlap:  " );
        if (prop.deviceOverlap) fprintf( stderr,"Enabled\n" );
        else fprintf( stderr,"Disabled\n");
        fprintf( stderr,"Kernel execution timeout :  " );
        if (prop.kernelExecTimeoutEnabled) fprintf( stderr,"Enabled\n" );
        else fprintf( stderr,"Disabled\n" );

        fprintf( stderr,"   --- Memory Information for device %d ---\n", i );
        fprintf( stderr,"Total global mem:  %ld\n", prop.totalGlobalMem );
        fprintf( stderr,"Total constant Mem:  %ld\n", prop.totalConstMem );
        fprintf( stderr,"Max mem pitch:  %ld\n", prop.memPitch );
        fprintf( stderr,"Texture Alignment:  %ld\n", prop.textureAlignment );

        fprintf( stderr,"   --- MP Information for device %d ---\n", i );
        fprintf( stderr,"Multiprocessor count:  %d\n", prop.multiProcessorCount );
        fprintf( stderr,"Shared mem per mp:  %ld\n", prop.sharedMemPerBlock );
        fprintf( stderr,"Registers per mp:  %d\n", prop.regsPerBlock );
        fprintf( stderr,"Threads in warp:  %d\n", prop.warpSize );
        fprintf( stderr,"Max threads per block:  %d\n", prop.maxThreadsPerBlock );
        fprintf( stderr,"Max thread dimensions:  (%d, %d, %d)\n",prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2] );
        fprintf( stderr,"Max grid dimensions:  (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2] );
        fprintf( stderr,"\n" );

	}


	//nvmlDeviceGetHandleByIndex(GPUDeviceID, &device_nvml);
}


__host__ void cudaTransfer_toDev(int vn, int pn, int e_num, Vec3 *vlist, PLIST *plist, Edge_list *elist,int nnum,Vec3 *n_points){
   //Output Zero ={0.0,0.0,0.0,0.0,0.0,0.0} 
   cudaError_t err = cudaSuccess; // Error code to check return values for CUDA calls
    fprintf(stderr, "cudaMemcpy Host -----> Device\n");
	
	cudaMalloc((Vec3**)&dev_vlist, sizeof(Vec3)*vn);
	cudaMalloc((Edge_list**)&dev_elist, sizeof(Edge_list)*e_num);
	cudaMalloc((PLIST**)&dev_plist, sizeof(PLIST)*pn);
	cudaMalloc((Output**)&dev_res, sizeof(Output)*nnum);
	cudaMalloc((Vec3**)&dev_npoints,sizeof(Vec3)*nnum);
	
    err = cudaGetLastError();
    if (err != cudaSuccess){
        fprintf(stderr, "[ERROR!] Failed in cudaMalloc  (error code %s)\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }		
	
	cudaMemcpy(dev_vlist,vlist,sizeof(Vec3)*vn,cudaMemcpyHostToDevice);
	cudaMemcpy(dev_elist,elist,sizeof(Edge_list)*e_num,cudaMemcpyHostToDevice);
	cudaMemcpy(dev_plist,plist,sizeof(PLIST)*pn,cudaMemcpyHostToDevice);

	cudaMemcpy(dev_npoints,n_points,sizeof(Vec3)*nnum,cudaMemcpyHostToDevice);

    err = cudaGetLastError();
    if (err != cudaSuccess){
        fprintf(stderr, "[ERROR!] Failed in cudaMemcpy Host -----> Device  (error code %s)\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
 	 else{
		 fprintf(stderr, "Done.\n\n");
	 }
}


__host__ void cuda_HostUpdate(int nnum, Output *res){

	cudaError_t err = cudaSuccess;
	fprintf(stderr, "\ncudaMemcpy Host <----- Device\n");
	
    cudaMemcpy(res,dev_res,sizeof(Output)*(nnum),cudaMemcpyDeviceToHost); //result for each polygon

	err = cudaGetLastError();
   	 if (err != cudaSuccess){
        fprintf(stderr, "[ERROR!] Failed in cudaMemcpy Host <----- Device  (error code %s)\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
   	 }
	 else{
		 fprintf(stderr, "Done.\n");
	 }
	 
}

__host__ void cuda_dealloc(void){
	cudaError_t err = cudaSuccess;

	cudaFree(dev_vlist);
	cudaFree(dev_elist);
	cudaFree(dev_plist);
	cudaFree(dev_res);
	
	cudaDeviceReset();
	err = cudaGetLastError();
   	 if (err != cudaSuccess){
        fprintf(stderr, "Failed to cuda device reset  (error code %s)\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
   	 }
}

//Calculation kernel
__global__ void CalcForAllPolygons(const Vec3* __restrict__ vlist, const PLIST* __restrict__ plist, const Edge_list* __restrict__ elist, Output *res,Vec3* __restrict__  n_points){

  int p_id = (blockIdx.y * blockDim.y + threadIdx.y)*(blockDim.x * gridDim.x)+(blockIdx.x * blockDim.x + threadIdx.x);
  //int p_id = blockIdx.x * blockDim.x + threadIdx.x;
  if(p_id >= nnum) return;

Vec3 v1,v2,v3;
Vec3 vertex1,vertex2,vertex3;
Vec3 center,attract,nvec;
Vec3 centRefugalAcc;

double lap;
double edge1,edge2,edge3,peri,area;
double g_potential=0.0, r_potential=0.0;
double slope;
double lat, lon, radius;
double scale_inv = 1.0/POLYGON_SCALE;
PLIST tmp_;
Output dev_result;
 
        //tmp_ = plist[p_id];
        tmp_ = plist[0];
        v1=vlist[tmp_.v[0]];
		v2=vlist[tmp_.v[1]];
		v3=vlist[tmp_.v[2]];

		//center  = Centroid(v1,v2,v3);
		center = n_points[p_id];
		center.x *= POLYGON_SCALE;
		center.y *= POLYGON_SCALE;
		center.z *= POLYGON_SCALE;
		vertex1 = Vscale(v1,scale_inv);
		vertex2 = Vscale(v2,scale_inv);
		vertex3 = Vscale(v3,scale_inv);

		//////////Heron's formula for area of a triangle
		edge1 = Vnorm(Vsub(vertex1,vertex2));
		edge2 = Vnorm(Vsub(vertex2,vertex3));
		edge3 = Vnorm(Vsub(vertex1,vertex3));
		peri  = (edge1 + edge2 + edge3)*0.5;
		area  = sqrt((peri*(peri-edge1)*(peri-edge2)*(peri-edge3)));
		
		g_potential = getGravitationAtPoint(center,&attract,&lap, vlist,plist, elist);
		getTriangleCenter(center, &nvec, vlist, plist);

		centRefugalAcc = Vcross(omega,Vcross(omega,center));
		centRefugalAcc = Vscale(centRefugalAcc,-1.0);
		reclat( center, &radius, &lon, &lat );
		if (lon < 0) lon = M_PI * 2.0 + lon;
		
		r_potential = 0.5 * Vnorm(omega) * Vnorm(omega) * (center.x*center.x+center.y*center.y);
		attract = Vadd(attract,centRefugalAcc);
		attract = Vscale(attract,-1.0);
		slope = Vsep(attract, nvec);
		
     	dev_result.cent     = Vscale(center,scale_inv);
		dev_result.lon      = lon*rad;
		dev_result.lat      = lat*rad;
		dev_result.centRAcc = centRefugalAcc;
		dev_result.att      = attract;
		dev_result.g_p	    = g_potential;
		dev_result.r_p	    = r_potential;
		dev_result.slope    = slope*rad;
		dev_result.nvec     = nvec;
		dev_result.area     = area;
		res[p_id] = dev_result;
}

__device__ double getGravitationAtPoint(Vec3 point,Vec3* attract,double* lap, const Vec3* __restrict__ vlist, const PLIST* __restrict__ plist, const Edge_list* __restrict__ elist){
	long int fp64bits;
	double potential=0.0, laplacian=0.0;
	double r_ir_j,e_ij;
	Vec3 v_i,v_j,v_k;
	double pvec[3];
	unsigned int a,b;
	double attraction[3]={0.0,0.0,0.0};
	double E_e[3][3];
	double L_e, res;
	int i;
	Edge_list Edge;
	
	Vec3 nA,nB,e_ab,nAab,nBba;
	unsigned int idx_A,idx_B;
	double numerator,denominator;
	Vec3 n_f;
	PLIST P_A, P_B;
	Vec3 attract_tmp;
	double Omega_f;
	// Acceralation for calculation of gravitation.
	double R=Vnorm(point);
	

	for(i=0; i < edge_num; i++){
		Edge = elist[i];
		a=Edge.v_a;
		b=Edge.v_b;
				
	    //calculate Edge dyad		
		idx_A=Edge.face_a;
		idx_B=Edge.face_b;
		
		P_A = plist[idx_A];
		P_B = plist[idx_B];

				nA = Vcross(Vsub(vlist[P_A.v[1]],vlist[P_A.v[0]]), Vsub(vlist[P_A.v[2]],vlist[P_A.v[0]]) );
				nA = Vscale(nA,1.0/Vnorm(nA)); //Normalize
				nB = Vcross(Vsub(vlist[P_B.v[1]],vlist[P_B.v[0]]), Vsub(vlist[P_B.v[2]],vlist[P_B.v[0]]) );
				nB = Vscale(nB,1.0/Vnorm(nB)); //Normalize
				
				//Calculation of non_edge_points (in Edge dyad process) are moved to host pre-process.
				e_ab=Vsub(vlist[a],vlist[b]);
				//the sign inversion for cross product (a value less than 0) is replaced to eliminate if-branches.
				//temporary reuse r_ir_j and res
				r_ir_j = Vdot(Vcross(Vsub(Edge.nonEdgeVerB,Edge.nonEdgeVerA),nA),e_ab);
				fp64bits = *((long int*)&r_ir_j);
				res = ((int)(fp64bits>>63)&1) * (-2.0)+1.0;
				e_ab=Vscale( Vscale(e_ab,res),1.0/Vnorm(e_ab));//Normalize

				nAab=Vcross(nA,e_ab);
				nBba=Vcross(Vscale(nB,-1.0),e_ab);
				
				E_e[0][0]=nA.x*nAab.x+nB.x*nBba.x;
				E_e[1][0]=nA.y*nAab.x+nB.y*nBba.x;
				E_e[2][0]=nA.z*nAab.x+nB.z*nBba.x;
				E_e[0][1]=nA.x*nAab.y+nB.x*nBba.y;
				E_e[1][1]=nA.y*nAab.y+nB.y*nBba.y;
				E_e[2][1]=nA.z*nAab.y+nB.z*nBba.y;
				E_e[0][2]=nA.x*nAab.z+nB.x*nBba.z;
				E_e[1][2]=nA.y*nAab.z+nB.y*nBba.z;
				E_e[2][2]=nA.z*nAab.z+nB.z*nBba.z;
					
		       //////////// calculate Edge dyad   ---- END
	
		r_ir_j=Vnorm(Vsub(vlist[a],point))+Vnorm(Vsub(vlist[b],point)); //r_i+r_j
		e_ij=Vnorm(Vsub(vlist[b],vlist[a]));
		L_e=log((r_ir_j+e_ij)/(r_ir_j-e_ij)); //L_e
		///////////////////////////////////////
		
		pvec[0]=vlist[a].x-point.x;
		pvec[1]=vlist[a].y-point.y;
		pvec[2]=vlist[a].z-point.z;

		

				potential += ( pvec[0]*pvec[0]*E_e[0][0] + pvec[0]*pvec[1]*E_e[0][1] 	+ pvec[0]*pvec[2]*E_e[0][2]
							+pvec[1]*pvec[0]*E_e[1][0] + pvec[1]*pvec[1]*E_e[1][1]  + pvec[1]*pvec[2]*E_e[1][2]
							+pvec[2]*pvec[0]*E_e[2][0] + pvec[2]*pvec[1]*E_e[2][1]  + pvec[2]*pvec[2]*E_e[2][2] ) *L_e;
						
                //Eq.13		
				attraction[0] += (-pvec[0]*E_e[0][0] -pvec[1]*E_e[0][1] -pvec[2]*E_e[0][2])*L_e;
				attraction[1] += (-pvec[0]*E_e[1][0] -pvec[1]*E_e[1][1] -pvec[2]*E_e[1][2])*L_e;
				attraction[2] += (-pvec[0]*E_e[2][0] -pvec[1]*E_e[2][1] -pvec[2]*E_e[2][2])*L_e;
	}
 


	for(i=0; i < pnum; i++){
		P_A = plist[i];
		
		v_i=Vsub(vlist[P_A.v[0]],point);
		v_j=Vsub(vlist[P_A.v[1]],point);
		v_k=Vsub(vlist[P_A.v[2]],point);
		
		n_f=Vcross(Vsub(vlist[P_A.v[1]],vlist[P_A.v[0]]),Vsub(vlist[P_A.v[2]],vlist[P_A.v[0]]));
		n_f=Vscale(n_f, 1.0/Vnorm(n_f)); //Normalize
		
		//calculate dyad
		E_e[0][0]=n_f.x*n_f.x;
		E_e[1][0]=n_f.y*n_f.x;
		E_e[2][0]=n_f.z*n_f.x;
		E_e[0][1]=n_f.x*n_f.y;
		E_e[1][1]=n_f.y*n_f.y;
		E_e[2][1]=n_f.z*n_f.y;
		E_e[0][2]=n_f.x*n_f.z;
		E_e[1][2]=n_f.y*n_f.z;
		E_e[2][2]=n_f.z*n_f.z;
		
		//calculate Omega_f
		numerator = Vdot(v_i,Vcross(v_j,v_k));
		denominator = Vnorm(v_i)*Vnorm(v_j)*Vnorm(v_k)
					  + Vnorm(v_i)*Vdot(v_j,v_k)
					  + Vnorm(v_j)*Vdot(v_k,v_i)
					  + Vnorm(v_k)*Vdot(v_i,v_j);
		Omega_f=2*atan2(numerator,denominator);
		
		pvec[0]=v_i.x;
		pvec[1]=v_i.y;
		pvec[2]=v_i.z;		
		
				potential += (-pvec[0]*pvec[0]*E_e[0][0]-pvec[0]*pvec[1]*E_e[0][1]-pvec[0]*pvec[2]*E_e[0][2]
								-pvec[1]*pvec[0]*E_e[1][0]-pvec[1]*pvec[1]*E_e[1][1]-pvec[1]*pvec[2]*E_e[1][2]
								-pvec[2]*pvec[0]*E_e[2][0]-pvec[2]*pvec[1]*E_e[2][1]-pvec[2]*pvec[2]*E_e[2][2]
								) *Omega_f;
								
								
				attraction[0] += (pvec[0]*E_e[0][0]+pvec[1]*E_e[0][1]+pvec[2]*E_e[0][2])*Omega_f;	
				attraction[1] += (pvec[0]*E_e[1][0]+pvec[1]*E_e[1][1]+pvec[2]*E_e[1][2])*Omega_f;
				attraction[2] += (pvec[0]*E_e[2][0]+pvec[1]*E_e[2][1]+pvec[2]*E_e[2][2])*Omega_f;				
				
				
		laplacian+=Omega_f; 
	}
	
	
	attract->x = inp.G_Sig*(attraction[0]);
	attract->y = inp.G_Sig*(attraction[1]);
	attract->z = inp.G_Sig*(attraction[2]);
	
	*lap = -inp.G_Sig*laplacian;
	res = 0.5*inp.G_Sig*potential;
	
	
	////Boundary check
	if(inp.simpleCalcBoundary>=0.0 && R>=inp.simpleCalcBoundary){
		// attract
		attract_tmp=Vsub(inp.centerOfGravity,point);
		attract_tmp=Vscale(attract_tmp,1.0/Vnorm(attract_tmp)); //Normalize
		attract_tmp=Vscale(attract_tmp,inp.volumeOfPolygon*inp.G_Sig/(R*R));
		*attract=attract_tmp;
		// --
		*lap=0.0;//lap
		res = inp.volumeOfPolygon*inp.G_Sig/R;
	}
	
	return res;
}

__device__ Vec3 getTriangleCenter(Vec3 pos, Vec3 *nvec, const Vec3* __restrict__ vlist, const PLIST* __restrict__ plist){
	Vec3 candidate,vec,uvec;
	pos=Vscale(pos,1.0/Vnorm(pos));
	double max=-1.0;
	int f_it, cand;
	double dot_up;
	Vec3 cand_v;
	
	PLIST P_A;
	
	for(f_it=0;f_it<pnum;f_it++){
		P_A = plist[f_it];
		
		vec = Vscale(Vadd(Vadd(vlist[P_A.v[0]],vlist[P_A.v[1]]),vlist[P_A.v[2]]),1.0/3.0);
		uvec = Vscale(vec,1.0/Vnorm(vec));
		dot_up = Vdot(uvec,pos);
		
		if(dot_up > max){
			candidate=vec;
			cand =f_it;
			max = dot_up;
		}
	}	
	
	P_A = plist[cand];	
	cand_v = Vcross(Vsub(vlist[P_A.v[1]], vlist[P_A.v[0]]) , Vsub(vlist[P_A.v[2]], vlist[P_A.v[0]]));
	nvec->x = cand_v.x;
	nvec->y = cand_v.y;
	nvec->z = cand_v.z;
	
	return candidate;
}
	
__host__ double cudaKernel_call(int vn, int pn, int en, Vec3 omg, int POLY_SCALE, Constant inp_cons, int nn){
cudaEvent_t st, et;
cudaError_t err = cudaSuccess;

int n_threads;
//n_threads = pn + (32-pn%32)%32; // the number of threads working on GPU should be multiple of 32
n_threads = nn + (32-nn%32)%32;
 dim3 grid(n_threads/32,1,1), block(32,1,1);
float etime;
double rd = 180.0/M_PI;

 //Constant memory
   cudaMemcpyToSymbol(vnum,&vn,sizeof(int));
   cudaMemcpyToSymbol(pnum,&pn,sizeof(int));
   cudaMemcpyToSymbol(edge_num,&en,sizeof(int));
   cudaMemcpyToSymbol(omega,&omg,sizeof(Vec3));
   cudaMemcpyToSymbol(POLYGON_SCALE,&POLY_SCALE,sizeof(int));
   cudaMemcpyToSymbol(inp,&inp_cons,sizeof(Constant));
   cudaMemcpyToSymbol(rad,&rd,sizeof(double));
   cudaMemcpyToSymbol(nnum,&nn,sizeof(int));//
   
   err = cudaGetLastError();
   	 if (err != cudaSuccess){
        fprintf(stderr, "[ERROR!] Failed in Assignment of Constant Memory (%s)\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
   	 }
	 
	 
   fprintf(stderr," CUDA kernel launch\n");
   cudaEventCreate(&st);
   cudaEventCreate(&et);
   cudaEventRecord(st);
   CalcForAllPolygons<<<grid, block>>>(dev_vlist, dev_plist, dev_elist, dev_res, dev_npoints); //execute on GPU
   
	err = cudaGetLastError();
   	 if (err != cudaSuccess){
        fprintf(stderr, "[ERROR!] Failed in Kernel call (%s)\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
   	 }
	 cudaDeviceSynchronize();
	 cudaEventRecord(et);
	 cudaEventSynchronize(et);
	 
	 fprintf(stderr," Kernel execution (of %d threads) is Synchronized.\n",n_threads);
	 err = cudaGetLastError();
   	 if (err != cudaSuccess){
        fprintf(stderr, "[ERROR!] Errors in kernel execution occured. (%s)\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
   	 }
	 
	cudaEventElapsedTime(&etime, st, et);
	cudaEventDestroy(st);
	cudaEventDestroy(et);

	return etime;
}
