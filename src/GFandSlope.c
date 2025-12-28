/*
 *  GFandSlope_div_area_cuda.c
 *  by f.kono@u-aizu.ac.jp
 *  
 *  v20211117: CUDA ver. (single/double precision)
  *     v20230428: Modified thread assignment to kernels
 *
 *  Original C++ code is created by m5151126 on Nov. 11, 2011.
 *  Copyright 2011 University of Aizu. All rights reserved.
 *
 */
 
#define POLYGON_SCALE (1000) // Scale from km to SI
#define GRAVITY_CONST (6.67259e-11) // Constant in SI

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
//#include <nvml.h>

#ifdef SP
#include "./sp/GFhost.h"
#else
#include "GFhost.h"
#endif

extern void cuda_Devinfo(void);
extern double cudaKernel_call(int, int, int, Vec3, int, Constant, int);
extern double cudaKernel_call_np(int, int, int, Vec3, int, Constant);
extern void cudaTransfer_toDev(int, int, int, Vec3 *, PLIST *, Edge_list *, int, Vec3 *);
extern void cudaTransfer_toDev_np(int, int, int, Vec3 *, PLIST *, Edge_list *);
extern void cuda_dealloc(void);
extern void cuda_HostUpdate(int,Output*);

Vec3 Centroid(Vec3, Vec3, Vec3);
void OutputResults(char *, int, Output*);
extern void GravitationalField(double,double,double);
extern void GetPolygonCenter(Vec3 *);

extern int LoadPolygonData(char*, int*, int*);
extern int LoadPointsData(char*, int*);//
extern int ProcessInitial(char*);

Vec3 *vlist; //list of verteces of the shape model
PLIST *plist; //list of plates/faces of the shape model

int nnum;//
Vec3 *n_points; //list of points at which the gravity is to be computed


extern Edge_list *elist;
int vnum; //number of verticies
int pnum; //number of faces on the shape model
extern int edge_num; //number of edges

Output *res; //result array (pnum) --> nn
Constant inp;

extern double simpleCalcBoundary;
extern double G_Sig;
extern double volumeOfPolygon;
extern Vec3 centerOfGravity;

int START_POLYGON = 1, END_POLYGON = 0;
double PERIOD=-1, DENSITY=-1;
int GPUDeviceID = 0;
char inputFile[256] = "", outputFile[256] = "", pointsFile[256];

int main(int argc, char* argv[]){
	int idx, init_ck;
	char buf[256];
	Vec3 omega;
	FILE* fp;
	double etime;
	
	cuda_Devinfo();
	for(int files=argc==1?0:1; files<argc; files++){
		
		init_ck = ProcessInitial(argc==1? "input.txt" : argv[files]);
		if(init_ck!=1) return init_ck;
		omega.x = omega.y = 0.0;
		if(PERIOD!=0) omega.z = 2.0*M_PI/(PERIOD * 3600);
		else omega.z = 0.0;

		LoadPolygonData(inputFile, &vnum, &pnum);
		if (strcmp(pointsFile,"NONE")==0) {
			nnum=vnum;
			n_points = (Vec3 *)malloc(sizeof(Vec3)*nnum);
			GetPolygonCenter(n_points);
		} else {
			LoadPointsData(pointsFile,&nnum);
		}
		
		fprintf(stderr,"\nData Initialization...\n");
		GravitationalField(POLYGON_SCALE, GRAVITY_CONST, DENSITY);
		
		res = (Output *)malloc(sizeof(Output)*nnum); // for results
		fprintf(stderr,"Initialization finished!\n\n");

		//Recall that coordinates of vertices in the original GF inputs (e.g. Itokawa Nf=49152 models) are given in the unit of killometers.
		fprintf(stderr,"================ Input File Analysis Report ================\n");
		fprintf(stderr,"Number of vertices: %d\n",vnum);
		fprintf(stderr,"Number of edges: %d\n",edge_num);
		fprintf(stderr,"Number of polygons: %d\n",pnum);
		fprintf(stderr,"Number of points: %d\n",nnum);
		fprintf(stderr,"Center of gravity: (%f, %f, %f) km\n",centerOfGravity.x,centerOfGravity.y,centerOfGravity.z);
		fprintf(stderr,"Volume of polygon: %f m^3\n",volumeOfPolygon);
		fprintf(stderr,"============================================================\n\n");

		inp.simpleCalcBoundary = simpleCalcBoundary;
		inp.G_Sig = G_Sig;
		inp.volumeOfPolygon = volumeOfPolygon;
		inp.centerOfGravity = centerOfGravity;
		
		cudaTransfer_toDev(vnum, pnum, edge_num, vlist, plist, elist, nnum, n_points);		  
		fprintf(stderr,"Main calculation\n");
		etime = cudaKernel_call(vnum, pnum, edge_num, omega, POLYGON_SCALE, inp, nnum);
		fprintf(stderr,"Main Calculation finished!\n");

		cuda_HostUpdate(nnum, res); //Dev->Host  
		cuda_dealloc();

		fprintf(stderr,"\nSaving Results to %s\n",outputFile);
		OutputResults(outputFile,nnum,res);
		fprintf(stderr,"--------- Calculation is finished correctly! ---------\n");

		free(vlist);
		free(plist);
		free(elist);
		free(n_points);
		free(res);
			
		etime/=1000.0; //ms -> sec.
		fprintf(stderr,"Elapsed time on Device(Main computation process): %f sec.\n\n",etime);
	}	
	
	return 0;	
}

void OutputResults(char *filename, int nnum, Output *res){
	FILE* fp;
	fp=fopen(filename,"w");	

	if ( START_POLYGON == 1) {//Results Legend
		fprintf(fp, "#Data: %s\n",inputFile);
		fprintf(fp, "#Number of Polygons: %d\n",pnum);  
		fprintf(fp, "#Number of Points: %d\n",nnum);  
		fprintf(fp, "#Density: %f [kg/m^3]\n",DENSITY);  
		if (PERIOD==0.0) {
			fprintf(fp, "#Rotational Period: NO ROTATION\n");
		} else {
			fprintf(fp, "#Rotational Period: %f [h]\n",PERIOD);
		}
		fprintf(fp, "#Unit: All coordinates in km, and SI units for other values.\n"); 
		fprintf(fp, "#PlateID ");
		fprintf(fp, "Point.x Point.y Point.z ");
		fprintf(fp, "Lon Lat ");
		fprintf(fp, "CRefAccx CRefAccy CRefAccz ");
		fprintf(fp, "GravAcc.x GravAcc.y GravAcc.z ");
		fprintf(fp, "TotalAcc.x TotalAcc.y TotalAcc.z ");
		fprintf(fp, "Gpotential Rpotential Tpotential ");
		if (strcmp(pointsFile,"NONE")==0) {
			fprintf(fp, "GeopotentialSlope ");
			fprintf(fp, "Normal.x Normal.y Normal.z ");
			fprintf(fp, "Area ");
		} 
		fprintf(fp, "\n");
    }

	for(int i=0; i < nnum;i++){  
		fprintf(fp, "%d %f %f %f ", i+1,res[i].cent.x, res[i].cent.y, res[i].cent.z);
		fprintf(fp, "%f %f ", res[i].lon, res[i].lat);
		fprintf(fp, "%.6e %.6e %.6e ", res[i].centRAcc.x, res[i].centRAcc.y, res[i].centRAcc.z);
		fprintf(fp, "%.6e %.6e %.6e ", res[i].att.x, res[i].att.y, res[i].att.z);
		fprintf(fp, "%.6e %.6e %.6e ", res[i].att.x-res[i].centRAcc.x, res[i].att.y-res[i].centRAcc.y, res[i].att.z-res[i].centRAcc.z);
		fprintf(fp, "%.6e %.6e %.6e ", res[i].g_p,res[i].r_p,-1*res[i].g_p-res[i].r_p);
		if (strcmp(pointsFile,"NONE")==0) {
			fprintf(fp, "%e %e %e %e %e ", res[i].slope,res[i].nvec.x,res[i].nvec.y,res[i].nvec.z,res[i].area);
		}
		fprintf(fp, "\n");
	}
	fclose(fp);
}
