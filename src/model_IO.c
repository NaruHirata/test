/*************************************
  IO set for GFandSlope   by f.kono
**************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "GFhost.h"

int LoadPolygonData(char*, int*, int*);
int LoadPointsData(char*, int*);
int ProcessInitial(char*);

extern int START_POLYGON, END_POLYGON;
extern double PERIOD, DENSITY;
extern int GPUDeviceID;
extern char inputFile[64], outputFile[64],pointsFile[64];

extern Vec3* vlist;
extern PLIST* plist;
extern Vec3* n_points;

//default input format for GFandSlope
int LoadPolygonData(char* inputFile, int* vnum, int* pnum){
	int idx;
	FILE *fp;
	char buf[256];
	
	fprintf(stderr,"Loading Model Data...\n");
	fp=fopen(inputFile,"r");
	if(fp==NULL){
		fprintf(stderr,"[ERROR!] The model file \"%s\" was not found.\n", inputFile);	 
		return -1;
	}

	fprintf(stderr,">>> Vertices\n");
	fscanf(fp,"%d",vnum);
	int nv = *vnum;
	vlist = (Vec3 *)malloc(sizeof(Vec3)*nv);
	for(int i=0;i<nv;i++) fscanf(fp,"%d %lf %lf %lf",&idx, &vlist[i].x, &vlist[i].y, &vlist[i].z);
/*
	#ifdef SP
	   for(int i=0;i<nv;i++) fscanf(fp,"%d %f %f %f",&idx, vlist[i].x, vlist[i].y, vlist[i].z);
	#else
	   for(int i=0;i<nv;i++) fscanf(fp,"%d %lf %lf %lf",&idx, vlist[i].x, vlist[i].y, vlist[i].z);
	#endif
*/

	fprintf(stderr,">>> Triplet of Polygons\n");
	fscanf(fp,"%d",pnum);
	int np = *pnum;
	plist = (PLIST *)malloc(sizeof(PLIST)*np);
	
	//for restart	
	fprintf(stderr,">>> a\n");
	for(int i=0; i<START_POLYGON-1; i++) fgets(buf, sizeof(buf),fp);
	fprintf(stderr,">>> b\n");

	if(END_POLYGON == 0) END_POLYGON = np;
	fprintf(stderr,">>> c\n");
	
	for(int i=START_POLYGON-1;i<np;i++){
		fscanf(fp,"%d %d %d %d",&idx, &plist[i].v[0], &plist[i].v[1], &plist[i].v[2]);
		//convert into 0-origin
		plist[i].v[0]--;
		plist[i].v[1]--;
		plist[i].v[2]--;
	}

	fclose(fp);
	fprintf(stderr, "Done.\n");
	return 1;
}

int LoadPointsData(char* pointsFile, int* nnum){
  	int idx;
	FILE *fp;
	char buf[256];
	
	fprintf(stderr,"Loading Points Data...\n");
	fp=fopen(pointsFile,"r");
	if(fp==NULL){
		fprintf(stderr,"[ERROR!] The points file \"%s\" was not found.\n", pointsFile);	 
		return -1;
	}

	fprintf(stderr,">>> Points\n");
	fscanf(fp,"%d",nnum);
	int np = *nnum;
	n_points = (Vec3 *)malloc(sizeof(Vec3)*np);
	for(int i=0;i<np;i++) fscanf(fp,"%lf %lf %lf",&n_points[i].x, &n_points[i].y, &n_points[i].z);
	return 1;
}


/*
[In main func.]

int START_POLYGON = 1, END_POLYGON = 0;
double PERIOD=-1, DENSITY=-1;
int GPUDeviceID = 0;
char inputFile[64] = "", outputFile[64] = "";

int init_ck = ProcessInitial(argc==1? "input" : argv[files]);
if(init_ck!=1) return init_ck;

if(LoadPolygonData(inputFile, &vnum, vlist, &pnum, plist)!=1) return -1;
*/

int ProcessInitial(char* filename){
	FILE* fp;
	char buf[128];
	char tagname[64], arg[64]; 

	START_POLYGON = 1;
	END_POLYGON = 0;
	fp=fopen(filename,"r");
	if(fp==NULL){
		fprintf(stderr,"[ERROR!] The input file \"%s\" was not found.\n", filename);
		return -1;
	}

	while(fgets(buf,sizeof(buf),fp)!=NULL){
		if(sscanf(buf,"%[^:]:%s", tagname, arg)==2){
			if(strcmp(tagname,"period")==0) PERIOD = atof(arg);
			else if(strcmp(tagname,"density")==0) DENSITY = atof(arg);
			else if(strcmp(tagname,"input_polygon")==0) strcpy(inputFile, arg);
			else if(strcmp(tagname,"input_points")==0) strcpy(pointsFile,arg);
			else if(strcmp(tagname,"output")==0) strcpy(outputFile, arg);
			else if(strcmp(tagname,"gpu")==0) GPUDeviceID = atoi(arg);
		}
	}
	fclose(fp);
	
	if(PERIOD<0){
		fprintf(stderr,"[ERROR!] Period of the model is not specified.\n");
		return -2;
	}
	else if(DENSITY<0){
		fprintf(stderr,"[ERROR!] Density of the model is not specified.\n");
		return -3;
	}
	else if(strcmp(inputFile,"")==0){
		fprintf(stderr,"[ERROR!] Input file for calculations is not specified.\n");
		return -4;
	}
	else if(strcmp(outputFile,"")==0){
		fprintf(stderr,"[ERROR!] Output file for calculation results is not specified.\n");
		return -5;
	}
	else if(strcmp(pointsFile,"")==0){
	  fprintf(stderr,"[ERROR!] Input file for points is not specified.\n");
	  return -6;
	}
	

	fprintf(stderr,"=================== Calculation Settings ===================\n");
	fprintf(stderr,"Loaded from %s\n\n", filename);
	fprintf(stderr,"PERIOD: %f [hour]\n", PERIOD);
	fprintf(stderr,"DENSITY: %f [kg/m^3]\n", DENSITY);
	fprintf(stderr,"Input Model File: %s\n", inputFile);
	if (strcmp(pointsFile,"NONE")==0) {
		fprintf(stderr,"No Points File: Polygon centers will be used.\n");
	} else {
		fprintf(stderr,"Input Points File: %s\n",pointsFile);
	}
	fprintf(stderr,"Output File: %s\n", outputFile);
	fprintf(stderr,"Target Device(GPU) ID: %d\n", GPUDeviceID);
	fprintf(stderr,"============================================================\n\n");
	
	//fprintf(stderr,"START_POLYGON: %d\n", START_POLYGON);
    //fprintf(stderr,"END_POLYGON: %d\n", END_POLYGON);
	
	return 1;
}
