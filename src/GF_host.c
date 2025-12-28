#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#ifdef SP
#include "./sp/GFhost.h"
#else
#include "GFhost.h"
#endif

#include "GFvec3.h"

typedef struct node* NodePointer;
Node **Vertex;//edge list array
Node *newnode;

void makeFaceIndex(void);
void Calc_grav_vol_Polygon(void);
void Map_Edge2Face(Node*,int,int);
int TotalEdges(void);
void EList2Array(void);
void GetPolygonCenter(Vec3*);
Vec3 Centroid(Vec3, Vec3, Vec3);
	
extern int vnum;
extern int pnum; //number of faces on a model
int edge_num; //number of edges

extern Vec3 *vlist;
extern PLIST *plist;

Edge_list *elist;

double scale,simpleCalcBoundary, G_Sig;
Vec3 centerOfGravity;
double volumeOfPolygon;


void GravitationalField(double _scale,double const_G,double const_Sigma){
	int i;
	PLIST P_A, P_B;
	scale=_scale;
	simpleCalcBoundary=-1.0;
	//const_G=6.67259e-11;  const_Sigma=1.0;
	G_Sig = const_G*const_Sigma;
	
	//Scaling for vertex coordinate
	for(i=0;i<vnum;i++){
		vlist[i]=Vscale(vlist[i],scale);
	}
	Calc_grav_vol_Polygon(); //loadPolygonFile()の代わり
	makeFaceIndex();
	
	
	//create non-edge point preliminary
	for(i=0;i<edge_num;i++){
		P_A = plist[elist[i].face_a];
		P_B = plist[elist[i].face_b];
		
		if(P_A.v[0]!= elist[i].v_a&&P_A.v[0]!=elist[i].v_b)  elist[i].nonEdgeVerA = vlist[P_A.v[0]];
		else if(P_A.v[1]!=elist[i].v_a && P_A.v[1]!=elist[i].v_b) elist[i].nonEdgeVerA = vlist[P_A.v[1]];
		else elist[i].nonEdgeVerA = vlist[P_A.v[2]];
		
		if( P_B.v[0]!=elist[i].v_a && P_B.v[0]!=elist[i].v_b)  elist[i].nonEdgeVerB = vlist[P_B.v[0]];
		else if(P_B.v[1]!=elist[i].v_a&& P_B.v[1]!=elist[i].v_b)  elist[i].nonEdgeVerB = vlist[P_B.v[1]];
		else elist[i].nonEdgeVerB = vlist[P_B.v[2]];
	}
}


void GetPolygonCenter(Vec3 *n_points){
	// Host calculation
	int i;
	for(i=0; i < vnum; i++){
		n_points[i] = Centroid(vlist[plist[i].v[0]], vlist[plist[i].v[1]], vlist[plist[i].v[2]]);			
	}
}

Vec3 Centroid(Vec3 v1, Vec3 v2, Vec3 v3){
	Vec3 V_c;
	double k=1.0;
	V_c.x = (v1.x + v2.x + v3.x)/3.0*k;
	V_c.y = (v1.y + v2.y + v3.y)/3.0*k;
	V_c.z = (v1.z + v2.z + v3.z)/3.0*k;
	return V_c;
}

void Calc_grav_vol_Polygon(void){
	int i;
	double V=0.0, dV;
	Vec3 center = {0.0,0.0,0.0};
	Vec3 C = {0.0,0.0,0.0};

	for(i=0;i<pnum;i++){
		//四面体(平行六面体/6=三角柱/3=三角錐)
		dV = Vdot( Vcross( Vsub(vlist[plist[i].v[1]],vlist[plist[i].v[0]]),  Vsub(vlist[plist[i].v[2]],vlist[plist[i].v[0]]) )  ,   Vsub(vlist[plist[i].v[0]],C)  )/6.0;
		V+=dV;
		center = Vcentral4(vlist[plist[i].v[0]],vlist[plist[i].v[1]],vlist[plist[i].v[2]],C,dV,center);
	}
	centerOfGravity=Vscale(center,1/V);
	volumeOfPolygon=V;
	// --end-- for calculation of center of gravity and volume of polygon.
	
	
}

void makeFaceIndex(void){
	int i,j;
	
	int a,b; //2辺を構成する頂点(論文の表記)
	int tmp;
	
	fprintf(stderr," makeFaceIndex start...\n");
	//隣接リスト初期化
	Vertex = (Node **)malloc(sizeof(Node*)*vnum);
	for(i=0;i<vnum;i++){
		Vertex[i] = (Node *)malloc(sizeof(Node));
		Vertex[i]->next = NULL;
	}


	fprintf(stderr," List initialized.\n");
	//三角形ごとに
	for(i=0;i<pnum;i++){
		for(j=0;j<3;j++){ // 0-1 1-2 2-0 で辺を作れる入力データ構造に既になっている
		 a = plist[i].v[j];
		 b = plist[i].v[(j+1)%3];
		 if(b<a){ tmp=a; a=b; b=tmp; }
			Map_Edge2Face(Vertex[a],b,i); //頂点対(=辺)と接面の対応づけ 
		}
	}
	
	fprintf(stderr," Map_Edge2Face completed.\n");
	edge_num = TotalEdges();		
	elist = (Edge_list *)malloc(sizeof(Edge_list)*edge_num);
	EList2Array();//elist配列(AOS)ができる
	fprintf(stderr," Array(AOS) for edges are completed.\n");
	
}

void Map_Edge2Face(Node* st, int terminal, int adj_poly){
	Node* itr;
    	
    //当該始点頂点の隣接リストをたどる
    for(itr=st; itr->next!=NULL; itr=itr->next){
		//st-terminal間の辺が登録済み(=接する三角形が1つ登録済み)
		if(itr->next->vertex==terminal){
			itr->next->idx_b = adj_poly;
			break;
		}
	}
	
	if(itr->next==NULL){
 	  newnode = (Node *)malloc(sizeof(Node));
	  newnode->vertex = terminal;
	  newnode->idx_a = adj_poly;
	  newnode->idx_b = -1;
	  newnode->next = NULL;
	  itr->next = newnode;
	}
}

int TotalEdges(void){
	int i, num=0;
	Node* itr;
    	
    //当該始点頂点の隣接リストをたどる
    for(i=0; i<vnum; i++){
		for(itr=Vertex[i]->next; itr!=NULL; itr=itr->next){
			num++;
		}
	}	
	return num;	
}

//dealloc含む
void EList2Array(void){
	int i, num=0;
	Node *itr, *delnode;
    	
    //当該始点頂点の隣接リストをたどる
    for(i=0; i<vnum; i++){
		for(itr=Vertex[i]; itr->next!=NULL; ){
			delnode = itr->next;
			elist[num].v_a = i;
			elist[num].v_b = delnode->vertex;
			elist[num].face_a = delnode->idx_a;
			elist[num].face_b = delnode->idx_b;
			num++;
			
			itr->next=delnode->next;
			free(delnode);
		}
	}
	free(Vertex);
}

int TriangleIntersect(Vec3 Orig,Vec3 dir,Vec3 v0,Vec3 v1,Vec3 v2,Vec3* vec){
	Vec3 e1,e2,pvec,tvec,qvec;
	Vec3 ret;
	double det,t,u,v;
	
	e1 = Vsub(v1,v0);
	e2 = Vsub(v2,v0);
	dir = Vscale(dir,1.0/Vnorm(dir));
	pvec=Vcross(dir,e2);
	det=Vdot(e1,pvec);
	
	tvec=Vsub(Orig,v0);
	u=Vdot(tvec,pvec);
	qvec=Vcross(tvec,e1);
	v=Vdot(dir,qvec);
	if(det > 1e-3){
		if(u < 0.0 || u > det)return 0;
		if(v < 0.0 || u+v > det)return 0;
		
	}else if(det < -(1e-3)){
		if(u > 0.0 || u < det)return 0;
		if(v > 0.0 || u+v < det)return 0;
	}else{
		return 0;
	}
	
	t=Vdot(e2,qvec)*1.0/det;
    ret = Vadd(Vscale(dir,t),Orig);
	vec->x=ret.x;
	vec->y=ret.y;
	vec->z=ret.z;
	
	return 1;
}


