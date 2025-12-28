typedef struct __attribute__ ((aligned (16))){
	double x;
	double y;
	double z;
}Vec3;

typedef struct __attribute__ ((aligned (16))){
	int v[3];
}PLIST;


typedef struct __attribute__ ((aligned (16))){
	int v_a, v_b;
	int face_a, face_b;
	Vec3 nonEdgeVerA,nonEdgeVerB;
}Edge_list;

typedef struct __attribute__ ((aligned (16))){
    double simpleCalcBoundary;
    double G_Sig;
    double volumeOfPolygon;
    Vec3 centerOfGravity;
}Constant;


typedef struct node{
  int vertex;
  int idx_a;
  int idx_b;
  struct node* next;
}Node;


typedef struct __attribute__ ((aligned (16))){
	Vec3 cent;
	Vec3 centRAcc;
	Vec3 att;
	Vec3 nvec;
	double g_p, r_p;
    double lon, lat;
	double slope;
	double area;
}Output;


