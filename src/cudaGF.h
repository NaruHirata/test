typedef struct __align__(16){
	double x;
	double y;
	double z;
}Vec3;

typedef struct __align__(16){
	int v[3];
}PLIST;

typedef struct __align__(16){
	int v_a, v_b;
	int face_a, face_b;
	Vec3 nonEdgeVerA,nonEdgeVerB;
}Edge_list;


typedef struct __align__(16){
	Vec3 cent;
	Vec3 centRAcc;
	Vec3 att;
	Vec3 nvec;
	double g_p, r_p;
    double lon, lat;
	double slope;
	double area;
}Output;

typedef struct __align__(16){
    double simpleCalcBoundary;
    double G_Sig;
    double volumeOfPolygon;
    Vec3 centerOfGravity;	
}Constant;





__device__ Vec3 Centroid(Vec3 v1, Vec3 v2, Vec3 v3){
	Vec3 V_c;
	double k=1.0;
	V_c.x = (v1.x + v2.x + v3.x)/3.0*k;
	V_c.y = (v1.y + v2.y + v3.y)/3.0*k;
	V_c.z = (v1.z + v2.z + v3.z)/3.0*k;
	return V_c;
}

__device__ Vec3 Vscale(Vec3 a, double k){
	Vec3 ka;
 	 ka.x = a.x*k;
	 ka.y = a.y*k;
	 ka.z = a.z*k;
	return ka;
}
__device__ Vec3 Vadd(Vec3 a, Vec3 b ){
	Vec3 add;
       add.x = a.x + b.x;
       add.y = a.y + b.y;
       add.z = a.z + b.z;
	return add;
}
__device__ Vec3 Vsub(Vec3 a, Vec3 b ){

	Vec3 sub;
       sub.x = a.x - b.x;
       sub.y = a.y - b.y;
       sub.z = a.z - b.z;
	return sub;
}
__device__ Vec3 Vcross(Vec3 a, Vec3 b){
	Vec3 cross;
       cross.x = a.y*b.z - a.z*b.y;
       cross.y = a.z*b.x - a.x*b.z;
       cross.z = a.x*b.y - a.y*b.x;
	return cross;
}
__device__ double Vdot( Vec3 a, Vec3 b ){
   return a.x*b.x + a.y*b.y + a.z*b.z;
}
__device__ double Vnorm( Vec3 a ){
   return sqrt(a.x*a.x + a.y*a.y + a.z*a.z);
}



__device__ void reclat( Vec3 rectan, double *radius, double *lon, double *lat){
   double vmax, x1, y1, z1;

   vmax = max( fabs(rectan.x), max( fabs(rectan.y), fabs(rectan.z) )  );
   vmax = max( fabs(rectan.x), max( fabs(rectan.y), fabs(rectan.z) )  );
    
   if ( vmax > 0.0){
      x1        = rectan.x / vmax;
      y1        = rectan.y / vmax;
      z1        = rectan.z / vmax;
      *radius   = vmax * sqrt( x1*x1 + y1*y1 + z1*z1 );
      *lat = atan2(z1, sqrt( x1*x1 + y1*y1 ) );

      if ( x1 == 0.0 && y1 == 0.0)  *lon = 0.0;
      else *lon = atan2(y1, x1);
	  
   }
   else{//the zero vector. 
      *radius = 0.0;
      *lon    = 0.0;
      *lat    = 0.0;
      }
} 


//Angle of vectors
__device__ double Vsep( Vec3 v1, Vec3 v2){
   double dmag, val_dot, angle;
   Vec3 u1,u2, vtemp;
   
   dmag = Vnorm(v1);
   if ( dmag == 0.0 ) return 0.0;
   u1 = Vscale(v1,1.0/dmag);
   
   dmag = Vnorm(v2);
   if ( dmag == 0.0 ) return 0.0;
   u2 = Vscale(v2,1.0/dmag);
    
   val_dot = Vdot(u1,u2);
   if ( val_dot > 0.0 ){
	  vtemp = Vsub(u1,u2);  
      angle = 2.00 * asin(0.50 * Vnorm(vtemp));
   }
   else if ( val_dot < 0.0 ){
      vtemp = Vadd(u1,u2);  
      angle = M_PI - 2.00 * asin(0.50 * Vnorm(vtemp));
   }
   else angle = M_PI/2.0;

   return angle;
}
