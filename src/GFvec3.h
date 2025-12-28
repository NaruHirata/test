Vec3 Vscale(Vec3, double);
Vec3 Vadd(Vec3, Vec3);
Vec3 Vsub(Vec3, Vec3);
Vec3 Vcross(Vec3, Vec3);
double Vdot(Vec3, Vec3);
double Vnorm(Vec3);
Vec3 Vcentral4(Vec3, Vec3, Vec3, Vec3, double, Vec3);



Vec3 Vscale(Vec3 a, double k){
	Vec3 ka;
 	 ka.x = a.x*k;
	 ka.y = a.y*k;
	 ka.z = a.z*k;
	return ka;
}
Vec3 Vadd(Vec3 a, Vec3 b ){
	Vec3 add;
       add.x = a.x + b.x;
       add.y = a.y + b.y;
       add.z = a.z + b.z;
	return add;
}
Vec3 Vsub(Vec3 a, Vec3 b ){

	Vec3 sub;
       sub.x = a.x - b.x;
       sub.y = a.y - b.y;
       sub.z = a.z - b.z;
	return sub;
}
Vec3 Vcross(Vec3 a, Vec3 b){
	Vec3 cross;
       cross.x = a.y*b.z - a.z*b.y;
       cross.y = a.z*b.x - a.x*b.z;
       cross.z = a.x*b.y - a.y*b.x;
	return cross;
}
double Vdot( Vec3 a, Vec3 b ){
   return a.x*b.x + a.y*b.y + a.z*b.z;
}
double Vnorm( Vec3 a ){
   return sqrt(a.x*a.x + a.y*a.y + a.z*a.z);
}
Vec3 Vcentral4(Vec3 v1, Vec3 v2, Vec3 v3, Vec3 C, double dV, Vec3 center){
	Vec3 cent;
	
	cent.x = ((v1.x+v2.x+v3.x+C.x)*0.25)*dV + center.x;
	cent.y = ((v1.y+v2.y+v3.y+C.y)*0.25)*dV + center.y;
	cent.z = ((v1.z+v2.z+v3.z+C.z)*0.25)*dV + center.z;
	return cent;
}