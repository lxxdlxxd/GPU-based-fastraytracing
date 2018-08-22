#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include <cstdio>
#include <ctime>
#include<cuda_runtime_api.h>
#include "device_launch_parameters.h"
#include<random>
#include"common_functions_customized.h"
#include"common_functions.h"
#include <curand_kernel.h>

#define NUMPERBATCH 1024
#define PHOTONENERGY 140
#define DEVIDENT 8
#define QUEUELENGTH 10
#define BADBUFFERSIZE 100
#define NUMVECTORBIN 600

std::default_random_engine generator;
std::uniform_real_distribution<float> distribution(0.0,1.0);


const double PI=3.141592653589793238462;


const float highlimit=1;
const float lowerlimit=0.9996876464;
const float rightlimit=2*PI;
const float leftlimit=0;



/*
const float highlimit=0.00999966669;
const float lowerlimit=-0.00999966669;
const float rightlimit=0.3;
const float leftlimit=-0.3;
*/


/*
const float highlimit=0.0975327668;
const float lowerlimit=-0.0975327668;
const float rightlimit=0.3;
const float leftlimit=-0.3;
*/

/*
const float highlimit=0.0975327668;
const float lowerlimit=-0.0975327668;
const float rightlimit=0.9;
const float leftlimit=-0.9;
*/

struct specFacet{
	float n[3];
	float v1[3];
	float v2[3];
	float v3[3];
	char type;
};

struct phantom{
	float position[3];
	int num;
};

struct photonType{								//define a photon by its location, travelling direction and energy
	float position[3];
	float direction[3];
	float energy;
};

struct facetRecord{                             //record the facets on the path of the photon in a straight line and record its corresponding distance to photon position as well, this is also used to calculate the position of scattering
	int index;
	float distancePhotonToHitPoint;
	char type;
};

struct vectors{
	double vectorNorm1[3], vectorNorm2[3], vectorNorm3[3];
};


int readPart(specFacet** pointer, int* numFacet, char* directory, char type){
	FILE* fid; char * data; 
	int lSize, *numFacePointer, numRead=0, i;
	specFacet* facet; 
	float *dataContentPointer;
	fid=fopen(directory,"rb");
	if(fid==NULL){
		printf("Uable to load part file: '%s'...\n", directory);
		return 0;
	}
	else{
		printf("Successfully loaded part file: '%s'!\n", directory);
	}
	fseek(fid, 0, SEEK_END);
	lSize=ftell(fid);
	rewind(fid);
	data = (char*)malloc(sizeof(char)*lSize);
	if(fread(data,1,lSize,fid)!=lSize){
		printf("Error in reading data...\n");
		return 0;
	}
	numFacePointer=(int*)&data[80];
	*numFacet=*numFacePointer;
	printf("Num of facet is %d\n",*numFacet);

	facet = (specFacet*)malloc(sizeof(specFacet)*(*numFacet));
	if(facet==NULL){
		printf("Error in malloc...\n");
		return 0;
	}
	dataContentPointer=(float*)&data[84];
	while(numRead<(*numFacet)){
		dataContentPointer=(float*)&data[84+50*numRead];
		for(i=0;i<3;i++){
			facet[numRead].n[i]=*(dataContentPointer+i);
		}
		for(i=0;i<3;i++){
			facet[numRead].v1[i]=*(dataContentPointer+i+3);
		}
		for(i=0;i<3;i++){
			facet[numRead].v2[i]=*(dataContentPointer+i+6);
		}
		for(i=0;i<3;i++){
			facet[numRead].v3[i]=*(dataContentPointer+i+9);
		}
		facet[numRead].type=type;
		numRead+=1;
	}
	*pointer=facet;
	return 1;
}

int catenate(specFacet** stack, int* numStack, specFacet** element, int* numElement){
	int i;
	specFacet* stackSum;
	stackSum=(specFacet*)malloc(sizeof(specFacet)*(*numStack+*numElement));
	if(stackSum==NULL){
		printf("Fail to do catenate...\n");
		return 0;
	}
	for(i=0;i<*numStack;i++)
	{
		stackSum[i]=(*stack)[i];
	}
	for(i=*numStack;i<*numStack+*numElement;i++)
	{
		stackSum[i]=(*element)[i-*numStack];
//		nums=((*element)[i]).v3[2];
	}
	//free(*stack);
	*stack=stackSum;
	*numStack=*numStack+*numElement;
	return 1;
}

void rotateFacetAroundZ(specFacet *stackRotated, specFacet *stack, int numStack, float vector[3]){
	float xComponent, yComponent, norm, cosPhi, sinPhi, M[4];
	int i;

	xComponent=vector[0];
	yComponent=vector[1];

	norm=sqrt(xComponent*xComponent+yComponent*yComponent);

	if(norm==0){
		M[0]=1;
		M[1]=0;
		M[2]=0;
		M[3]=1;
	}else{
		cosPhi=xComponent/norm;
		sinPhi=yComponent/norm;

		M[0]=cosPhi;
		M[1]=-sinPhi;
		M[2]=sinPhi;
		M[3]=cosPhi;
	}

	for(i=0;i<numStack;i++){
		stackRotated[i].n[0]=stack[i].n[0]*M[0]+stack[i].n[1]*M[1];
		stackRotated[i].n[1]=stack[i].n[0]*M[2]+stack[i].n[1]*M[3];
		stackRotated[i].n[2]=stack[i].n[2];

		stackRotated[i].v1[0]=stack[i].v1[0]*M[0]+stack[i].v1[1]*M[1];
		stackRotated[i].v1[1]=stack[i].v1[0]*M[2]+stack[i].v1[1]*M[3];
		stackRotated[i].v1[2]=stack[i].v1[2];

		stackRotated[i].v2[0]=stack[i].v2[0]*M[0]+stack[i].v2[1]*M[1];
		stackRotated[i].v2[1]=stack[i].v2[0]*M[2]+stack[i].v2[1]*M[3];
		stackRotated[i].v2[2]=stack[i].v2[2];

		stackRotated[i].v3[0]=stack[i].v3[0]*M[0]+stack[i].v3[1]*M[1];
		stackRotated[i].v3[1]=stack[i].v3[0]*M[2]+stack[i].v3[1]*M[3];
		stackRotated[i].v3[2]=stack[i].v3[2];

		stackRotated[i].type=stack[i].type;
	}
}

void rotateFacetToVector(specFacet *stackRotated, specFacet *stack, int numStack, float alpha, float *centerOfAperturePosition){
	float norm, cosAlpha, sinAlpha, cosTheta, sinTheta, cosPhi, sinPhi, M[9], vectorNorm[3];
	int i;

	cosAlpha=cos(alpha);
	sinAlpha=sin(alpha);

	norm=sqrt(centerOfAperturePosition[0]*centerOfAperturePosition[0]+centerOfAperturePosition[1]*centerOfAperturePosition[1]+centerOfAperturePosition[2]*centerOfAperturePosition[2]);

	vectorNorm[0]=centerOfAperturePosition[0]/norm;
	vectorNorm[1]=centerOfAperturePosition[1]/norm;
	vectorNorm[2]=centerOfAperturePosition[2]/norm;

	if(abs(vectorNorm[2])!=1){
		cosTheta=vectorNorm[2];
		sinTheta=sqrt(1-cosTheta*cosTheta);
		cosPhi=vectorNorm[0]/sinTheta;
		sinPhi=vectorNorm[1]/sinTheta;

		M[0]=cosPhi*cosTheta*cosAlpha-sinPhi*sinAlpha;
		M[1]=-cosPhi*cosTheta*sinAlpha-sinPhi*cosAlpha;
		M[2]=cosPhi*sinTheta;
		M[3]=sinPhi*cosTheta*cosAlpha+cosPhi*sinAlpha;
		M[4]=-sinPhi*cosTheta*sinAlpha+cosPhi*cosAlpha;
		M[5]=sinPhi*sinTheta;
		M[6]=-sinTheta*cosAlpha;
		M[7]=sinTheta*sinAlpha;
		M[8]=cosTheta;
	}else if(vectorNorm[2]==1){
		M[0]=cosAlpha;
		M[1]=-sinAlpha;
		M[2]=0;
		M[3]=sinAlpha;
		M[4]=cosAlpha;
		M[5]=0;
		M[6]=0;
		M[7]=0;
		M[8]=1;
	}else if(vectorNorm[2]==-1){
		M[0]=-cosAlpha;
		M[1]=sinAlpha;
		M[2]=0;
		M[3]=-sinAlpha;
		M[4]=-cosAlpha;
		M[5]=0;
		M[6]=0;
		M[7]=0;
		M[8]=-1;
	}

	for(i=0;i<numStack;i++){
		stackRotated[i].n[0]=stack[i].n[0]*M[0]+stack[i].n[1]*M[1]+stack[i].n[2]*M[2];
		stackRotated[i].n[1]=stack[i].n[0]*M[3]+stack[i].n[1]*M[4]+stack[i].n[2]*M[5];
		stackRotated[i].n[2]=stack[i].n[0]*M[6]+stack[i].n[1]*M[7]+stack[i].n[2]*M[8];

		stackRotated[i].v1[0]=stack[i].v1[0]*M[0]+stack[i].v1[1]*M[1]+stack[i].v1[2]*M[2];
		stackRotated[i].v1[1]=stack[i].v1[0]*M[3]+stack[i].v1[1]*M[4]+stack[i].v1[2]*M[5];
		stackRotated[i].v1[2]=stack[i].v1[0]*M[6]+stack[i].v1[1]*M[7]+stack[i].v1[2]*M[8];

		stackRotated[i].v2[0]=stack[i].v2[0]*M[0]+stack[i].v2[1]*M[1]+stack[i].v2[2]*M[2];
		stackRotated[i].v2[1]=stack[i].v2[0]*M[3]+stack[i].v2[1]*M[4]+stack[i].v2[2]*M[5];
		stackRotated[i].v2[2]=stack[i].v2[0]*M[6]+stack[i].v2[1]*M[7]+stack[i].v2[2]*M[8];

		stackRotated[i].v3[0]=stack[i].v3[0]*M[0]+stack[i].v3[1]*M[1]+stack[i].v3[2]*M[2];
		stackRotated[i].v3[1]=stack[i].v3[0]*M[3]+stack[i].v3[1]*M[4]+stack[i].v3[2]*M[5];
		stackRotated[i].v3[2]=stack[i].v3[0]*M[6]+stack[i].v3[1]*M[7]+stack[i].v3[2]*M[8];

		stackRotated[i].type=stack[i].type;
	}
}

void rotationFacetEdgePointGeneration(float *MBack, float *EdgePointFinal, float *EdgePointOriginal, float alpha, float *centerOfAperturePosition){
	float norm, cosAlpha, sinAlpha, cosTheta, sinTheta, cosPhi, sinPhi, M[9], vectorOriginToCenterOfAperture[3];
	cosAlpha=cos(alpha);
	sinAlpha=sin(alpha);
	norm=sqrt(centerOfAperturePosition[0]*centerOfAperturePosition[0]+centerOfAperturePosition[1]*centerOfAperturePosition[1]+centerOfAperturePosition[2]*centerOfAperturePosition[2]);

	vectorOriginToCenterOfAperture[0]=centerOfAperturePosition[0]/norm;
	vectorOriginToCenterOfAperture[1]=centerOfAperturePosition[1]/norm;
	vectorOriginToCenterOfAperture[2]=centerOfAperturePosition[2]/norm;

	if(abs(vectorOriginToCenterOfAperture[2])!=1){
		cosTheta=vectorOriginToCenterOfAperture[2];
		sinTheta=sqrt(1-cosTheta*cosTheta);
		cosPhi=vectorOriginToCenterOfAperture[0]/sinTheta;
		sinPhi=vectorOriginToCenterOfAperture[1]/sinTheta;

		M[0]=cosPhi*cosTheta*cosAlpha-sinPhi*sinAlpha;
		M[1]=-cosPhi*cosTheta*sinAlpha-sinPhi*cosAlpha;
		M[2]=cosPhi*sinTheta;
		M[3]=sinPhi*cosTheta*cosAlpha+cosPhi*sinAlpha;
		M[4]=-sinPhi*cosTheta*sinAlpha+cosPhi*cosAlpha;
		M[5]=sinPhi*sinTheta;
		M[6]=-sinTheta*cosAlpha;
		M[7]=sinTheta*sinAlpha;
		M[8]=cosTheta;

		MBack[0]=cosPhi*cosTheta;
		MBack[1]=sinPhi*cosTheta;
		MBack[2]=-sinTheta;
		MBack[3]=-sinPhi;
		MBack[4]=cosPhi;
		MBack[5]=0;
		MBack[6]=cosPhi*sinTheta;
		MBack[7]=sinPhi*sinTheta;
		MBack[8]=cosTheta;

	}else if(vectorOriginToCenterOfAperture[2]==1){
		M[0]=cosAlpha;
		M[1]=-sinAlpha;
		M[2]=0;
		M[3]=sinAlpha;
		M[4]=cosAlpha;
		M[5]=0;
		M[6]=0;
		M[7]=0;
		M[8]=1;

		MBack[0]=1;
		MBack[1]=0;
		MBack[2]=0;
		MBack[3]=0;
		MBack[4]=1;
		MBack[5]=0;
		MBack[6]=0;
		MBack[7]=0;
		MBack[8]=1;
	}else if(vectorOriginToCenterOfAperture[2]==-1){
		M[0]=-cosAlpha;
		M[1]=sinAlpha;
		M[2]=0;
		M[3]=-sinAlpha;
		M[4]=-cosAlpha;
		M[5]=0;
		M[6]=0;
		M[7]=0;
		M[8]=-1;

		MBack[0]=1;
		MBack[1]=0;
		MBack[2]=0;
		MBack[3]=0;
		MBack[4]=1;
		MBack[5]=0;
		MBack[6]=0;
		MBack[7]=0;
		MBack[8]=1;
	}

	EdgePointFinal[0]=EdgePointOriginal[0]*M[0]+EdgePointOriginal[1]*M[1]+EdgePointOriginal[2]*M[2];
	EdgePointFinal[1]=EdgePointOriginal[0]*M[3]+EdgePointOriginal[1]*M[4]+EdgePointOriginal[2]*M[5];
	EdgePointFinal[2]=EdgePointOriginal[0]*M[6]+EdgePointOriginal[1]*M[7]+EdgePointOriginal[2]*M[8];
}

void inverseMatrixGeneration(float *M, float angleAlpha, float *vectorCA){
	float norm, cosAlpha, sinAlpha, cosTheta, sinTheta, cosPhi, sinPhi, vectorOriginToCenterOfAperture[3];

	cosAlpha=cos(angleAlpha);
	sinAlpha=sin(angleAlpha);
	norm=sqrt(vectorCA[0]*vectorCA[0]+vectorCA[1]*vectorCA[1]+vectorCA[2]*vectorCA[2]);

	vectorOriginToCenterOfAperture[0]=vectorCA[0]/norm;
	vectorOriginToCenterOfAperture[1]=vectorCA[1]/norm;
	vectorOriginToCenterOfAperture[2]=vectorCA[2]/norm;

	if(abs(vectorOriginToCenterOfAperture[2])!=1){
		cosTheta=vectorOriginToCenterOfAperture[2];
		sinTheta=sqrt(1-cosTheta*cosTheta);
		cosPhi=vectorOriginToCenterOfAperture[0]/sinTheta;
		sinPhi=vectorOriginToCenterOfAperture[1]/sinTheta;

		M[0]=cosAlpha*cosTheta*cosPhi-sinAlpha*sinPhi;
		M[1]=cosAlpha*cosTheta*sinPhi+sinAlpha*cosPhi;
		M[2]=-cosAlpha*sinTheta;
		M[3]=-sinAlpha*cosTheta*cosPhi-cosAlpha*sinPhi;
		M[4]=-sinAlpha*cosTheta*sinPhi+cosAlpha*cosPhi;
		M[5]=sinAlpha*sinTheta;
		M[6]=sinTheta*cosPhi;
		M[7]=sinTheta*sinPhi;
		M[8]=cosTheta;
	}else if(vectorOriginToCenterOfAperture[2]==1){
		M[0]=cosAlpha;
		M[1]=sinAlpha;
		M[2]=0;
		M[3]=-sinAlpha;
		M[4]=cosAlpha;
		M[5]=0;
		M[6]=0;
		M[7]=0;
		M[8]=1;
	}else if(vectorOriginToCenterOfAperture[2]==-1){
		M[0]=-cosAlpha;
		M[1]=-sinAlpha;
		M[2]=0;
		M[3]=sinAlpha;
		M[4]=-cosAlpha;
		M[5]=0;
		M[6]=0;
		M[7]=0;
		M[8]=-1;
	}
}

void rotationAngleGeneration(float *rotationSpec, float *EdgePointFinal, float *sourcePosition, float *centerOfAperturePosition){
	float vectorSCA[3], norm, vZ[3], dotProduct, vYLocal[3], vXLocal[3], cosAlpha, sinAlpha;

	vectorSCA[0]=centerOfAperturePosition[0]-sourcePosition[0];
	vectorSCA[1]=centerOfAperturePosition[1]-sourcePosition[1];
	vectorSCA[2]=centerOfAperturePosition[2]-sourcePosition[2];

	norm=sqrt(vectorSCA[0]*vectorSCA[0]+vectorSCA[1]*vectorSCA[1]+vectorSCA[2]*vectorSCA[2]);

	vectorSCA[0]=vectorSCA[0]/norm;
	vectorSCA[1]=vectorSCA[1]/norm;
	vectorSCA[2]=vectorSCA[2]/norm;

	vZ[0]=0;
	vZ[1]=0;
	vZ[2]=1;

	dotProduct=vectorSCA[0]*vZ[0]+vectorSCA[1]*vZ[1]+vectorSCA[2]*vZ[2];

	vYLocal[0]=vZ[0]-dotProduct*vectorSCA[0];
	vYLocal[1]=vZ[1]-dotProduct*vectorSCA[1];
	vYLocal[2]=vZ[2]-dotProduct*vectorSCA[2];

	norm=sqrt(vYLocal[0]*vYLocal[0]+vYLocal[1]*vYLocal[1]+vYLocal[2]*vYLocal[2]);

	if(norm!=0){
		vYLocal[0]=vYLocal[0]/norm;
		vYLocal[1]=vYLocal[1]/norm;
		vYLocal[2]=vYLocal[2]/norm;

		vXLocal[0]=vectorSCA[2]*vYLocal[1]-vectorSCA[1]*vYLocal[2];
		vXLocal[1]=vectorSCA[0]*vYLocal[2]-vectorSCA[2]*vYLocal[0];
		vXLocal[2]=vectorSCA[1]*vYLocal[0]-vectorSCA[0]*vYLocal[1];
	}else if(dotProduct>0){
		vYLocal[0]=-1;
		vYLocal[1]=0;
		vYLocal[2]=0;

		vXLocal[0]=0;
		vXLocal[1]=1;
		vXLocal[2]=0;
	}else if(dotProduct<0){
		vYLocal[0]=1;
		vYLocal[1]=0;
		vYLocal[2]=0;

		vXLocal[0]=0;
		vXLocal[1]=1;
		vXLocal[2]=0;
	}
	cosAlpha=EdgePointFinal[0]*vXLocal[0]+EdgePointFinal[1]*vXLocal[1]+EdgePointFinal[2]*vXLocal[2];
	sinAlpha=EdgePointFinal[0]*vYLocal[0]+EdgePointFinal[1]*vYLocal[1]+EdgePointFinal[2]*vYLocal[2];
	norm=sqrt(cosAlpha*cosAlpha+sinAlpha*sinAlpha);
	cosAlpha=cosAlpha/norm;
	sinAlpha=sinAlpha/norm;
	rotationSpec[0]=cosAlpha;
	rotationSpec[1]=sinAlpha;
}

void velocityRotationMatrixGeneration(float *M, float *rotationSpec, float *sourcePosition, float *centerOfAperturePosition, int flag){
	float cosAlpha, sinAlpha, cosTheta, sinTheta, cosPhi, sinPhi, vectorSourceToCA[3], norm;

	cosAlpha=rotationSpec[0];
	sinAlpha=rotationSpec[1];

	vectorSourceToCA[0]=centerOfAperturePosition[0]-sourcePosition[0];
	vectorSourceToCA[1]=centerOfAperturePosition[1]-sourcePosition[1];
	vectorSourceToCA[2]=centerOfAperturePosition[2]-sourcePosition[2];

	norm=sqrt(vectorSourceToCA[0]*vectorSourceToCA[0]+vectorSourceToCA[1]*vectorSourceToCA[1]+vectorSourceToCA[2]*vectorSourceToCA[2]);

	vectorSourceToCA[0]=vectorSourceToCA[0]/norm;
	vectorSourceToCA[1]=vectorSourceToCA[1]/norm;
	vectorSourceToCA[2]=vectorSourceToCA[2]/norm;

	if(flag==0){
		if(abs(vectorSourceToCA[2])!=1){
			cosTheta=vectorSourceToCA[2];
			sinTheta=sqrt(1-cosTheta*cosTheta);
			cosPhi=vectorSourceToCA[0]/sinTheta;
			sinPhi=vectorSourceToCA[1]/sinTheta;

			M[0]=cosPhi*cosTheta*cosAlpha-sinPhi*sinAlpha;
			M[1]=-cosPhi*cosTheta*sinAlpha-sinPhi*cosAlpha;
			M[2]=cosPhi*sinTheta;
			M[3]=sinPhi*cosTheta*cosAlpha+cosPhi*sinAlpha;
			M[4]=-sinPhi*cosTheta*sinAlpha+cosPhi*cosAlpha;
			M[5]=sinPhi*sinTheta;
			M[6]=-sinTheta*cosAlpha;
			M[7]=sinTheta*sinAlpha;
			M[8]=cosTheta;
		}else if(vectorSourceToCA[2]==-1){
			M[0]=-cosAlpha;
			M[1]=sinAlpha;
			M[2]=0;
			M[3]=-sinAlpha;
			M[4]=-cosAlpha;
			M[5]=0;
			M[6]=0;
			M[7]=0;
			M[8]=-1;
		}
		else if(vectorSourceToCA[2]==1){
			M[0]=cosAlpha;
			M[1]=-sinAlpha;
			M[2]=0;
			M[3]=sinAlpha;
			M[4]=cosAlpha;
			M[5]=0;
			M[6]=0;
			M[7]=0;
			M[8]=1;
		}
	}else if(flag==1){
		if(abs(vectorSourceToCA[2])!=1){
			cosTheta=vectorSourceToCA[2];
			sinTheta=sqrt(1-cosTheta*cosTheta);
			cosPhi=vectorSourceToCA[0]/sinTheta;
			sinPhi=vectorSourceToCA[1]/sinTheta;

			M[0]=cosPhi*sinTheta;
			M[1]=-sinPhi*cosAlpha-cosPhi*cosTheta*sinAlpha;
			M[2]=sinPhi*sinAlpha-cosPhi*cosTheta*cosAlpha;
			M[3]=sinPhi*sinTheta;
			M[4]=cosPhi*cosAlpha-sinPhi*cosTheta*sinAlpha;
			M[5]=-cosPhi*sinAlpha-sinPhi*cosTheta*cosAlpha;
			M[6]=cosTheta;
			M[7]=sinTheta*sinAlpha;
			M[8]=sinTheta*cosAlpha;
		}else if(vectorSourceToCA[2]==-1){
			M[0]=0;
			M[1]=sinAlpha;
			M[2]=cosAlpha;
			M[3]=0;
			M[4]=cosAlpha;
			M[5]=-sinAlpha;
			M[6]=-1;
			M[7]=0;
			M[8]=0;
		}
		else if(vectorSourceToCA[2]==1){
			M[0]=0;
			M[1]=-sinAlpha;
			M[2]=-cosAlpha;
			M[3]=0;
			M[4]=cosAlpha;
			M[5]=-sinAlpha;
			M[6]=1;
			M[7]=0;
			M[8]=0;
		}
	}
}

__global__ void velocityCalculation(float *nv_dev, float *solidAnglePro_dev, float area, float distOriginToDetector, float *photonPosition_dev, int numLocalX, int numLocalY, int idxX, int idxY, float pitch, int numTotalPixel){
	int idx=blockDim.x*blockIdx.x+threadIdx.x, idx_x, idx_y;
	if(idx<numTotalPixel){
		float norm, vectorPhotonToPixel[3];

		idx_x=idx/numLocalY-numLocalX/2;
		idx_y=idx%numLocalY-numLocalY/2;

		vectorPhotonToPixel[0]=(idxX+idx_x)*pitch+pitch/2-photonPosition_dev[0];
		vectorPhotonToPixel[1]=(idxY+idx_y)*pitch+pitch/2-photonPosition_dev[1];
		vectorPhotonToPixel[2]=distOriginToDetector-photonPosition_dev[2];

		norm=sqrt(vectorPhotonToPixel[0]*vectorPhotonToPixel[0]+vectorPhotonToPixel[1]*vectorPhotonToPixel[1]+vectorPhotonToPixel[2]*vectorPhotonToPixel[2]);

		nv_dev[idx*3]=vectorPhotonToPixel[0]/norm;
		nv_dev[idx*3+1]=vectorPhotonToPixel[1]/norm;
		nv_dev[idx*3+2]=vectorPhotonToPixel[2]/norm;

		solidAnglePro_dev[idx]=area/pow(norm,3)*vectorPhotonToPixel[2];
	}
}

void velocityVectorGeneration(int *cors, float *solidAnglePro, float *solidAnglePro_dev, float *nv, float *nv_dev, float distOriginToCA, float distOriginToDetector, float *photonPosition_dev, int numLocalX, int numLocalY, float pitch, float *photonPosition){
	int numTotalPixel;
	float norm, dotProduct, SCA[3], photonPositionInteraction[3], distPhotonToDetector, distPhotonToHitPoint, area;

	area=pitch*pitch;
	numTotalPixel=numLocalX*numLocalY;

	SCA[0]=-photonPosition[0];
	SCA[1]=-photonPosition[1];
	SCA[2]=distOriginToCA-photonPosition[2];

	norm=sqrt(SCA[0]*SCA[0]+SCA[1]*SCA[1]+SCA[2]*SCA[2]);

	SCA[0]=SCA[0]/norm;
	SCA[1]=SCA[1]/norm;
	SCA[2]=SCA[2]/norm;

	dotProduct=SCA[2];

	distPhotonToDetector=distOriginToDetector-photonPosition[2];
	distPhotonToHitPoint=distPhotonToDetector/dotProduct;

	photonPositionInteraction[0]=photonPosition[0]+distPhotonToHitPoint*SCA[0];
	photonPositionInteraction[1]=photonPosition[1]+distPhotonToHitPoint*SCA[1];
	photonPositionInteraction[2]=photonPosition[2]+distPhotonToHitPoint*SCA[2];

	cors[0]=floor(photonPositionInteraction[0]/pitch);
	cors[1]=floor(photonPositionInteraction[1]/pitch);

	velocityCalculation<<<numTotalPixel/32,32>>>(nv_dev, solidAnglePro_dev, area, distOriginToDetector, photonPosition_dev, numLocalX, numLocalY, cors[0], cors[1], pitch, numTotalPixel);
	cudaDeviceSynchronize();

	cudaMemcpy(nv, nv_dev, sizeof(float)*numTotalPixel*3, cudaMemcpyDeviceToHost);
	cudaMemcpy(solidAnglePro, solidAnglePro_dev, sizeof(float)*numTotalPixel, cudaMemcpyDeviceToHost);
}

__global__ void preparation(vectors *vectorsResult_dev, specFacet* stack_dev, float* photonPosition_dev, int numStack){
	int idx=blockDim.x*blockIdx.x+threadIdx.x, sign;
	double vectorPhotonToVertice1[3], vectorPhotonToVertice2[3], vectorPhotonToVertice3[3], temp; 
	if(idx<numStack){
		vectorPhotonToVertice1[0]=stack_dev[idx].v1[0]-photonPosition_dev[0];
		vectorPhotonToVertice1[1]=stack_dev[idx].v1[1]-photonPosition_dev[1];
		vectorPhotonToVertice1[2]=stack_dev[idx].v1[2]-photonPosition_dev[2];

		vectorPhotonToVertice2[0]=stack_dev[idx].v2[0]-photonPosition_dev[0];
		vectorPhotonToVertice2[1]=stack_dev[idx].v2[1]-photonPosition_dev[1];
		vectorPhotonToVertice2[2]=stack_dev[idx].v2[2]-photonPosition_dev[2];

		vectorPhotonToVertice3[0]=stack_dev[idx].v3[0]-photonPosition_dev[0];
		vectorPhotonToVertice3[1]=stack_dev[idx].v3[1]-photonPosition_dev[1];
		vectorPhotonToVertice3[2]=stack_dev[idx].v3[2]-photonPosition_dev[2];

		vectorsResult_dev[idx].vectorNorm1[0]=vectorPhotonToVertice2[1]*vectorPhotonToVertice3[2]-vectorPhotonToVertice2[2]*vectorPhotonToVertice3[1];
		vectorsResult_dev[idx].vectorNorm1[1]=vectorPhotonToVertice2[2]*vectorPhotonToVertice3[0]-vectorPhotonToVertice2[0]*vectorPhotonToVertice3[2];
		vectorsResult_dev[idx].vectorNorm1[2]=vectorPhotonToVertice2[0]*vectorPhotonToVertice3[1]-vectorPhotonToVertice2[1]*vectorPhotonToVertice3[0];

		vectorsResult_dev[idx].vectorNorm2[0]=vectorPhotonToVertice3[1]*vectorPhotonToVertice1[2]-vectorPhotonToVertice3[2]*vectorPhotonToVertice1[1];
		vectorsResult_dev[idx].vectorNorm2[1]=vectorPhotonToVertice3[2]*vectorPhotonToVertice1[0]-vectorPhotonToVertice3[0]*vectorPhotonToVertice1[2];
		vectorsResult_dev[idx].vectorNorm2[2]=vectorPhotonToVertice3[0]*vectorPhotonToVertice1[1]-vectorPhotonToVertice3[1]*vectorPhotonToVertice1[0];

		vectorsResult_dev[idx].vectorNorm3[0]=vectorPhotonToVertice1[1]*vectorPhotonToVertice2[2]-vectorPhotonToVertice1[2]*vectorPhotonToVertice2[1];
		vectorsResult_dev[idx].vectorNorm3[1]=vectorPhotonToVertice1[2]*vectorPhotonToVertice2[0]-vectorPhotonToVertice1[0]*vectorPhotonToVertice2[2];
		vectorsResult_dev[idx].vectorNorm3[2]=vectorPhotonToVertice1[0]*vectorPhotonToVertice2[1]-vectorPhotonToVertice1[1]*vectorPhotonToVertice2[0];

		temp=vectorPhotonToVertice1[0]*vectorsResult_dev[idx].vectorNorm1[0]+vectorPhotonToVertice1[1]*vectorsResult_dev[idx].vectorNorm1[1]+vectorPhotonToVertice1[2]*vectorsResult_dev[idx].vectorNorm1[2];
		sign=(temp>0)-(temp<0);


		vectorsResult_dev[idx].vectorNorm1[0]=vectorsResult_dev[idx].vectorNorm1[0]*sign;
		vectorsResult_dev[idx].vectorNorm1[1]=vectorsResult_dev[idx].vectorNorm1[1]*sign;
		vectorsResult_dev[idx].vectorNorm1[2]=vectorsResult_dev[idx].vectorNorm1[2]*sign;

		vectorsResult_dev[idx].vectorNorm2[0]=vectorsResult_dev[idx].vectorNorm2[0]*sign;
		vectorsResult_dev[idx].vectorNorm2[1]=vectorsResult_dev[idx].vectorNorm2[1]*sign;
		vectorsResult_dev[idx].vectorNorm2[2]=vectorsResult_dev[idx].vectorNorm2[2]*sign;

		vectorsResult_dev[idx].vectorNorm3[0]=vectorsResult_dev[idx].vectorNorm3[0]*sign;
		vectorsResult_dev[idx].vectorNorm3[1]=vectorsResult_dev[idx].vectorNorm3[1]*sign;
		vectorsResult_dev[idx].vectorNorm3[2]=vectorsResult_dev[idx].vectorNorm3[2]*sign;
	}												
}

__global__ void scout(char* workPlace_dev, vectors* vectorsResult_dev, float* nv_dev, int numStack, int sizeBlock){
	int idxNv, idxFacet;
	idxNv=blockIdx.x*3;
	idxFacet=sizeBlock*blockIdx.y+threadIdx.x;

	if(idxFacet<numStack){
		workPlace_dev[blockIdx.x*numStack+idxFacet]=((vectorsResult_dev[idxFacet].vectorNorm1[0]*nv_dev[idxNv]+vectorsResult_dev[idxFacet].vectorNorm1[1]*nv_dev[idxNv+1]+vectorsResult_dev[idxFacet].vectorNorm1[2]*nv_dev[idxNv+2])>=0)&&((vectorsResult_dev[idxFacet].vectorNorm2[0]*nv_dev[idxNv]+vectorsResult_dev[idxFacet].vectorNorm2[1]*nv_dev[idxNv+1]+vectorsResult_dev[idxFacet].vectorNorm2[2]*nv_dev[idxNv+2])>=0)&&((vectorsResult_dev[idxFacet].vectorNorm3[0]*nv_dev[idxNv]+vectorsResult_dev[idxFacet].vectorNorm3[1]*nv_dev[idxNv+1]+vectorsResult_dev[idxFacet].vectorNorm3[2]*nv_dev[idxNv+2])>=0);
	}
}

int seekIntersectionFacet(facetRecord* facetPenetrated, photonType *photon,specFacet *facets, int numFacet, int facetIdx){
	float vectorPhotonToVertice1[3], distancePhotonToHitPointL, nv[3], nf[3];
	specFacet *pointerL=facets+facetIdx;

	nv[0]=(*photon).direction[0];
	nv[1]=(*photon).direction[1];
	nv[2]=(*photon).direction[2];

	nf[0]=(*pointerL).n[0];
	nf[1]=(*pointerL).n[1];
	nf[2]=(*pointerL).n[2];
	
	vectorPhotonToVertice1[0]=(*pointerL).v1[0]-(*photon).position[0];
	vectorPhotonToVertice1[1]=(*pointerL).v1[1]-(*photon).position[1];
	vectorPhotonToVertice1[2]=(*pointerL).v1[2]-(*photon).position[2];

	distancePhotonToHitPointL=(vectorPhotonToVertice1[0]*nf[0]+vectorPhotonToVertice1[1]*nf[1]+vectorPhotonToVertice1[2]*nf[2])/(nf[0]*nv[0]+nf[1]*nv[1]+nf[2]*nv[2]);
	if(distancePhotonToHitPointL>0){
		facetPenetrated[0].index=facetIdx;
		facetPenetrated[0].distancePhotonToHitPoint=distancePhotonToHitPointL;
		facetPenetrated[0].type=(*pointerL).type;
		return 1;
	}

	return 0;
}

int reorderSmallToLarge(facetRecord* faceletPenetrated, int *numFaceletPenetrated){
	facetRecord tempL;
	int iL, jL;
	for(iL=*numFaceletPenetrated-2; iL>=0; iL--){
		for(jL=0; jL<iL+1; jL++){
			if(faceletPenetrated[jL].distancePhotonToHitPoint>faceletPenetrated[jL+1].distancePhotonToHitPoint){
				tempL.index=faceletPenetrated[jL+1].index;
				tempL.distancePhotonToHitPoint=faceletPenetrated[jL+1].distancePhotonToHitPoint;
				tempL.type=faceletPenetrated[jL+1].type;
				faceletPenetrated[jL+1].index=faceletPenetrated[jL].index;
				faceletPenetrated[jL+1].distancePhotonToHitPoint=faceletPenetrated[jL].distancePhotonToHitPoint;
				faceletPenetrated[jL+1].type=faceletPenetrated[jL].type;
				faceletPenetrated[jL].index=tempL.index;
				faceletPenetrated[jL].distancePhotonToHitPoint=tempL.distancePhotonToHitPoint;
				faceletPenetrated[jL].type=tempL.type;
			}
		}
	}
	for(iL=0;iL<*numFaceletPenetrated-1;iL++){
		if(faceletPenetrated[iL].distancePhotonToHitPoint==faceletPenetrated[iL+1].distancePhotonToHitPoint){
			if(iL+2<*numFaceletPenetrated){
				for(jL=iL+1;jL<*numFaceletPenetrated-1;jL++){
					faceletPenetrated[jL].index=faceletPenetrated[jL+1].index;
					faceletPenetrated[jL].distancePhotonToHitPoint=faceletPenetrated[jL+1].distancePhotonToHitPoint;
					faceletPenetrated[jL].type=faceletPenetrated[jL+1].type;
				}
				*numFaceletPenetrated=*numFaceletPenetrated-1;
			}
		}
	}
	return 1;
}

int attenuation(photonType* photonBuff, photonType* photon, facetRecord* faceletPenetrated, int numFaceletPenetrated){
	int i=0, num_t;
	float trans, distance=0, numRandom, lambda_t=3.935855, lambda_d=0.28194408;	//3.935855

	if(faceletPenetrated[numFaceletPenetrated-1].type=='d'){
		num_t=(numFaceletPenetrated>>1)-1;
		if(num_t>=1){
			for(i=0;i<num_t;i++){
				distance=distance+faceletPenetrated[(i<<1)+1].distancePhotonToHitPoint-faceletPenetrated[(i<<1)].distancePhotonToHitPoint;
			}
		}
		trans=exp(-lambda_t*distance);
		numRandom=distribution(generator);
		if(numRandom<=trans){	
			numRandom=distribution(generator);
			distance=faceletPenetrated[(i<<1)+1].distancePhotonToHitPoint-faceletPenetrated[(i<<1)].distancePhotonToHitPoint;
			trans=exp(-lambda_d*distance);
			if(numRandom>trans){
				numRandom=distribution(generator);
				distance=-log(1-(1-trans)*numRandom)/lambda_d+faceletPenetrated[(i<<1)].distancePhotonToHitPoint;
				(*photonBuff).position[0]=(*photon).position[0]+distance*(*photon).direction[0];
				(*photonBuff).position[1]=(*photon).position[1]+distance*(*photon).direction[1];
				(*photonBuff).position[2]=(*photon).position[2]+distance*(*photon).direction[2];
				(*photonBuff).direction[0]=(*photon).direction[0];
				(*photonBuff).direction[1]=(*photon).direction[1];
				(*photonBuff).direction[2]=(*photon).direction[2];
				(*photonBuff).energy=(*photon).energy;
				return 1;
			}
		}
	}
	return 0;
}

float attenuationPointSpreadFunction(photonType* photonBuff, photonType* photon, facetRecord* faceletPenetrated, int numFaceletPenetrated){
	int i=0, num_t;
	float trans, distance=0, lambda_t=3.935855;	

	if(faceletPenetrated[numFaceletPenetrated-1].type=='d'){
		num_t=(numFaceletPenetrated>>1)-1;
		if(num_t>=1){
			for(i=0;i<num_t;i++){
				distance=distance+faceletPenetrated[(i<<1)+1].distancePhotonToHitPoint-faceletPenetrated[(i<<1)].distancePhotonToHitPoint;
			}
		}
		trans=exp(-lambda_t*distance);
		return trans;
	}
	return 0;
}

__global__ void randomNumberGeneration(double *randomArray, curandState *state_dev, unsigned long int accumulatedNum){
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	curand_init((accumulatedNum*NUMPERBATCH*2+idx), 2, 0, &state_dev[idx]);
	randomArray[idx]=curand_uniform(&state_dev[idx]);
}

/*
__global__ void randomNumberGenerationTest(float *randomArray, curandState *state_dev, unsigned long int accumulatedNum){
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	curand_init(idx, 0, 1, &state_dev[idx]);
	randomArray[idx]=curand_uniform(&state_dev[idx]);
}
*/

__global__ void speedVectorGeneration(float *nv_dev, double *randomArray, const float highlimit, const float lowerlimit, const float rightlimit, const float leftlimit, const double PI, float *matrixRotation_dev){
	float phi, vTemp[3];
	double sinTheta, cosTheta;
	int idx=blockIdx.x*blockDim.x+threadIdx.x;

	cosTheta=highlimit-randomArray[idx<<1]*(highlimit-lowerlimit);
	sinTheta=sqrt(1-cosTheta*cosTheta);
	phi=rightlimit-randomArray[(idx<<1)+1]*(rightlimit-leftlimit);

	vTemp[0]=sinTheta*cos(phi);
	vTemp[1]=sinTheta*sin(phi);
	vTemp[2]=cosTheta;

	nv_dev[idx*3]=vTemp[0]*matrixRotation_dev[0]+vTemp[1]*matrixRotation_dev[1]+vTemp[2]*matrixRotation_dev[2];
	nv_dev[idx*3+1]=vTemp[0]*matrixRotation_dev[3]+vTemp[1]*matrixRotation_dev[4]+vTemp[2]*matrixRotation_dev[5];
	nv_dev[idx*3+2]=vTemp[0]*matrixRotation_dev[6]+vTemp[1]*matrixRotation_dev[7]+vTemp[2]*matrixRotation_dev[8];
}

void fastRayTracingBinMode(int *detector, float detectorWidth, float detectorHeight, float pitch, float *nv, float *nv_dev, double *randomArray_dev, curandState *state_dev, int numPhoton, float *photonPosition, float *photonPosition_dev, specFacet *stack, specFacet *stack_dev, int numStack, char *workPlace, char *workPlace_dev, vectors *vectorsResult_dev, unsigned long int *accumulatedNum, float *matrixRotation_dev, float *MBack){
	int i, j, k, l, numVelo, numBatch, numResidual, sizeBlock=256, numFacetPenetrated, detectorIdxWidth, detectorIdxHeight, detectorNumWidth=detectorWidth/pitch, detectorNumHeight=detectorHeight/pitch;
	float position[3];
	long long* pointer_stride;
	photonType photon, photonBuff;
	facetRecord facetPenetrated[QUEUELENGTH];

	numVelo=numStack/sizeBlock+1;
	dim3 gridBlock(NUMPERBATCH, numVelo);
	numBatch=numPhoton/NUMPERBATCH;
	numResidual=numPhoton%NUMPERBATCH;

	cudaMemcpy(photonPosition_dev, photonPosition, sizeof(float)*3, cudaMemcpyHostToDevice);
	preparation<<<numVelo,sizeBlock>>>(vectorsResult_dev, stack_dev, photonPosition_dev, numStack);
	cudaDeviceSynchronize();

	for(i=0;i<numBatch;i++){
		randomNumberGeneration<<<NUMPERBATCH/sizeBlock*2,sizeBlock>>>(randomArray_dev, state_dev, *accumulatedNum);
		cudaDeviceSynchronize();
		speedVectorGeneration<<<NUMPERBATCH/sizeBlock,sizeBlock>>>(nv_dev, randomArray_dev, highlimit, lowerlimit, rightlimit, leftlimit, PI, matrixRotation_dev);
		cudaDeviceSynchronize();
		scout<<<gridBlock,sizeBlock>>>(workPlace_dev, vectorsResult_dev, nv_dev, numStack, sizeBlock);
		cudaDeviceSynchronize();
		cudaMemcpy(nv, nv_dev, sizeof(float)*3*NUMPERBATCH, cudaMemcpyDeviceToHost);
		cudaMemcpy(workPlace, workPlace_dev, sizeof(char)*numStack*NUMPERBATCH, cudaMemcpyDeviceToHost);

		for(j=0;j<NUMPERBATCH;j++){
			numFacetPenetrated=0;
			photon.position[0]=photonPosition[0];
			photon.position[1]=photonPosition[1];
			photon.position[2]=photonPosition[2];
			photon.direction[0]=nv[3*j];
			photon.direction[1]=nv[3*j+1];
			photon.direction[2]=nv[3*j+2];								//generate the photon direction
			photon.energy=PHOTONENERGY;

			pointer_stride=(long long*)&workPlace[j*numStack];
				
			for(k=0;k<numStack-DEVIDENT;k=k+DEVIDENT){									//the workPlace is tested by the extending it to "long long" (8 bytes) other than the original "char" (1 byte), the purpose of this is to go through workPlace faster, since it is sparse
				if(*pointer_stride){
					for(l=0;l<DEVIDENT;l++){
						if(workPlace[k+l+j*numStack]){
							if(seekIntersectionFacet(&facetPenetrated[numFacetPenetrated], &photon, stack, numStack, k+l)){
								numFacetPenetrated=numFacetPenetrated+1;					//check the whether the facets from scout has been penetrated by the photon, if so, record the facet
							}
						}
					}
				}
				pointer_stride+=1;
			}

			for(l=0;l<numStack-k;l++){
				if(workPlace[k+l+j*numStack]){
					if(seekIntersectionFacet(&facetPenetrated[numFacetPenetrated], &photon, stack, numStack, k+l)){
						numFacetPenetrated=numFacetPenetrated+1;					//check the whether the facets from scout has been penetrated by the photon, if so, record the facet
					}
				}
			}
			if(numFacetPenetrated!=0&&numFacetPenetrated%2==0){
				reorderSmallToLarge(facetPenetrated, &numFacetPenetrated);			//reorder the facets in facetPenetrated
				if(attenuation(&photonBuff, &photon, facetPenetrated,numFacetPenetrated)==1){	//if the function attenuation returns a value, then that means the photon has been successfully detected by the detector
					position[0]=photonBuff.position[0]*MBack[0]+photonBuff.position[1]*MBack[1]+photonBuff.position[2]*MBack[2];
					position[1]=photonBuff.position[0]*MBack[3]+photonBuff.position[1]*MBack[4]+photonBuff.position[2]*MBack[5];
					position[2]=photonBuff.position[0]*MBack[6]+photonBuff.position[1]*MBack[7]+photonBuff.position[2]*MBack[8];

					detectorIdxWidth=floor((position[1]+detectorWidth/2)/pitch);
					detectorIdxHeight=floor((position[0]+detectorWidth/2)/pitch);

					if((detectorIdxWidth>=0&&detectorIdxWidth<detectorNumWidth)&&(detectorIdxHeight>=0&&detectorIdxHeight<detectorNumHeight)){
						detector[detectorIdxWidth*detectorNumWidth+detectorIdxHeight]=detector[detectorIdxWidth*detectorNumWidth+detectorIdxHeight]+1;
					}
				}
			}
		}
		*accumulatedNum=*accumulatedNum+1;
		//printf("batch number %d is finished\n",i);
	}
	randomNumberGeneration<<<NUMPERBATCH/sizeBlock*2,sizeBlock>>>(randomArray_dev, state_dev, *accumulatedNum);
	cudaDeviceSynchronize();
	speedVectorGeneration<<<NUMPERBATCH/sizeBlock,sizeBlock>>>(nv_dev, randomArray_dev, highlimit, lowerlimit, rightlimit, leftlimit, PI, matrixRotation_dev);
	cudaDeviceSynchronize();
	scout<<<gridBlock,sizeBlock>>>(workPlace_dev, vectorsResult_dev, nv_dev, numStack, sizeBlock);
	cudaDeviceSynchronize();
	cudaMemcpy(nv, nv_dev, sizeof(float)*3*NUMPERBATCH, cudaMemcpyDeviceToHost);
	cudaMemcpy(workPlace, workPlace_dev, sizeof(char)*numStack*NUMPERBATCH, cudaMemcpyDeviceToHost);
	for(i=0;i<numResidual;i++){
		numFacetPenetrated=0;
		photon.position[0]=photonPosition[0];
		photon.position[1]=photonPosition[1];
		photon.position[2]=photonPosition[2];
		photon.direction[0]=nv[3*i];
		photon.direction[1]=nv[3*i+1];
		photon.direction[2]=nv[3*i+2];								//generate the photon direction
		photon.energy=PHOTONENERGY;

		pointer_stride=(long long*)&workPlace[i*numStack];
				
		for(k=0;k<numStack-DEVIDENT;k=k+DEVIDENT){									//the workPlace is tested by the extending it to "long long" (8 bytes) other than the original "char" (1 byte), the purpose of this is to go through workPlace faster, since it is sparse
			if(*pointer_stride){
				for(l=0;l<DEVIDENT;l++){
					if(workPlace[k+l+i*numStack]){
						if(seekIntersectionFacet(&facetPenetrated[numFacetPenetrated], &photon, stack, numStack, k+l)){
							numFacetPenetrated=numFacetPenetrated+1;					//check the whether the facets from scout has been penetrated by the photon, if so, record the facet
						}
					}
				}
			}
			pointer_stride+=1;
		}

		for(l=0;l<numStack-k;l++){
			if(workPlace[k+l+i*numStack]){
				if(seekIntersectionFacet(&facetPenetrated[numFacetPenetrated], &photon, stack, numStack, k+l)){
					numFacetPenetrated=numFacetPenetrated+1;					//check the whether the facets from scout has been penetrated by the photon, if so, record the facet
				}
			}
		}
		if(numFacetPenetrated!=0&&numFacetPenetrated%2==0){
			reorderSmallToLarge(facetPenetrated, &numFacetPenetrated);			//reorder the facets in facetPenetrated
			if(attenuation(&photonBuff, &photon, facetPenetrated,numFacetPenetrated)==1){	//if the function attenuation returns a value, then that means the photon has been successfully detected by the detector
				position[0]=photonBuff.position[0]*MBack[0]+photonBuff.position[1]*MBack[1]+photonBuff.position[2]*MBack[2];
				position[1]=photonBuff.position[0]*MBack[3]+photonBuff.position[1]*MBack[4]+photonBuff.position[2]*MBack[5];
				position[2]=photonBuff.position[0]*MBack[6]+photonBuff.position[1]*MBack[7]+photonBuff.position[2]*MBack[8];

				detectorIdxWidth=floor((position[1]+detectorWidth/2)/pitch);
				detectorIdxHeight=floor((position[0]+detectorWidth/2)/pitch);

				if(detectorIdxWidth==48&&detectorIdxHeight==48){
					detectorIdxWidth=detectorIdxWidth;
				}

				if((detectorIdxWidth>=0&&detectorIdxWidth<1000)&&(detectorIdxHeight>=0&&detectorIdxHeight<1000)){
					detector[detectorIdxWidth*detectorNumWidth+detectorIdxHeight]=detector[detectorIdxWidth*detectorNumWidth+detectorIdxHeight]+1;
				}
			}
		}
	}
}

void fastRayTracingListMode(FILE *fidw, float *nv, float *nv_dev, double *randomArray_dev, curandState *state_dev, int numPhoton, float *photonPosition, float *photonPosition_dev, specFacet *stack, specFacet *stack_dev, int numStack, char *workPlace, char *workPlace_dev, vectors *vectorsResult_dev, unsigned long int *accumulatedNum, float *matrixRotation_dev, float *MBack){
	int i, j, k, l, numVelo, numBatch, numResidual, sizeBlock=256, numFacetPenetrated;
	float position[3];
	long long* pointer_stride;
	photonType photon, photonBuff;
	facetRecord facetPenetrated[QUEUELENGTH];

	numVelo=numStack/sizeBlock+1;
	dim3 gridBlock(NUMPERBATCH, numVelo);
	numBatch=numPhoton/NUMPERBATCH;
	numResidual=numPhoton%NUMPERBATCH;

	cudaMemcpy(photonPosition_dev, photonPosition, sizeof(float)*3, cudaMemcpyHostToDevice);
	preparation<<<numVelo,sizeBlock>>>(vectorsResult_dev, stack_dev, photonPosition_dev, numStack);
	cudaDeviceSynchronize();

	for(i=0;i<numBatch;i++){
		randomNumberGeneration<<<NUMPERBATCH/sizeBlock*2,sizeBlock>>>(randomArray_dev, state_dev, *accumulatedNum);
		cudaDeviceSynchronize();
		speedVectorGeneration<<<NUMPERBATCH/sizeBlock,sizeBlock>>>(nv_dev, randomArray_dev, highlimit, lowerlimit, rightlimit, leftlimit, PI, matrixRotation_dev);
		cudaDeviceSynchronize();
		scout<<<gridBlock,sizeBlock>>>(workPlace_dev, vectorsResult_dev, nv_dev, numStack, sizeBlock);
		cudaDeviceSynchronize();
		cudaMemcpy(nv, nv_dev, sizeof(float)*3*NUMPERBATCH, cudaMemcpyDeviceToHost);
		cudaMemcpy(workPlace, workPlace_dev, sizeof(char)*numStack*NUMPERBATCH, cudaMemcpyDeviceToHost);

		for(j=0;j<NUMPERBATCH;j++){
			numFacetPenetrated=0;
			photon.position[0]=photonPosition[0];
			photon.position[1]=photonPosition[1];
			photon.position[2]=photonPosition[2];
			photon.direction[0]=nv[3*j];
			photon.direction[1]=nv[3*j+1];
			photon.direction[2]=nv[3*j+2];								//generate the photon direction
			photon.energy=PHOTONENERGY;

			pointer_stride=(long long*)&workPlace[j*numStack];
				
			for(k=0;k<numStack-DEVIDENT;k=k+DEVIDENT){									//the workPlace is tested by the extending it to "long long" (8 bytes) other than the original "char" (1 byte), the purpose of this is to go through workPlace faster, since it is sparse
				if(*pointer_stride){
					for(l=0;l<DEVIDENT;l++){
						if(workPlace[k+l+j*numStack]){
							if(seekIntersectionFacet(&facetPenetrated[numFacetPenetrated], &photon, stack, numStack, k+l)){
								numFacetPenetrated=numFacetPenetrated+1;					//check the whether the facets from scout has been penetrated by the photon, if so, record the facet
							}
						}
					}
				}
				pointer_stride+=1;
			}

			for(l=0;l<numStack-k;l++){
				if(workPlace[k+l+j*numStack]){
					if(seekIntersectionFacet(&facetPenetrated[numFacetPenetrated], &photon, stack, numStack, k+l)){
						numFacetPenetrated=numFacetPenetrated+1;					//check the whether the facets from scout has been penetrated by the photon, if so, record the facet
					}
				}
			}
			if(numFacetPenetrated!=0&&numFacetPenetrated%2==0){
				reorderSmallToLarge(facetPenetrated, &numFacetPenetrated);			//reorder the facets in facetPenetrated
				if(attenuation(&photonBuff, &photon, facetPenetrated,numFacetPenetrated)==1){	//if the function attenuation returns a value, then that means the photon has been successfully detected by the detector
					position[0]=photonBuff.position[0]*MBack[0]+photonBuff.position[1]*MBack[1]+photonBuff.position[2]*MBack[2];
					position[1]=photonBuff.position[0]*MBack[3]+photonBuff.position[1]*MBack[4]+photonBuff.position[2]*MBack[5];
					position[2]=photonBuff.position[0]*MBack[6]+photonBuff.position[1]*MBack[7]+photonBuff.position[2]*MBack[8];

					fwrite(position, sizeof(float), 3, fidw);
				}
			}
		}
		*accumulatedNum=*accumulatedNum+1;
		//printf("batch number %d is finished\n",i);
	}
	randomNumberGeneration<<<NUMPERBATCH/sizeBlock*2,sizeBlock>>>(randomArray_dev, state_dev, *accumulatedNum);
	cudaDeviceSynchronize();
	speedVectorGeneration<<<NUMPERBATCH/sizeBlock,sizeBlock>>>(nv_dev, randomArray_dev, highlimit, lowerlimit, rightlimit, leftlimit, PI, matrixRotation_dev);
	cudaDeviceSynchronize();
	scout<<<gridBlock,sizeBlock>>>(workPlace_dev, vectorsResult_dev, nv_dev, numStack, sizeBlock);
	cudaDeviceSynchronize();
	cudaMemcpy(nv, nv_dev, sizeof(float)*3*NUMPERBATCH, cudaMemcpyDeviceToHost);
	cudaMemcpy(workPlace, workPlace_dev, sizeof(char)*numStack*NUMPERBATCH, cudaMemcpyDeviceToHost);
	for(i=0;i<numResidual;i++){
		numFacetPenetrated=0;
		photon.position[0]=photonPosition[0];
		photon.position[1]=photonPosition[1];
		photon.position[2]=photonPosition[2];
		photon.direction[0]=nv[3*i];
		photon.direction[1]=nv[3*i+1];
		photon.direction[2]=nv[3*i+2];								//generate the photon direction
		photon.energy=PHOTONENERGY;

		pointer_stride=(long long*)&workPlace[i*numStack];
				
		for(k=0;k<numStack-DEVIDENT;k=k+DEVIDENT){									//the workPlace is tested by the extending it to "long long" (8 bytes) other than the original "char" (1 byte), the purpose of this is to go through workPlace faster, since it is sparse
			if(*pointer_stride){
				for(l=0;l<DEVIDENT;l++){
					if(workPlace[k+l+i*numStack]){
						if(seekIntersectionFacet(&facetPenetrated[numFacetPenetrated], &photon, stack, numStack, k+l)){
							numFacetPenetrated=numFacetPenetrated+1;					//check the whether the facets from scout has been penetrated by the photon, if so, record the facet
						}
					}
				}
			}
			pointer_stride+=1;
		}

		for(l=0;l<numStack-k;l++){
			if(workPlace[k+l+i*numStack]){
				if(seekIntersectionFacet(&facetPenetrated[numFacetPenetrated], &photon, stack, numStack, k+l)){
					numFacetPenetrated=numFacetPenetrated+1;					//check the whether the facets from scout has been penetrated by the photon, if so, record the facet
				}
			}
		}
		if(numFacetPenetrated!=0&&numFacetPenetrated%2==0){
			reorderSmallToLarge(facetPenetrated, &numFacetPenetrated);			//reorder the facets in facetPenetrated
			if(attenuation(&photonBuff, &photon, facetPenetrated,numFacetPenetrated)==1){	//if the function attenuation returns a value, then that means the photon has been successfully detected by the detector
				position[0]=photonBuff.position[0]*MBack[0]+photonBuff.position[1]*MBack[1]+photonBuff.position[2]*MBack[2];
				position[1]=photonBuff.position[0]*MBack[3]+photonBuff.position[1]*MBack[4]+photonBuff.position[2]*MBack[5];
				position[2]=photonBuff.position[0]*MBack[6]+photonBuff.position[1]*MBack[7]+photonBuff.position[2]*MBack[8];

				fwrite(position, sizeof(float), 3, fidw);
			}
		}
	}
}

void fastRayTracingBinMode_rotationSlit(int *vectorBin, float halfVector, float pitchBin, float *nv, float *nv_dev, double *randomArray_dev, curandState *state_dev, int numPhoton, float *photonPosition, float *photonPosition_dev, specFacet *stack, specFacet *stack_dev, int numStack, char *workPlace, char *workPlace_dev, vectors *vectorsResult_dev, unsigned long int *accumulatedNum, float *matrixRotation_dev){
	int idxBin, i, j, k, l, numVelo, numBatch, numResidual, sizeBlock=256, numFacetPenetrated;
	float position[3];
	long long* pointer_stride;
	photonType photon, photonBuff;
	facetRecord facetPenetrated[QUEUELENGTH];

	numVelo=numStack/sizeBlock+1;
	dim3 gridBlock(NUMPERBATCH, numVelo);
	numBatch=numPhoton/NUMPERBATCH;
	numResidual=numPhoton%NUMPERBATCH;

	cudaMemcpy(photonPosition_dev, photonPosition, sizeof(float)*3, cudaMemcpyHostToDevice);
	preparation<<<numVelo,sizeBlock>>>(vectorsResult_dev, stack_dev, photonPosition_dev, numStack);
	cudaDeviceSynchronize();

	for(i=0;i<numBatch;i++){
		randomNumberGeneration<<<NUMPERBATCH/sizeBlock*2,sizeBlock>>>(randomArray_dev, state_dev, *accumulatedNum);
		cudaDeviceSynchronize();
		speedVectorGeneration<<<NUMPERBATCH/sizeBlock,sizeBlock>>>(nv_dev, randomArray_dev, highlimit, lowerlimit, rightlimit, leftlimit, PI, matrixRotation_dev);
		cudaDeviceSynchronize();
		scout<<<gridBlock,sizeBlock>>>(workPlace_dev, vectorsResult_dev, nv_dev, numStack, sizeBlock);
		cudaDeviceSynchronize();
		cudaMemcpy(nv, nv_dev, sizeof(float)*3*NUMPERBATCH, cudaMemcpyDeviceToHost);
		cudaMemcpy(workPlace, workPlace_dev, sizeof(char)*numStack*NUMPERBATCH, cudaMemcpyDeviceToHost);

		for(j=0;j<NUMPERBATCH;j++){
			numFacetPenetrated=0;
			photon.position[0]=photonPosition[0];
			photon.position[1]=photonPosition[1];
			photon.position[2]=photonPosition[2];
			photon.direction[0]=nv[3*j];
			photon.direction[1]=nv[3*j+1];
			photon.direction[2]=nv[3*j+2];								//generate the photon direction
			photon.energy=PHOTONENERGY;

			pointer_stride=(long long*)&workPlace[j*numStack];
				
			for(k=0;k<numStack-DEVIDENT;k=k+DEVIDENT){									//the workPlace is tested by the extending it to "long long" (8 bytes) other than the original "char" (1 byte), the purpose of this is to go through workPlace faster, since it is sparse
				if(*pointer_stride){
					for(l=0;l<DEVIDENT;l++){
						if(workPlace[k+l+j*numStack]){
							if(seekIntersectionFacet(&facetPenetrated[numFacetPenetrated], &photon, stack, numStack, k+l)){
								numFacetPenetrated=numFacetPenetrated+1;					//check the whether the facets from scout has been penetrated by the photon, if so, record the facet
							}
						}
					}
				}
				pointer_stride+=1;
			}

			for(l=0;l<numStack-k;l++){
				if(workPlace[k+l+j*numStack]){
					if(seekIntersectionFacet(&facetPenetrated[numFacetPenetrated], &photon, stack, numStack, k+l)){
						numFacetPenetrated=numFacetPenetrated+1;					//check the whether the facets from scout has been penetrated by the photon, if so, record the facet
					}
				}
			}
			if(numFacetPenetrated!=0&&numFacetPenetrated%2==0){
				reorderSmallToLarge(facetPenetrated, &numFacetPenetrated);			//reorder the facets in facetPenetrated
				if(attenuation(&photonBuff, &photon, facetPenetrated,numFacetPenetrated)==1){	//if the function attenuation returns a value, then that means the photon has been successfully detected by the detector
					position[0]=photonBuff.position[0];
					position[1]=photonBuff.position[1];
					position[2]=photonBuff.position[2];

					idxBin=floor((position[0]+halfVector)/pitchBin);
					vectorBin[idxBin]=vectorBin[idxBin]+1;
				}
			}
		}
		*accumulatedNum=*accumulatedNum+1;
		//printf("batch number %d is finished\n",i);
	}
	randomNumberGeneration<<<NUMPERBATCH/sizeBlock*2,sizeBlock>>>(randomArray_dev, state_dev, *accumulatedNum);
	cudaDeviceSynchronize();
	speedVectorGeneration<<<NUMPERBATCH/sizeBlock,sizeBlock>>>(nv_dev, randomArray_dev, highlimit, lowerlimit, rightlimit, leftlimit, PI, matrixRotation_dev);
	cudaDeviceSynchronize();
	scout<<<gridBlock,sizeBlock>>>(workPlace_dev, vectorsResult_dev, nv_dev, numStack, sizeBlock);
	cudaDeviceSynchronize();
	cudaMemcpy(nv, nv_dev, sizeof(float)*3*NUMPERBATCH, cudaMemcpyDeviceToHost);
	cudaMemcpy(workPlace, workPlace_dev, sizeof(char)*numStack*NUMPERBATCH, cudaMemcpyDeviceToHost);
	for(i=0;i<numResidual;i++){
		numFacetPenetrated=0;
		photon.position[0]=photonPosition[0];
		photon.position[1]=photonPosition[1];
		photon.position[2]=photonPosition[2];
		photon.direction[0]=nv[3*i];
		photon.direction[1]=nv[3*i+1];
		photon.direction[2]=nv[3*i+2];								//generate the photon direction
		photon.energy=PHOTONENERGY;

		pointer_stride=(long long*)&workPlace[i*numStack];
				
		for(k=0;k<numStack-DEVIDENT;k=k+DEVIDENT){									//the workPlace is tested by the extending it to "long long" (8 bytes) other than the original "char" (1 byte), the purpose of this is to go through workPlace faster, since it is sparse
			if(*pointer_stride){
				for(l=0;l<DEVIDENT;l++){
					if(workPlace[k+l+i*numStack]){
						if(seekIntersectionFacet(&facetPenetrated[numFacetPenetrated], &photon, stack, numStack, k+l)){
							numFacetPenetrated=numFacetPenetrated+1;					//check the whether the facets from scout has been penetrated by the photon, if so, record the facet
						}
					}
				}
			}
			pointer_stride+=1;
		}

		for(l=0;l<numStack-k;l++){
			if(workPlace[k+l+i*numStack]){
				if(seekIntersectionFacet(&facetPenetrated[numFacetPenetrated], &photon, stack, numStack, k+l)){
					numFacetPenetrated=numFacetPenetrated+1;					//check the whether the facets from scout has been penetrated by the photon, if so, record the facet
				}
			}
		}
		if(numFacetPenetrated!=0&&numFacetPenetrated%2==0){
			reorderSmallToLarge(facetPenetrated, &numFacetPenetrated);			//reorder the facets in facetPenetrated
			if(attenuation(&photonBuff, &photon, facetPenetrated,numFacetPenetrated)==1){	//if the function attenuation returns a value, then that means the photon has been successfully detected by the detector
				position[0]=photonBuff.position[0];
				position[1]=photonBuff.position[1];
				position[2]=photonBuff.position[2];

				idxBin=floor((position[0]+halfVector)/pitchBin);
				vectorBin[idxBin]=vectorBin[idxBin]+1;
			}
		}
	}
	*accumulatedNum=*accumulatedNum+1;
}

void fastRayTracingBinMode_rotationSlit_pinholeComparison(int *detectorBin, float halfVector, float pitchBin, float *nv, float *nv_dev, double *randomArray_dev, curandState *state_dev, int numPhoton, float *photonPosition, float *photonPosition_dev, specFacet *stack, specFacet *stack_dev, int numStack, char *workPlace, char *workPlace_dev, vectors *vectorsResult_dev, unsigned long int *accumulatedNum, float *matrixRotation_dev){
	int idxBinX, idxBinY, i, j, k, l, numVelo, numBatch, numResidual, sizeBlock=256, numFacetPenetrated;
	float position[3];
	long long* pointer_stride;
	photonType photon, photonBuff;
	facetRecord facetPenetrated[QUEUELENGTH];

	numVelo=numStack/sizeBlock+1;
	dim3 gridBlock(NUMPERBATCH, numVelo);
	numBatch=numPhoton/NUMPERBATCH;
	numResidual=numPhoton%NUMPERBATCH;

	cudaMemcpy(photonPosition_dev, photonPosition, sizeof(float)*3, cudaMemcpyHostToDevice);
	preparation<<<numVelo,sizeBlock>>>(vectorsResult_dev, stack_dev, photonPosition_dev, numStack);
	cudaDeviceSynchronize();

	for(i=0;i<numBatch;i++){
		randomNumberGeneration<<<NUMPERBATCH/sizeBlock*2,sizeBlock>>>(randomArray_dev, state_dev, *accumulatedNum);
		cudaDeviceSynchronize();
		speedVectorGeneration<<<NUMPERBATCH/sizeBlock,sizeBlock>>>(nv_dev, randomArray_dev, highlimit, lowerlimit, rightlimit, leftlimit, PI, matrixRotation_dev);
		cudaDeviceSynchronize();
		scout<<<gridBlock,sizeBlock>>>(workPlace_dev, vectorsResult_dev, nv_dev, numStack, sizeBlock);
		cudaDeviceSynchronize();
		cudaMemcpy(nv, nv_dev, sizeof(float)*3*NUMPERBATCH, cudaMemcpyDeviceToHost);
		cudaMemcpy(workPlace, workPlace_dev, sizeof(char)*numStack*NUMPERBATCH, cudaMemcpyDeviceToHost);

		for(j=0;j<NUMPERBATCH;j++){
			numFacetPenetrated=0;
			photon.position[0]=photonPosition[0];
			photon.position[1]=photonPosition[1];
			photon.position[2]=photonPosition[2];
			photon.direction[0]=nv[3*j];
			photon.direction[1]=nv[3*j+1];
			photon.direction[2]=nv[3*j+2];								//generate the photon direction
			photon.energy=PHOTONENERGY;

			pointer_stride=(long long*)&workPlace[j*numStack];
				
			for(k=0;k<numStack-DEVIDENT;k=k+DEVIDENT){									//the workPlace is tested by the extending it to "long long" (8 bytes) other than the original "char" (1 byte), the purpose of this is to go through workPlace faster, since it is sparse
				if(*pointer_stride){
					for(l=0;l<DEVIDENT;l++){
						if(workPlace[k+l+j*numStack]){
							if(seekIntersectionFacet(&facetPenetrated[numFacetPenetrated], &photon, stack, numStack, k+l)){
								numFacetPenetrated=numFacetPenetrated+1;					//check the whether the facets from scout has been penetrated by the photon, if so, record the facet
							}
						}
					}
				}
				pointer_stride+=1;
			}

			for(l=0;l<numStack-k;l++){
				if(workPlace[k+l+j*numStack]){
					if(seekIntersectionFacet(&facetPenetrated[numFacetPenetrated], &photon, stack, numStack, k+l)){
						numFacetPenetrated=numFacetPenetrated+1;					//check the whether the facets from scout has been penetrated by the photon, if so, record the facet
					}
				}
			}
			if(numFacetPenetrated!=0&&numFacetPenetrated%2==0){
				reorderSmallToLarge(facetPenetrated, &numFacetPenetrated);			//reorder the facets in facetPenetrated
				if(attenuation(&photonBuff, &photon, facetPenetrated,numFacetPenetrated)==1){	//if the function attenuation returns a value, then that means the photon has been successfully detected by the detector
					position[0]=photonBuff.position[0];
					position[1]=photonBuff.position[1];
					position[2]=photonBuff.position[2];

					idxBinX=floor((position[0]+halfVector)/pitchBin);
					idxBinY=floor((position[1]+halfVector)/pitchBin);
					detectorBin[idxBinX*NUMVECTORBIN+idxBinY]=detectorBin[idxBinX*NUMVECTORBIN+idxBinY]+1;
				}
			}
		}
		*accumulatedNum=*accumulatedNum+1;
		//printf("batch number %d is finished\n",i);
	}
	randomNumberGeneration<<<NUMPERBATCH/sizeBlock*2,sizeBlock>>>(randomArray_dev, state_dev, *accumulatedNum);
	cudaDeviceSynchronize();
	speedVectorGeneration<<<NUMPERBATCH/sizeBlock,sizeBlock>>>(nv_dev, randomArray_dev, highlimit, lowerlimit, rightlimit, leftlimit, PI, matrixRotation_dev);
	cudaDeviceSynchronize();
	scout<<<gridBlock,sizeBlock>>>(workPlace_dev, vectorsResult_dev, nv_dev, numStack, sizeBlock);
	cudaDeviceSynchronize();
	cudaMemcpy(nv, nv_dev, sizeof(float)*3*NUMPERBATCH, cudaMemcpyDeviceToHost);
	cudaMemcpy(workPlace, workPlace_dev, sizeof(char)*numStack*NUMPERBATCH, cudaMemcpyDeviceToHost);
	for(i=0;i<numResidual;i++){
		numFacetPenetrated=0;
		photon.position[0]=photonPosition[0];
		photon.position[1]=photonPosition[1];
		photon.position[2]=photonPosition[2];
		photon.direction[0]=nv[3*i];
		photon.direction[1]=nv[3*i+1];
		photon.direction[2]=nv[3*i+2];								//generate the photon direction
		photon.energy=PHOTONENERGY;

		pointer_stride=(long long*)&workPlace[i*numStack];
				
		for(k=0;k<numStack-DEVIDENT;k=k+DEVIDENT){									//the workPlace is tested by the extending it to "long long" (8 bytes) other than the original "char" (1 byte), the purpose of this is to go through workPlace faster, since it is sparse
			if(*pointer_stride){
				for(l=0;l<DEVIDENT;l++){
					if(workPlace[k+l+i*numStack]){
						if(seekIntersectionFacet(&facetPenetrated[numFacetPenetrated], &photon, stack, numStack, k+l)){
							numFacetPenetrated=numFacetPenetrated+1;					//check the whether the facets from scout has been penetrated by the photon, if so, record the facet
						}
					}
				}
			}
			pointer_stride+=1;
		}

		for(l=0;l<numStack-k;l++){
			if(workPlace[k+l+i*numStack]){
				if(seekIntersectionFacet(&facetPenetrated[numFacetPenetrated], &photon, stack, numStack, k+l)){
					numFacetPenetrated=numFacetPenetrated+1;					//check the whether the facets from scout has been penetrated by the photon, if so, record the facet
				}
			}
		}
		if(numFacetPenetrated!=0&&numFacetPenetrated%2==0){
			reorderSmallToLarge(facetPenetrated, &numFacetPenetrated);			//reorder the facets in facetPenetrated
			if(attenuation(&photonBuff, &photon, facetPenetrated,numFacetPenetrated)==1){	//if the function attenuation returns a value, then that means the photon has been successfully detected by the detector
				position[0]=photonBuff.position[0];
				position[1]=photonBuff.position[1];
				position[2]=photonBuff.position[2];

				idxBinX=floor((position[0]+halfVector)/pitchBin);
				idxBinY=floor((position[1]+halfVector)/pitchBin);
				detectorBin[idxBinX*NUMVECTORBIN+idxBinY]=detectorBin[idxBinX*NUMVECTORBIN+idxBinY]+1;
			}
		}
	}
	*accumulatedNum=*accumulatedNum+1;
}

void fastRayTracingPointSpreadFunction(FILE *fid, int *numMax, float *solidAnglePro, float *patchArray, int numX, int numY, int idxX, int idxY, float pitch, float *nv, float *nv_dev, float *photonPosition, float *photonPosition_dev, specFacet *stack, specFacet *stack_dev, int numStack, char *workPlace, char *workPlace_dev, vectors *vectorsResult_dev, char flag){
	int i, j, k, l, numPhoton=numX*numY, numVelo, numBatch, numResidual, sizeBlock=256, numFacetPenetrated, badBuffer[BADBUFFERSIZE], badBufferCount, idx;
	float buffer=0;
	long long* pointer_stride;
	photonType photon, photonBuff;
	facetRecord facetPenetrated[QUEUELENGTH];

	numVelo=numStack/sizeBlock+1;
	dim3 gridBlock(NUMPERBATCH, numVelo);
	numBatch=numPhoton/NUMPERBATCH;
	numResidual=numPhoton%NUMPERBATCH;

	cudaMemcpy(photonPosition_dev, photonPosition, sizeof(float)*3, cudaMemcpyHostToDevice);
	preparation<<<numVelo,sizeBlock>>>(vectorsResult_dev, stack_dev, photonPosition_dev, numStack);
	cudaDeviceSynchronize();

	badBufferCount=0;
	
	if(flag=='p'){
		fwrite(&idxX, sizeof(int), 1, fid);
		fwrite(&idxY, sizeof(int), 1, fid);
	}else if(flag=='v'){
		fwrite(&idxX, sizeof(int), 1, fid);
	}else if(flag=='h'){
		fwrite(&idxX, sizeof(int), 1, fid);
	}else{
		printf("Invalid data PSF type.\n");
		exit(0);
	}

	for(i=0;i<numBatch;i++){
		scout<<<gridBlock,sizeBlock>>>(workPlace_dev, vectorsResult_dev, nv_dev, numStack, sizeBlock);
		cudaDeviceSynchronize();
		cudaMemcpy(workPlace, workPlace_dev, sizeof(char)*numStack*NUMPERBATCH, cudaMemcpyDeviceToHost);

		for(j=0;j<NUMPERBATCH;j++){
			numFacetPenetrated=0;
			photon.position[0]=photonPosition[0];
			photon.position[1]=photonPosition[1];
			photon.position[2]=photonPosition[2];
			photon.direction[0]=nv[3*j];
			photon.direction[1]=nv[3*j+1];
			photon.direction[2]=nv[3*j+2];								//generate the photon direction
			photon.energy=PHOTONENERGY;

			pointer_stride=(long long*)&workPlace[j*numStack];
				
			for(k=0;k<numStack-DEVIDENT;k=k+DEVIDENT){									//the workPlace is tested by the extending it to "long long" (8 bytes) other than the original "char" (1 byte), the purpose of this is to go through workPlace faster, since it is sparse
				if(*pointer_stride){
					for(l=0;l<DEVIDENT;l++){
						if(workPlace[j*numStack+k+l]){
							if(seekIntersectionFacet(&facetPenetrated[numFacetPenetrated], &photon, stack, numStack, k+l)){
								numFacetPenetrated=numFacetPenetrated+1;					//check the whether the facets from scout has been penetrated by the photon, if so, record the facet
							}
						}
					}
				}
				pointer_stride+=1;
			}

			for(l=0;l<numStack-k;l++){
				if(workPlace[j*numStack+k+l]){
					if(seekIntersectionFacet(&facetPenetrated[numFacetPenetrated], &photon, stack, numStack, k+l)){
						numFacetPenetrated=numFacetPenetrated+1;					//check the whether the facets from scout has been penetrated by the photon, if so, record the facet
					}
				}
			}
			reorderSmallToLarge(facetPenetrated, &numFacetPenetrated);			//reorder the facets in facetPenetrated
			if(numFacetPenetrated%2!=0){
				badBuffer[badBufferCount]=NUMPERBATCH*i+j;
				badBufferCount++;
			}else{
				patchArray[NUMPERBATCH*i+j+numX+1]=solidAnglePro[NUMPERBATCH*i+j]*attenuationPointSpreadFunction(&photonBuff, &photon, facetPenetrated,numFacetPenetrated);
			}
		}
		//printf("batch number %d is finished\n",i);
	}
	scout<<<gridBlock,sizeBlock>>>(workPlace_dev, vectorsResult_dev, nv_dev, numStack, sizeBlock);
	cudaDeviceSynchronize();
	cudaMemcpy(workPlace, workPlace_dev, sizeof(char)*numStack*NUMPERBATCH, cudaMemcpyDeviceToHost);
	for(i=0;i<numResidual;i++){
		numFacetPenetrated=0;
		photon.position[0]=photonPosition[0];
		photon.position[1]=photonPosition[1];
		photon.position[2]=photonPosition[2];
		photon.direction[0]=nv[3*i];
		photon.direction[1]=nv[3*i+1];
		photon.direction[2]=nv[3*i+2];								//generate the photon direction
		photon.energy=PHOTONENERGY;

		pointer_stride=(long long*)&workPlace[i*numStack];
				
		for(k=0;k<numStack-DEVIDENT;k=k+DEVIDENT){									//the workPlace is tested by the extending it to "long long" (8 bytes) other than the original "char" (1 byte), the purpose of this is to go through workPlace faster, since it is sparse
			if(*pointer_stride){
				for(l=0;l<DEVIDENT;l++){
					if(workPlace[k+l+i*numStack]){
						if(seekIntersectionFacet(&facetPenetrated[numFacetPenetrated], &photon, stack, numStack, k+l)){
							numFacetPenetrated=numFacetPenetrated+1;					//check the whether the facets from scout has been penetrated by the photon, if so, record the facet
						}
					}
				}
			}
			pointer_stride+=1;
		}

		for(l=0;l<numStack-k;l++){
			if(workPlace[k+l+i*numStack]){
				if(seekIntersectionFacet(&facetPenetrated[numFacetPenetrated], &photon, stack, numStack, k+l)){
					numFacetPenetrated=numFacetPenetrated+1;					//check the whether the facets from scout has been penetrated by the photon, if so, record the facet
				}
			}
		}
		reorderSmallToLarge(facetPenetrated, &numFacetPenetrated);			//reorder the facets in facetPenetrated
		if(numFacetPenetrated%2!=0){
			badBuffer[badBufferCount]=NUMPERBATCH*numBatch+i;
			badBufferCount++;
		}else{
			patchArray[NUMPERBATCH*numBatch+i+numX+1]=solidAnglePro[NUMPERBATCH*numBatch+i]*attenuationPointSpreadFunction(&photonBuff, &photon, facetPenetrated,numFacetPenetrated);
		}
	}
	for(i=0;i<badBufferCount;i++){
		idx=badBuffer[i];
		patchArray[idx]=(patchArray[idx-numX-1]+patchArray[idx-numX]+patchArray[idx-numX+1]+patchArray[idx-1]+patchArray[idx+1]+patchArray[idx+numX-1]+patchArray[idx+numX]+patchArray[idx+numX+1])/8;
	}
	if(badBufferCount>*numMax){
		*numMax=badBufferCount;
	}
	if(flag=='p'){
		fwrite(&patchArray[numX+1], sizeof(float), numPhoton, fid);
	}else if(flag=='v'||flag=='h'){
		for(i=0;i<numX;i++){
			buffer=0;
			for(j=0;j<numY;j++){
				buffer=buffer+patchArray[numX+1+i*numY+j];
			}
			buffer=buffer*1000/32;
			fwrite(&buffer, sizeof(float), 1, fid);
		}
	}else{
		printf("Invalid data PSF type.\n");
		exit(0);
	}
}

__global__ void forwardProjection(float *image_dev, int idxProjection, float magnitude, int *cor_dev, float *psfPartial_dev, int numLocalX, int numLocalY, int detectorNumWidth, int detectorNumHeight, int detectorNum){
	int idx=blockDim.x*blockIdx.x+threadIdx.x, idxLocalX, idxLocalY, idxLocal, detectorIdxWidth, detectorIdxHeight, detectorIdx;
	if(idx<numLocalX*numLocalY){
		idxLocalX=idx/numLocalY;
		idxLocalY=idx%numLocalY;
		idxLocal=idxLocalX*numLocalY+idxLocalY;

		detectorIdxWidth=idxLocalX-numLocalX/2+cor_dev[0]+detectorNumWidth/2;
		detectorIdxHeight=idxLocalY-numLocalY/2+cor_dev[1]+detectorNumHeight/2;
		detectorIdx=detectorNumHeight*detectorIdxWidth+detectorIdxHeight;

		if(detectorIdx>0&&detectorIdx<detectorNum){
			image_dev[detectorIdx+idxProjection*detectorNum]=image_dev[detectorIdx+idxProjection*detectorNum]+magnitude*psfPartial_dev[idxLocal];
		}
	}
}

__global__ void forwardProjectionVerticalSlit(float *image_dev, int idxProjection, float magnitude, int *cor_dev, float *psfPartial_dev, int numLocalX, int detectorNumWidth, int detectorNum){
	int idx=blockDim.x*blockIdx.x+threadIdx.x, detectorIdx;
	if(idx<numLocalX){

		detectorIdx=idx-numLocalX/2+cor_dev[0]+detectorNumWidth/2;

		if(detectorIdx>0&&detectorIdx<detectorNum){
			image_dev[detectorIdx+idxProjection*detectorNum]=image_dev[detectorIdx+idxProjection*detectorNum]+magnitude*psfPartial_dev[idx];
		}
	}
}

__global__ void forwardProjectionHorizontalSlit(float *image_dev, int idxProjection, float magnitude, int *cor_dev, float *psfPartial_dev, int numLocalY, int detectorNumHeight, int detectorNum){
	int idx=blockDim.x*blockIdx.x+threadIdx.x, detectorIdx;
	if(idx<numLocalY){

		detectorIdx=idx-numLocalY/2+cor_dev[0]+detectorNumHeight/2;

		if(detectorIdx>0&&detectorIdx<detectorNum){
			image_dev[detectorIdx+idxProjection*detectorNum]=image_dev[detectorIdx+idxProjection*detectorNum]+magnitude*psfPartial_dev[idx];
		}
	}
}

void forwardProjectionAllPoints(float *image, float *image_dev, int *cor, int *cor_dev, float *psfPartial, float *psfPartial_dev, int numProjections, float *object, int objectPointNum, int numLocalX, int numLocalY, int detectorNumWidth, int detectorNumHeight, int detectorNum, char flag){
	char bufferPSFName[50];
	int idxProjection, i, numBlock;
	FILE *fidPSF;

	if(flag=='p'){
		numBlock=numLocalX*numLocalY/32+1;

		for(idxProjection=0;idxProjection<numProjections;idxProjection++){
			sprintf(bufferPSFName, "G:\\research\\data\\PSF%d.bin", idxProjection);
			fidPSF=fopen(bufferPSFName,"rb");
			for(i=0;i<objectPointNum;i++){
				fread(cor, sizeof(int), 2, fidPSF);
				fread(psfPartial, sizeof(float), numLocalX*numLocalY, fidPSF);
				cudaMemcpy(cor_dev, cor, sizeof(int)*2, cudaMemcpyHostToDevice);
				cudaMemcpy(psfPartial_dev, psfPartial, sizeof(float)*numLocalX*numLocalY, cudaMemcpyHostToDevice);
				forwardProjection<<<numBlock,32>>>(image_dev, idxProjection, object[i], cor_dev, psfPartial_dev, numLocalX, numLocalY, detectorNumWidth, detectorNumHeight, detectorNum);
				cudaDeviceSynchronize();
			}
			fclose(fidPSF);
			printf("Projection %d is finished\n", idxProjection);
		}
		cudaMemcpy(image, image_dev, sizeof(float)*detectorNum*numProjections, cudaMemcpyDeviceToHost);
	}else if(flag=='v'){
		numBlock=numLocalX/32+1;

		for(idxProjection=0;idxProjection<numProjections;idxProjection++){
			sprintf(bufferPSFName, "G:\\research\\data\\vslitPSF%d.bin", idxProjection);
			fidPSF=fopen(bufferPSFName,"rb");
			for(i=0;i<objectPointNum;i++){
				fread(cor, sizeof(int), 1, fidPSF);
				fread(psfPartial, sizeof(float), numLocalX, fidPSF);
				cudaMemcpy(cor_dev, cor, sizeof(int)*1, cudaMemcpyHostToDevice);
				cudaMemcpy(psfPartial_dev, psfPartial, sizeof(float)*numLocalX, cudaMemcpyHostToDevice);
				forwardProjectionVerticalSlit<<<numBlock,32>>>(image_dev, idxProjection, object[i], cor_dev, psfPartial_dev, numLocalX, detectorNumWidth, detectorNumWidth);
				cudaDeviceSynchronize();
			}
			fclose(fidPSF);
			printf("Projection %d is finished\n", idxProjection);
		}
		cudaMemcpy(image, image_dev, sizeof(float)*detectorNumWidth*numProjections, cudaMemcpyDeviceToHost);
	}else if(flag=='h'){
		numBlock=numLocalY/32+1;

		for(idxProjection=0;idxProjection<numProjections;idxProjection++){
			sprintf(bufferPSFName, "G:\\research\\data\\hslitPSF%d.bin", idxProjection);
			fidPSF=fopen(bufferPSFName,"rb");
			for(i=0;i<objectPointNum;i++){
				fread(cor, sizeof(int), 1, fidPSF);
				fread(psfPartial, sizeof(float), numLocalX, fidPSF);
				cudaMemcpy(cor_dev, cor, sizeof(int)*1, cudaMemcpyHostToDevice);
				cudaMemcpy(psfPartial_dev, psfPartial, sizeof(float)*numLocalX, cudaMemcpyHostToDevice);
				forwardProjectionHorizontalSlit<<<numBlock,32>>>(image_dev, idxProjection, object[i], cor_dev, psfPartial_dev, numLocalX, detectorNumHeight, detectorNumHeight);
				cudaDeviceSynchronize();
			}
			fclose(fidPSF);
			printf("Projection %d is finished\n", idxProjection);
		}
		cudaMemcpy(image, image_dev, sizeof(float)*detectorNumHeight*numProjections, cudaMemcpyDeviceToHost);
	}
}

__global__ void imageRatioCalculation(float *imageRatio_dev, float *image_dev, float *imageReal_dev, int numTotal){
	int idx=blockDim.x*blockIdx.x+threadIdx.x;
	if(idx<numTotal){
		imageRatio_dev[idx]=0;
		if(image_dev[idx]){
			imageRatio_dev[idx]=imageReal_dev[idx]/image_dev[idx];
		}
	}
}

__global__ void adjustCalculation(float *adjust_dev, float *imageRatio_dev, float *psfTotal_dev, int numTotal){
	int idx=blockDim.x*blockIdx.x+threadIdx.x;
	if(idx<numTotal){
		adjust_dev[idx]=imageRatio_dev[idx]*psfTotal_dev[idx];
	}
}