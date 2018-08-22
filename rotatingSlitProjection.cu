#include"header_simulation_engine.h"

#define NUMPROJECTIONS 90
#define QUEUELENGTH 10							//the size of maximum number of facelets a photon traveling in a straight line can pass
#define DEVIDENT 8								//The striding step in search the result matrix workplace, in order to search faster					
#define NUMPERBATCH 1024
#define NUMVECTORBIN 600
#define PITCHBIN 0.1
#define NUMANGLE 1
#define PHANTOMSIZEX 50
#define PHANTOMSIZEY 50
#define PHANTOMSIZEZ 50
#define PHANTOMPITCHX 0.5
#define PHANTOMPITCHY 0.5
#define PHANTOMPITCHZ 0.5

int main(){
	char *workPlace, *workPlace_dev, buffer[150];
	int idxAngle, idxProjection, value, numFacelet1, numFacelet2, numStack=0, i, j, k, cors[2], numMax=0, *vectorBin;
	float theta, phi, valueFloat, M_source[9], M_velocity[9], *M_velocity_dev, *nv, *nv_dev, photonPosition[3], tempPosition[3], centerOfAperturePosition[3], *photonPosition_dev, angleAlpha=0, distOriginToCA=51.5, rotationSpec[2], EdgePointFinal[3]={0,1,0}, *phantom;
	double *randomArray_dev;
	unsigned long int accumulatedNum=0;
	specFacet* part1,* part2, *stack, *stack_dev;
	vectors *vectorsResult_dev;
	FILE *fid, *fidPhantom;
	std::default_random_engine generator;
	curandState *state_dev;

	readPart(&part1, &numFacelet1, "G:\\research\\slitSPECT\\slitAperture_assemble50X50_spacing_0.3 - slitAperture50X50_0.3spacing-1.STL",'t');
	readPart(&part2, &numFacelet2, "G:\\research\\slitSPECT\\slitAperture_assemble50X50_spacing_0.3 - detector50X50-1.STL",'d');                
	catenate(&stack, &numStack, &part1, & numFacelet1);
	catenate(&stack, &numStack, &part2, & numFacelet2);

	cudaMalloc((void**)&stack_dev,sizeof(specFacet)*numStack);
	cudaMemcpy(stack_dev, stack, sizeof(specFacet)*numStack, cudaMemcpyHostToDevice);		
	nv=(float *)malloc(sizeof(float)*NUMPERBATCH*3);
	cudaMalloc((void**)&nv_dev, sizeof(float)*NUMPERBATCH*3);
	cudaMalloc((void**)&photonPosition_dev, sizeof(float)*3);
	cudaMalloc((void**)&M_velocity_dev, sizeof(float)*9);
	workPlace=(char *)malloc(sizeof(char)*numStack*NUMPERBATCH);
	cudaMalloc((void**)&workPlace_dev,sizeof(char)*numStack*NUMPERBATCH);
	cudaMalloc((void**)&vectorsResult_dev, sizeof(vectors)*numStack);
	vectorBin=(int *)malloc(sizeof(int)*NUMVECTORBIN);
	cudaMalloc((void**)&randomArray_dev, sizeof(float)*NUMPERBATCH*2);
	cudaMalloc((void**)&state_dev, sizeof(curandState)*NUMPERBATCH*2);
	phantom=(float *)malloc(sizeof(float)*PHANTOMSIZEX*PHANTOMSIZEY*PHANTOMSIZEZ);

	fidPhantom=fopen("G:\\research\\data\\phantom.bin","rb");
	fread(phantom, sizeof(float), PHANTOMSIZEX*PHANTOMSIZEY*PHANTOMSIZEZ, fidPhantom);
	fclose(fidPhantom);
	std::clock_t start, end;
	start=std::clock();
	//
	int test;
	//
	for(idxAngle=0;idxAngle<NUMANGLE;idxAngle++){

		theta=PI/2;
		phi=0;

		sprintf(buffer, "G:\\research\\data\\rotatingSlitProjection_resolution_phantomX1.8_90_600X10_0.9A%d_view_45_45_Poisson_withoutAAS.bin", idxAngle);
		fid=fopen(buffer,"wb");

		for(idxProjection=0;idxProjection<NUMPROJECTIONS;idxProjection++){		
			centerOfAperturePosition[0]=distOriginToCA*sin(theta)*cos(phi);//30.75;
			centerOfAperturePosition[1]=distOriginToCA*sin(theta)*sin(phi);//30.75;
			centerOfAperturePosition[2]=distOriginToCA*cos(theta);//43.487067043;

			angleAlpha=PI*idxProjection/NUMPROJECTIONS;

			inverseMatrixGeneration(M_source, angleAlpha, centerOfAperturePosition);

			centerOfAperturePosition[0]=0;
			centerOfAperturePosition[1]=0;
			centerOfAperturePosition[2]=distOriginToCA;

			for(i=0;i<NUMVECTORBIN;i++){
				vectorBin[i]=0;
			}
			/*
			int countTotal=0;
			for (i = 0; i < PHANTOMSIZEX; i++){
				for (j = 0; j < PHANTOMSIZEY; j++){
					for (k = 0; k < PHANTOMSIZEZ; k++){
						countTotal += phantom[i*PHANTOMSIZEY*PHANTOMSIZEZ + j*PHANTOMSIZEZ + k];
					}
				}
			}
			*/
			for(i=0;i<PHANTOMSIZEX;i++){
				for(j=0;j<PHANTOMSIZEY;j++){
					for(k=0;k<PHANTOMSIZEZ;k++){
						valueFloat=phantom[i*PHANTOMSIZEY*PHANTOMSIZEZ+j*PHANTOMSIZEZ+k]*180;//0.6;
						if(valueFloat){
							std::poisson_distribution<int> distributionPo(valueFloat);
							value=distributionPo(generator);
						}else{
							value=0;
						}
						if(value){						

							tempPosition[0]=(i-PHANTOMSIZEX/2.0)*PHANTOMPITCHX;
							tempPosition[1]=(j-PHANTOMSIZEY/2.0)*PHANTOMPITCHY;
							tempPosition[2]=(k-PHANTOMSIZEZ/2.0)*PHANTOMPITCHZ;
							photonPosition[0]=tempPosition[0]*M_source[0]+tempPosition[1]*M_source[1]+tempPosition[2]*M_source[2];
							photonPosition[1]=tempPosition[0]*M_source[3]+tempPosition[1]*M_source[4]+tempPosition[2]*M_source[5];
							photonPosition[2]=tempPosition[0]*M_source[6]+tempPosition[1]*M_source[7]+tempPosition[2]*M_source[8];
							cudaMemcpy(photonPosition_dev, photonPosition, sizeof(float)*3, cudaMemcpyHostToDevice);

							rotationAngleGeneration(rotationSpec, EdgePointFinal, photonPosition, centerOfAperturePosition);
							velocityRotationMatrixGeneration(M_velocity, rotationSpec, photonPosition, centerOfAperturePosition, 1);
							cudaMemcpy(M_velocity_dev, M_velocity, sizeof(float)*9, cudaMemcpyHostToDevice);
					
							fastRayTracingBinMode_rotationSlit(vectorBin, NUMVECTORBIN*PITCHBIN/2.0, PITCHBIN, nv, nv_dev, randomArray_dev, state_dev, value, photonPosition, photonPosition_dev, stack, stack_dev, numStack, workPlace, workPlace_dev, vectorsResult_dev, &accumulatedNum, M_velocity_dev);
						
						}
					}
				}
				
				printf("%d is complete\n", i);
				printf("maximum error is %d\n", numMax);
			}
			fwrite(vectorBin, sizeof(int), NUMVECTORBIN, fid);
			printf("projection %d is finished!\n", idxProjection);
		}
		printf("Angle %d is finished!\n", idxAngle);
		fclose(fid);
	}

	end=std::clock();
	printf("time is %d\n", end-start);
	getchar();
	return 0;
}