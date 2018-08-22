#include"header_simulation_engine.h"

#define NUMPROJECTIONS 1
#define QUEUELENGTH 10							//the size of maximum number of facelets a photon traveling in a straight line can pass
#define DEVIDENT 8								//The striding step in search the result matrix workplace, in order to search faster					
#define NUMPERBATCH 1024
#define NUMVECTORBIN 600
#define PITCHBIN 0.1
#define PHANTOMSIZEX 50
#define PHANTOMSIZEY 50
#define PHANTOMSIZEZ 50
#define PHANTOMPITCHX 0.5
#define PHANTOMPITCHY 0.5
#define PHANTOMPITCHZ 0.5

int main(){
	char *workPlace, *workPlace_dev, buffer[150];
	int idxProjection, numFacelet1, numFacelet2, value, numStack=0, i, j, k, cors[2], numMax=0, *detectorBin;
	float theta, phi, valueFloat, M_source[9], M_velocity[9], *M_velocity_dev, *nv, *nv_dev, photonPosition[3], tempPosition[3], centerOfAperturePosition[3], *photonPosition_dev, angleAlpha=0, distOriginToCA=51.5, rotationSpec[2], EdgePointFinal[3]={0,1,0}, *phantom;
	double *randomArray_dev;
	unsigned long int accumulatedNum=0;
	specFacet* part1,* part2, *stack, *stack_dev;
	vectors *vectorsResult_dev;
	FILE *fid, *fidPhantom;
	std::default_random_engine generator;
	curandState *state_dev;

	readPart(&part1, &numFacelet1, "G:\\research\\slitSPECT\\pinholeAperture_assemble50X50_spacing_0.3 - pinholeAperture50X50_0.3dia-1.STL",'t');
	readPart(&part2, &numFacelet2, "G:\\research\\slitSPECT\\pinholeAperture_assemble50X50_spacing_0.3 - detector50X50-1.STL",'d');
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
	detectorBin=(int *)malloc(sizeof(int)*NUMVECTORBIN*NUMVECTORBIN);
	cudaMalloc((void**)&randomArray_dev, sizeof(float)*NUMPERBATCH*2);
	cudaMalloc((void**)&state_dev, sizeof(curandState)*NUMPERBATCH*2);
	phantom=(float *)malloc(sizeof(float)*PHANTOMSIZEX*PHANTOMSIZEY*PHANTOMSIZEZ);

	std::clock_t start, end;
	start=std::clock();

	theta=PI/4;
	phi=PI/4;

	fid=fopen("G:\\research\\data\\rotatingSlitProjection_pinholeComparisonX0.3X3_90_600_PinholeX1_A1.0_view_45_45.bin","wb");
	fidPhantom=fopen("G:\\research\\data\\phantom.bin","rb");
	fread(phantom, sizeof(float), PHANTOMSIZEX*PHANTOMSIZEY*PHANTOMSIZEZ, fidPhantom);
	fclose(fidPhantom);

	for(idxProjection=0;idxProjection<NUMPROJECTIONS;idxProjection++){		
		centerOfAperturePosition[0]=distOriginToCA*sin(theta)*cos(phi);//30.75;
		centerOfAperturePosition[1]=distOriginToCA*sin(theta)*sin(phi);//30.75;
		centerOfAperturePosition[2]=distOriginToCA*cos(theta);//43.487067043;

		angleAlpha=PI*idxProjection/NUMPROJECTIONS;

		inverseMatrixGeneration(M_source, angleAlpha, centerOfAperturePosition);

		centerOfAperturePosition[0]=0;
		centerOfAperturePosition[1]=0;
		centerOfAperturePosition[2]=distOriginToCA;

		for(i=0;i<NUMVECTORBIN*NUMVECTORBIN;i++){
			detectorBin[i]=0;
		}
		//
		int countTotal = 0;
		for (i = 0; i < PHANTOMSIZEX; i++){
			for (j = 0; j < PHANTOMSIZEY; j++){
				for (k = 0; k < PHANTOMSIZEZ; k++){
					countTotal += phantom[i*PHANTOMSIZEY*PHANTOMSIZEZ + j*PHANTOMSIZEZ + k];
				}
			}
		}
		//
		for(i=0;i<PHANTOMSIZEX;i++){
			for(j=0;j<PHANTOMSIZEY;j++){
				for(k=0;k<PHANTOMSIZEZ;k++){
					valueFloat=phantom[i*PHANTOMSIZEY*PHANTOMSIZEZ+j*PHANTOMSIZEZ+k]*0.905499788919*100;
					if(valueFloat){
						std::poisson_distribution<int> distributionPo(valueFloat);
						value=distributionPo(generator);
					}else{
						value=0;
					}
					if(value){
						tempPosition[0]=i*PHANTOMPITCHX-PHANTOMSIZEX*PHANTOMPITCHX/2.0;
						tempPosition[1]=j*PHANTOMPITCHX-PHANTOMSIZEY*PHANTOMPITCHX/2.0;
						tempPosition[2]=k*PHANTOMPITCHX-PHANTOMSIZEZ*PHANTOMPITCHX/2.0;
						photonPosition[0]=tempPosition[0]*M_source[0]+tempPosition[1]*M_source[1]+tempPosition[2]*M_source[2];
						photonPosition[1]=tempPosition[0]*M_source[3]+tempPosition[1]*M_source[4]+tempPosition[2]*M_source[5];
						photonPosition[2]=tempPosition[0]*M_source[6]+tempPosition[1]*M_source[7]+tempPosition[2]*M_source[8];
						cudaMemcpy(photonPosition_dev, photonPosition, sizeof(float)*3, cudaMemcpyHostToDevice);

						rotationAngleGeneration(rotationSpec, EdgePointFinal, photonPosition, centerOfAperturePosition);
						velocityRotationMatrixGeneration(M_velocity, rotationSpec, photonPosition, centerOfAperturePosition, 0);
						cudaMemcpy(M_velocity_dev, M_velocity, sizeof(float)*9, cudaMemcpyHostToDevice);
					
						fastRayTracingBinMode_rotationSlit_pinholeComparison(detectorBin, NUMVECTORBIN*PITCHBIN/2, PITCHBIN, nv, nv_dev, randomArray_dev, state_dev, value, photonPosition, photonPosition_dev, stack, stack_dev, numStack, workPlace, workPlace_dev, vectorsResult_dev, &accumulatedNum, M_velocity_dev);
					}
				}
			}
			printf("%d is complete\n", i);
			printf("maximum error is %d\n", numMax);
		}
		fwrite(detectorBin, sizeof(int), NUMVECTORBIN*NUMVECTORBIN, fid);
		printf("projection %d is finished!\n", idxProjection);
	}
	fclose(fid);

	end=std::clock();
	fclose(fid);	
	
	printf("time is %d\n", end-start);
	getchar();
	return 0;
}