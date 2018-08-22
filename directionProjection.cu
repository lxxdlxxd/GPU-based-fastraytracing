#include"header_simulation_engine.h"

int main(){
	int numProjection, numFacelet1, numFacelet2, numStack=0, i, j, k;
	specFacet* part1,* part2, *stack, *stackRotatedToVector, *stackRotatedToVector_dev;

	readPart(&part1, &numFacelet1, "G:\\research\\slitSPECT\\pinholeAperture_assemble_brain_projection_1.5mm_2.25mm - pinholeAperture-1.STL",'t');
	readPart(&part2, &numFacelet2, "G:\\research\\slitSPECT\\pinholeAperture_assemble_brain_projection_1.5mm_2.25mm - detector-1.STL",'d');
	catenate(&stack, &numStack, &part1, & numFacelet1);
	catenate(&stack, &numStack, &part2, & numFacelet2);

	stackRotatedToVector=(specFacet *)malloc(numStack*sizeof(specFacet));

	float M[9], *nv_dev, photonPosition[3], *photonPosition_dev, EdgePointFinal[3], EdgePointOriginal[3], rotationSpec[2];
	float MBack[9], *nv, *M_dev;
	float centerOfAperturePosition[3], alpha, directionVector[15][3];
	double *randomArray_dev;
	char fileName[50];
	int numPhoton=0;
	short *phantom, value;
	unsigned long int accumulatedNum=0;
	char *workPlace, *workPlace_dev;
	vectors *vectorsResult_dev;
	curandState *state_dev;
	FILE *fid, *fidPhantom;
	std::default_random_engine generator;

	nv=(float *)malloc(sizeof(float)*NUMPERBATCH*3);
	cudaMalloc((void**)&nv_dev, sizeof(float)*NUMPERBATCH*3);
	cudaMalloc((void**)&state_dev, sizeof(curandState)*NUMPERBATCH*2);
	cudaMalloc((void**)&randomArray_dev, sizeof(float)*NUMPERBATCH*2);
	cudaMalloc((void**)&photonPosition_dev, sizeof(float)*3);
	cudaMalloc((void**)&stackRotatedToVector_dev, sizeof(specFacet)*numStack);
	workPlace=(char *)malloc(sizeof(char)*numStack*NUMPERBATCH);
	cudaMalloc((void**)&workPlace_dev,sizeof(char)*numStack*NUMPERBATCH);
	cudaMalloc((void**)&vectorsResult_dev, sizeof(vectors)*numStack);
	cudaMalloc((void**)&M_dev, sizeof(float)*9);
	
	phantom=(short *)malloc(sizeof(short)*181*217*181);
	fidPhantom=fopen("G:\\research\\numericalPhantom\\phantom_1.0mm_normal_gry.raws", "rb");
	fread(phantom, sizeof(short), 181*217*181, fidPhantom);
	fclose(fidPhantom);

	std::clock_t start, end;
	start=std::clock();

	directionVector[0][0]=-125.8;
	directionVector[0][1]=-139.3;
	directionVector[0][2]=-71.69;

	directionVector[1][0]=-163.84;
	directionVector[1][1]=-8.34;
	directionVector[1][2]=-116;

	directionVector[2][0]=-187.35;
	directionVector[2][1]=72.59;
	directionVector[2][2]=0;

	directionVector[3][0]=-163.84;
	directionVector[3][1]=-8.34;
	directionVector[3][2]=116;

	directionVector[4][0]=-125.8;
	directionVector[4][1]=-139.3;
	directionVector[4][2]=71.69;

	directionVector[5][0]=8.34;
	directionVector[5][1]=-163.84;
	directionVector[5][2]=-116;

	directionVector[6][0]=8.34;
	directionVector[6][1]=-163.84;
	directionVector[6][2]=116;

	directionVector[7][0]=-53.21;
	directionVector[7][1]=48.05;
	directionVector[7][2]=187.7;

	directionVector[8][0]=-91.25;
	directionVector[8][1]=179.01;
	directionVector[8][2]=0;

	directionVector[9][0]=-53.21;
	directionVector[9][1]=48.05;
	directionVector[9][2]=-187.7;

	directionVector[10][0]=53.21;
	directionVector[10][1]=-48.05;
	directionVector[10][2]=-187.7;

	directionVector[11][0]=91.25;
	directionVector[11][1]=-179.01;
	directionVector[11][2]=0;

	directionVector[12][0]=53.21;
	directionVector[12][1]=-48.05;
	directionVector[12][2]=187.7;

	directionVector[13][0]=-8.34;
	directionVector[13][1]=163.84;
	directionVector[13][2]=116;

	directionVector[14][0]=-8.34;
	directionVector[14][1]=163.84;
	directionVector[14][2]=-116;

	for(numProjection=0;numProjection<1;numProjection++){

		sprintf(fileName,"G:\\research\\data\\listModeProjection%d_test.bin",numProjection);
		fid=fopen(fileName,"wb");
		centerOfAperturePosition[0]=directionVector[numProjection][0];
		centerOfAperturePosition[1]=directionVector[numProjection][1];
		centerOfAperturePosition[2]=directionVector[numProjection][2];
		alpha=0;
		rotateFacetToVector(stackRotatedToVector, stack, numStack, alpha, centerOfAperturePosition);
		cudaMemcpy(stackRotatedToVector_dev, stackRotatedToVector, sizeof(specFacet)*numStack, cudaMemcpyHostToDevice);
		EdgePointOriginal[0]=0;
		EdgePointOriginal[1]=1;
		EdgePointOriginal[2]=0;
		rotationFacetEdgePointGeneration(MBack, EdgePointFinal, EdgePointOriginal, alpha, centerOfAperturePosition);

		for(i=0;i<181;i++){
			for(j=0;j<217;j++){
				for(k=0;k<181;k++){
					photonPosition[0]=i-90;
					photonPosition[1]=j-108;
					photonPosition[2]=k-90;
					rotationAngleGeneration(rotationSpec, EdgePointFinal, photonPosition, centerOfAperturePosition);
					value=phantom[217*181*i+181*j+k];
					if(value){
						std::poisson_distribution<int> distributionP(value);
						numPhoton=distributionP(generator);
					}else{
						numPhoton=0;
					}
					
					velocityRotationMatrixGeneration(M, rotationSpec, photonPosition, centerOfAperturePosition, 0);
					cudaMemcpy(photonPosition_dev, photonPosition, sizeof(float)*3, cudaMemcpyHostToDevice);
					cudaMemcpy(M_dev, M, sizeof(float)*9, cudaMemcpyHostToDevice);
					//fastRayTracing(detector, DETECTORWIDTH, DETECTORHEIGHT, PITCH, nv, nv_dev, randomArray_dev, state_dev, numPhoton, photonPosition, photonPosition_dev, stackRotatedToVector, stackRotatedToVector_dev, numStack, workPlace, workPlace_dev, vectorsResult_dev, &accumulatedNum, M_dev, MBack);
					if(numPhoton){
						fastRayTracingListMode(fid, nv, nv_dev, randomArray_dev, state_dev, numPhoton, photonPosition, photonPosition_dev, stackRotatedToVector, stackRotatedToVector_dev, numStack, workPlace, workPlace_dev, vectorsResult_dev, &accumulatedNum, M_dev, MBack);
					}
				}
			}
			printf("%d is complete\n", i);
		}

		fclose(fid);
		printf("Projection %d is completed!!\n", numProjection);
	}	
	end=std::clock();		
	
	printf("time is %d\n", end-start);
	getchar();
}