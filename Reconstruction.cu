#include"header_simulation_engine.h"

#define NUMITERATION 10
#define NUMPROJECTIONS 14 
#define DETECTORWIDTH 100
#define DETECTORHEIGHT 100
#define NUMLOCALX 32
#define NUMLOCALY 32
#define PITCH 0.1

int main(){
	char buffer[50];
	int i, idxProjection, idxImigrationLocal, idxImigrationImage, *imageReal, *imageRealVertical, *imageRealHorizontal, idxIteration, objectPointNum=50*50*50, numLocal=NUMLOCALX*NUMLOCALY, detectorNumWidth=DETECTORWIDTH/PITCH, detectorNumHeight=DETECTORHEIGHT/PITCH, detectorNum=detectorNumWidth*detectorNumHeight, *cor, *cor_dev, numBlockAdjust=(numLocal*NUMPROJECTIONS)/32+1, driftX=detectorNumWidth/2-NUMLOCALX/2, driftY=detectorNumHeight/2-NUMLOCALY/2;
	float *object, *image, *imageVertical, *imageHorizontal, *image_dev, *imageVertical_dev, *imageHorizontal_dev, *psfPartial, *psfPartial_dev, psfBuffer, integrationAdjust, integrationPsf;
	FILE *fidRecon, *fidImageReal, *fidPSF;

	object=(float *)malloc(sizeof(float)*objectPointNum);
	image=(float *)malloc(sizeof(float)*detectorNumWidth*detectorNumHeight*NUMPROJECTIONS);
	imageVertical=(float *)malloc(sizeof(float)*detectorNumWidth*NUMPROJECTIONS);
	imageHorizontal=(float *)malloc(sizeof(float)*detectorNumWidth*NUMPROJECTIONS);
	imageReal=(int *)malloc(sizeof(int)*detectorNumWidth*detectorNumHeight*NUMPROJECTIONS);
	imageRealVertical=(int *)malloc(sizeof(int)*detectorNumWidth*NUMPROJECTIONS);
	imageRealHorizontal=(int *)malloc(sizeof(int)*detectorNumWidth*NUMPROJECTIONS);
	cudaMalloc((void**)&image_dev,sizeof(float)*detectorNumWidth*detectorNumHeight*NUMPROJECTIONS);
	cudaMalloc((void**)&imageVertical_dev,sizeof(float)*detectorNumWidth*NUMPROJECTIONS);
	cudaMalloc((void**)&imageHorizontal_dev,sizeof(float)*detectorNumWidth*NUMPROJECTIONS);
	cor=(int *)malloc(sizeof(int)*2);
	cudaMalloc((void**)&cor_dev,sizeof(int)*2);
	psfPartial=(float *)malloc(sizeof(float)*numLocal);
	cudaMalloc((void**)&psfPartial_dev,sizeof(float)*numLocal);
	///*
	fidImageReal=fopen("G:\\research\\data\\projectionImage.bin", "rb");
	fread(imageReal, sizeof(int), detectorNumWidth*detectorNumHeight*NUMPROJECTIONS, fidImageReal);
	fclose(fidImageReal);
	//*/
	///*
	fidImageReal=fopen("G:\\research\\data\\projectionImageVerticalSlit.bin", "rb");
	fread(imageRealVertical, sizeof(int), detectorNumWidth*NUMPROJECTIONS, fidImageReal);
	fclose(fidImageReal);

	fidImageReal=fopen("G:\\research\\data\\projectionImageHorizontalSlit.bin", "rb");
	fread(imageRealHorizontal, sizeof(int), detectorNumWidth*NUMPROJECTIONS, fidImageReal);
	fclose(fidImageReal);
	//*/
	for(i=0;i<objectPointNum;i++){
		object[i]=1;
	}
	

	for(idxIteration=0;idxIteration<NUMITERATION;idxIteration++){
		///*
		for(i=0;i<detectorNumWidth*detectorNumHeight*NUMPROJECTIONS;i++){
			image[i]=0;
		}
		cudaMemcpy(image_dev, image, sizeof(float)*detectorNumWidth*detectorNumHeight*NUMPROJECTIONS, cudaMemcpyHostToDevice);		
		//*/
		///*
		for(i=0;i<detectorNumWidth*NUMPROJECTIONS;i++){
			imageVertical[i]=0;
		}
		cudaMemcpy(imageVertical_dev, imageVertical, sizeof(float)*detectorNumWidth*NUMPROJECTIONS, cudaMemcpyHostToDevice);
		for(i=0;i<detectorNumWidth*NUMPROJECTIONS;i++){
			imageHorizontal[i]=0;
		}
		cudaMemcpy(imageHorizontal_dev, imageHorizontal, sizeof(float)*detectorNumWidth*NUMPROJECTIONS, cudaMemcpyHostToDevice);

		forwardProjectionAllPoints(image, image_dev, cor, cor_dev, psfPartial, psfPartial_dev, NUMPROJECTIONS, object, objectPointNum, NUMLOCALX, NUMLOCALY, detectorNumWidth, detectorNumHeight, detectorNum, 'p');
		forwardProjectionAllPoints(imageVertical, imageVertical_dev, cor, cor_dev, psfPartial, psfPartial_dev, NUMPROJECTIONS, object, objectPointNum, NUMLOCALX, NUMLOCALY, detectorNumWidth, detectorNumHeight, detectorNumWidth, 'v');
		forwardProjectionAllPoints(imageHorizontal, imageHorizontal_dev, cor, cor_dev, psfPartial, psfPartial_dev, NUMPROJECTIONS, object, objectPointNum, NUMLOCALX, NUMLOCALY, detectorNumWidth, detectorNumHeight, detectorNumWidth, 'h');

		for(i=0;i<objectPointNum;i++){
			integrationAdjust=0;
			integrationPsf=0;
			for(idxProjection=0;idxProjection<NUMPROJECTIONS;idxProjection++){
				
				sprintf(buffer, "G:\\research\\data\\PSF%d.bin", idxProjection);
				fidPSF=fopen(buffer, "rb");
				fseek(fidPSF, ((numLocal+2)*i)*4, SEEK_SET);
				fread(cor, sizeof(int), 2, fidPSF);
				for(idxImigrationLocal=0;idxImigrationLocal<numLocal;idxImigrationLocal++){
					fread(&psfBuffer, sizeof(float), 1, fidPSF);
					idxImigrationImage=(cor[0]+driftX+idxImigrationLocal/32)*detectorNumHeight+(cor[1]+driftY+idxImigrationLocal%32);
					if(idxImigrationImage>=0&&idxImigrationImage<detectorNum){
						if(image[idxImigrationImage+detectorNum*idxProjection]){
							integrationAdjust=integrationAdjust+imageReal[idxImigrationImage+detectorNum*idxProjection]/image[idxImigrationImage+detectorNum*idxProjection]*psfBuffer;
							integrationPsf=integrationPsf+psfBuffer;
						}
					}
				}
				fclose(fidPSF);
				
				///*
				sprintf(buffer, "G:\\research\\data\\vslitPSF%d.bin", idxProjection);
				fidPSF=fopen(buffer, "rb");
				fseek(fidPSF, ((NUMLOCALX+1)*i)*4, SEEK_SET);
				fread(cor, sizeof(int), 1, fidPSF);
				for(idxImigrationLocal=0;idxImigrationLocal<NUMLOCALX;idxImigrationLocal++){
					fread(&psfBuffer, sizeof(float), 1, fidPSF);
					idxImigrationImage=cor[0]+driftX+idxImigrationLocal;
					if(idxImigrationImage>=0&&idxImigrationImage<detectorNumWidth){
						if(imageVertical[idxImigrationImage+detectorNumWidth*idxProjection]){
							integrationAdjust=integrationAdjust+imageRealVertical[idxImigrationImage+detectorNumWidth*idxProjection]/imageVertical[idxImigrationImage+detectorNumWidth*idxProjection]*psfBuffer;
							integrationPsf=integrationPsf+psfBuffer;
						}
					}
				}
				fclose(fidPSF);

				sprintf(buffer, "G:\\research\\data\\hslitPSF%d.bin", idxProjection);
				fidPSF=fopen(buffer, "rb");
				fseek(fidPSF, ((NUMLOCALX+1)*i)*4, SEEK_SET);
				fread(cor, sizeof(int), 1, fidPSF);
				for(idxImigrationLocal=0;idxImigrationLocal<NUMLOCALX;idxImigrationLocal++){
					fread(&psfBuffer, sizeof(float), 1, fidPSF);
					idxImigrationImage=cor[0]+driftX+idxImigrationLocal;
					if(idxImigrationImage>=0&&idxImigrationImage<detectorNumWidth){
						if(imageHorizontal[idxImigrationImage+detectorNumWidth*idxProjection]){
							integrationAdjust=integrationAdjust+imageRealHorizontal[idxImigrationImage+detectorNumWidth*idxProjection]/imageHorizontal[idxImigrationImage+detectorNumWidth*idxProjection]*psfBuffer;
							integrationPsf=integrationPsf+psfBuffer;
						}
					}
				}
				fclose(fidPSF);
				//*/
			}
			object[i]=object[i]*integrationAdjust/integrationPsf;
		}				
		
		printf("Iteration %d is finished!\n", idxIteration);
		fidRecon=fopen("G:\\research\\data\\recon.bin", "wb");
		fwrite(object, sizeof(float), objectPointNum, fidRecon);
		fclose(fidRecon);
	}
	fidRecon=fopen("G:\\research\\data\\recon.bin", "wb");
	fwrite(object, sizeof(float), objectPointNum, fidRecon);
	fclose(fidRecon);
	getchar();
}