#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define NUMX 800
#define NUMY 800
#define NUMPIX 600
#define NUMPROJECTIONS 90
#define HALFPIXLENGTH 30
#define PITCHX 0.1
#define PITCHY 0.1
#define PITCHVECTOR 0.1
#define MINION 1e-28
#define NUMANGLE 1

const double PI=3.141592653589793238462;

void fastRadonTransform(float *vector, float angle, float *image);
float pixelLineIntersection(float *positionPix, float pitchX, float pitchY, float *positionPointLine, float theta);

void main(){
	char buffer[150];
	int i, j, idxVector, idxTheta, *gRaw, pageCount=NUMX*NUMY*3;
	float *vector, *image, *gFiltered, *gProjected, angle, temp, halfXPix=(NUMX-1.0)/2, halfYPix=(NUMY-1.0)/2, halfVectorPix=(NUMPIX-1.0)/2, Cos, Sin, pitchVectorCos, pitchVectorSin, test, positionPix[2]={0,0}, positionPointLine[2]={0,-0.005}, *HMatrix;
	FILE *fid;

	HMatrix=(float *)malloc(sizeof(float)*NUMX*NUMY*NUMPROJECTIONS*3);

	fid=fopen("G:\\research\\data\\HMatrix.bin","wb");

	for(idxTheta=0;idxTheta<NUMPROJECTIONS;idxTheta++){

		angle=idxTheta*PI/NUMPROJECTIONS;
		Cos=cos(angle);
		Sin=sin(angle);
		pitchVectorCos=PITCHVECTOR*cos(angle);
		pitchVectorSin=PITCHVECTOR*sin(angle);
		for(i=0;i<NUMX;i++){
			for(j=0;j<NUMY;j++){
				positionPix[0]=(i-halfXPix)*PITCHX;
				positionPix[1]=(j-halfYPix)*PITCHY;

				idxVector=(positionPix[0]*Cos+positionPix[1]*Sin)/PITCHVECTOR+halfVectorPix+0.5;

				if(idxVector>=0&&idxVector<NUMPIX){
					positionPointLine[0]=(idxVector-halfVectorPix)*pitchVectorCos;
					positionPointLine[1]=(idxVector-halfVectorPix)*pitchVectorSin;
					HMatrix[pageCount*idxTheta+(i*NUMY+j)*3+1]=pixelLineIntersection(positionPix, PITCHX, PITCHY, positionPointLine, angle+PI/2);
				}else{
					HMatrix[pageCount*idxTheta+(i*NUMY+j)*3+1]=0;
				}

				idxVector--;

				if(idxVector>=0&&idxVector<NUMPIX){
					positionPointLine[0]=(idxVector-halfVectorPix)*pitchVectorCos;
					positionPointLine[1]=(idxVector-halfVectorPix)*pitchVectorSin;
					HMatrix[pageCount*idxTheta+(i*NUMY+j)*3]=pixelLineIntersection(positionPix, PITCHX, PITCHY, positionPointLine, angle+PI/2);
				}else{
					HMatrix[pageCount*idxTheta+(i*NUMY+j)*3]=0;
				}

				idxVector=idxVector+2;

				if(idxVector>=0&&idxVector<NUMPIX){
					positionPointLine[0]=(idxVector-halfVectorPix)*pitchVectorCos;
					positionPointLine[1]=(idxVector-halfVectorPix)*pitchVectorSin;
					HMatrix[pageCount*idxTheta+(i*NUMY+j)*3+2]=pixelLineIntersection(positionPix, PITCHX, PITCHY, positionPointLine, angle+PI/2);
				}else{
					HMatrix[pageCount*idxTheta+(i*NUMY+j)*3+2]=0;
				}				
			}
		}	
		printf("%d projection is finished\n", idxTheta);
	}
	
	fwrite(HMatrix, sizeof(float), NUMX*NUMY*NUMPROJECTIONS*3, fid);
	fclose(fid);
	return;
}


void fastRadonTransform(float *vector, float angle, float *image){
	int idxVectorPix, i, j, idxThisX, idxThisY;
	float xLimitOngrid=NUMX/2.0*PITCHX, yLimitOngrid=NUMY/2.0*PITCHY, lengthY, currentX, currentY, deltaXOngrid, deltaXOnline, deltaYOngrid, deltaYOnline, positionXOngrid, positionYOngrid, positionXOnline, positionYOnline, positionVectorPix[2], halfXPix=(NUMX-1.0)/2, halfYPix=(NUMY-1.0)/2, halfVectorPix=(NUMPIX-1.0)/2, cosV=cos(angle), sinV=sin(angle), tanV=tan(angle), tanNV=1/tan(angle), cosNV=1/cos(angle);

	deltaXOngrid=PITCHX;
	deltaXOnline=PITCHY*abs(tanV);

	deltaYOngrid=PITCHY;
	deltaYOnline=PITCHX*abs(tanNV);

	if(angle==0){
		for(idxVectorPix=0;idxVectorPix<NUMPIX;idxVectorPix++){
			idxThisX=(idxVectorPix-halfVectorPix)*PITCHVECTOR/PITCHX+halfXPix;
			if(idxThisX>0&&idxThisX<NUMX){
				for(idxThisY=0;idxThisY<NUMY;idxThisY++){
					vector[idxVectorPix]=vector[idxVectorPix]+image[idxThisX*NUMY+idxThisY]*PITCHY;
				}
			}
		}
	}else if(angle<PI/2){

		for(idxVectorPix=0;idxVectorPix<NUMPIX;idxVectorPix++){
			positionVectorPix[0]=(idxVectorPix-halfVectorPix)*cosV*PITCHVECTOR;
			positionVectorPix[1]=(idxVectorPix-halfVectorPix)*sinV*PITCHVECTOR;

			positionYOngrid=-NUMY/2.0*PITCHY;
			positionXOnline=positionVectorPix[0]-(positionYOngrid-positionVectorPix[1])*tanV;

			positionXOngrid=NUMX/2.0*PITCHX;
			positionYOnline=positionVectorPix[1]-(positionXOngrid-positionVectorPix[0])*tanNV;

			i=NUMX-1;
			j=0;

			if(positionYOngrid<positionYOnline){
				currentX=positionXOnline;
				currentY=positionYOngrid;

				positionXOnline=positionXOnline-deltaXOnline;
				positionYOngrid=positionYOngrid+deltaYOngrid;
			}else{
				currentX=positionXOngrid;
				currentY=positionYOnline;

				positionXOngrid=positionXOngrid-deltaXOngrid;
				positionYOnline=positionYOnline+deltaYOnline;
			}

			while(currentX>=-xLimitOngrid&&currentY<=yLimitOngrid&&i>=0&&j<=NUMY-1){
				if(positionYOngrid<positionYOnline){
					if(currentX<=xLimitOngrid&&currentY>=-yLimitOngrid){
						lengthY=positionYOngrid-currentY;
						vector[idxVectorPix]=vector[idxVectorPix]+image[i*NUMY+j]*lengthY*cosNV;
					}
					j++;
					currentX=positionXOnline;
					currentY=positionYOngrid;
					positionXOnline=positionXOnline-deltaXOnline;
					positionYOngrid=positionYOngrid+deltaYOngrid;
				}else{
					if(currentX<=xLimitOngrid&&currentY>=-yLimitOngrid){
						lengthY=positionYOnline-currentY;
						vector[idxVectorPix]=vector[idxVectorPix]+image[i*NUMY+j]*lengthY*cosNV;
					}	
					i--;
					currentX=positionXOngrid;
					currentY=positionYOnline;
					positionXOngrid=positionXOngrid-deltaXOngrid;
					positionYOnline=positionYOnline+deltaYOnline;
				}
			}

		}
	}else if(angle==float(PI/2)){
		for(idxVectorPix=0;idxVectorPix<NUMPIX;idxVectorPix++){
			idxThisY=(idxVectorPix-halfVectorPix)*PITCHVECTOR/PITCHY+halfYPix;
			if(idxThisY>0&&idxThisY<NUMY){
				for(idxThisX=0;idxThisX<NUMX;idxThisX++){
					vector[idxVectorPix]=vector[idxVectorPix]+image[idxThisX*NUMY+idxThisY]*PITCHX;
				}
			}
		}
	}else if(angle>PI/2){

		for(idxVectorPix=0;idxVectorPix<NUMPIX;idxVectorPix++){
			positionVectorPix[0]=(idxVectorPix-halfVectorPix)*cosV*PITCHVECTOR;
			positionVectorPix[1]=(idxVectorPix-halfVectorPix)*sinV*PITCHVECTOR;

			positionYOngrid=-NUMY/2.0*PITCHY;
			positionXOnline=positionVectorPix[0]-(positionYOngrid-positionVectorPix[1])*tanV;

			positionXOngrid=-NUMX/2.0*PITCHX;
			positionYOnline=positionVectorPix[1]-(positionXOngrid-positionVectorPix[0])*tanNV;

			i=0;
			j=0;

			if(positionYOngrid<positionYOnline){
				currentX=positionXOnline;
				currentY=positionYOngrid;

				positionXOnline=positionXOnline+deltaXOnline;
				positionYOngrid=positionYOngrid+deltaYOngrid;
			}else{
				currentX=positionXOngrid;
				currentY=positionYOnline;

				positionXOngrid=positionXOngrid+deltaXOngrid;
				positionYOnline=positionYOnline+deltaYOnline;
			}

			while(currentX<=xLimitOngrid&&currentY<=yLimitOngrid&&i<=NUMX-1&&j<=NUMY-1){
				if(positionYOngrid<positionYOnline){
					if(currentX>=-xLimitOngrid&&currentY>=-yLimitOngrid){
						lengthY=positionYOngrid-currentY;
						vector[idxVectorPix]=vector[idxVectorPix]-image[i*NUMY+j]*lengthY*cosNV;
					}
					j++;
					currentX=positionXOnline;
					currentY=positionYOngrid;
					positionXOnline=positionXOnline+deltaXOnline;
					positionYOngrid=positionYOngrid+deltaYOngrid;
				}else{
					if(currentX>=-xLimitOngrid&&currentY>=-yLimitOngrid){
						lengthY=positionYOnline-currentY;
						vector[idxVectorPix]=vector[idxVectorPix]-image[i*NUMY+j]*lengthY*cosNV;
					}	
					i++;
					currentX=positionXOngrid;
					currentY=positionYOnline;
					positionXOngrid=positionXOngrid+deltaXOngrid;
					positionYOnline=positionYOnline+deltaYOnline;
				}
			}
		}
	}
}

float pixelLineIntersection(float *positionPix, float pitchX, float pitchY, float *positionPointLine, float theta){
	float xLeft=positionPix[0]-pitchX/2, xRight=positionPix[0]+pitchX/2, yLow=positionPix[1]-pitchY/2, yHigh=positionPix[1]+pitchY/2, yOnline1, yOnline2, temp, pointUp, pointDown, length;

	if(theta==float(PI/2)||theta==float(3*PI/2)){
		if(positionPointLine[0]>=xLeft&&positionPointLine[0]<xRight){
			return pitchY;
		}else{
			return 0;
		}
	}else if(theta==0||theta==float(PI)){
		if(positionPointLine[1]>=yLow&&positionPointLine[1]<yHigh){
			return pitchY;
		}else{
			return 0;
		}
	}else{
		yOnline1=positionPointLine[1]+(xLeft-positionPointLine[0])*tan(theta);
		yOnline2=positionPointLine[1]+(xRight-positionPointLine[0])*tan(theta);

		if(yOnline1>yOnline2){
			temp=yOnline2;
			yOnline2=yOnline1;
			yOnline1=temp;
		}

		if(yOnline2>yHigh){
			pointUp=yHigh;
		}else{
			pointUp=yOnline2;
		}

		if(yOnline1>yLow){
			pointDown=yOnline1;
		}else{
			pointDown=yLow;
		}

		length=pointUp-pointDown;

		if(length>=0){
			return length/abs(sin(theta));
		}else{
			return 0;
		}
	}
}
