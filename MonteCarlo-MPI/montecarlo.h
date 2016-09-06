#ifndef __MONTECARLO_H
#define __MONTECARLO_H

#include "mpi.h"
#define MIN(a,b) (((a)<(b))?(a):(b))

MPI_Datatype MPI_TaskDataType;
enum TASK_TYPE {STOP_EXECUTION, EXECUTE};

struct Point {
	double x;
	double y;
};

struct Task {
	TASK_TYPE type;
	int       pointsToProcess;
	double    circleRadius;
};

struct Result {
	int processedPoints;
	int totalHits;
};


void   createAndCommitTaskDataType(Task* task);
int    calculateTask(Task* task);
Point  generatePoint(double radius);
double getRandomDouble(double max, double min);
bool   pointWithinBounds(Point* point, double radius);
Result masterWork(int numberOfWorkers);
void   slaveWork();

#endif