#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "mpi.h"

#include "montecarlo.h"

const int    MAX_POINTS = 20000;
const double CIRCLE_RADIUS = 1;
const int    POINTS_PER_SLAVE = 500;
const int    MASTER_RANK = 0;
#define MIN(a,b) (((a)<(b))?(a):(b))

MPI_Datatype MPI_TaskDataType;

int main(int argc, char* argv[])
{
	srand(time(NULL)); // Seed random function
	int numberOfWorkers, currentId;
	double startTime, stopTime;
	
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &currentId);
	MPI_Comm_size(MPI_COMM_WORLD, &numberOfWorkers);

	if (numberOfWorkers < 2)
	{
		printf("Program requires at least 2 nodes");
		MPI_Finalize();
		exit(1);
	}


	if (currentId == MASTER_RANK) // Master work
	{	
		Result result;

		startTime = MPI_Wtime();
		result = masterWork(numberOfWorkers);
		stopTime = MPI_Wtime();

		printf("PI: %f\n", 4*(result.totalHits/(double)result.processedPoints));
		printf("Execution Time: %f\n", stopTime - startTime);
	}
	else // Slaves work
	{
		slaveWork();
	}
	MPI_Finalize();
	return 0;
}

Result masterWork(int numberOfWorkers)
{
	// We sending constant number of point to calculate = POINTS_PER_SLAVE
	Task task = { EXECUTE, POINTS_PER_SLAVE, CIRCLE_RADIUS };
	Result result = {0, 0};
	MPI_Status status;


	createAndCommitTaskDataType(&task);

	// Initial work send. We send only number of points to process by slave.
	// Slave will generate point and check if it's in CIRCLE bound
	for (int i = 1; i < numberOfWorkers; i++)
	{
		task.pointsToProcess = MIN(POINTS_PER_SLAVE, MAX_POINTS - result.processedPoints);
		MPI_Send(&task, 1, MPI_TaskDataType, i, 0, MPI_COMM_WORLD);
	}

	// Send Rest  of work to clients that completes it's execution
	while (result.processedPoints < MAX_POINTS)
	{
		int receivedHits;
		// Receive hists
		MPI_Recv(&receivedHits, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
		result.processedPoints += task.pointsToProcess;
		result.totalHits += receivedHits;
		
		task.pointsToProcess = MIN(POINTS_PER_SLAVE, MAX_POINTS - result.processedPoints);
		// Send next task to slave that completed it's calculation
		MPI_Send(&task, 1, MPI_TaskDataType, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
	}

	// Send Stop execution to slaves
	task.type = STOP_EXECUTION;
	for (int i = 1; i < numberOfWorkers; i++)
	{
		MPI_Send(&task, 1, MPI_TaskDataType, i, 0, MPI_COMM_WORLD);
	}

	return result;
}

void slaveWork()
{
	Task task;
	MPI_Status status;
	bool needToStop = false;
	int hits = 0;

	createAndCommitTaskDataType(&task);
	// Calculate point until we don't need to stop
	while (!needToStop) {
		MPI_Recv(&task, 1, MPI_TaskDataType, MASTER_RANK, 0, MPI_COMM_WORLD, &status);
		// we got new task to process
		if (task.type == EXECUTE) {
			hits = calculateTask(&task);
			MPI_Send(&hits, 1, MPI_INT, MASTER_RANK, 0, MPI_COMM_WORLD);
		}
		else // we need to stop
		{
			needToStop = true;
		}
	}
}

void createAndCommitTaskDataType(Task* task)
{
	MPI_Datatype innerTypes[3] = { MPI_INT, MPI_INT, MPI_DOUBLE };
	MPI_Aint addressesOfInnerTypes[3];
	int innerTypeLen[3] = { 1, 1, 1 };
	
	// Calculate address offsets
	addressesOfInnerTypes[0] = (char*)&task->type - (char*)task;
	addressesOfInnerTypes[1] = (char*)&task->pointsToProcess - (char*)task;
	addressesOfInnerTypes[2] = (char*)&task->circleRadius - (char*)task;

	// Create and commit data type binding
	MPI_Type_create_struct(3, innerTypeLen, addressesOfInnerTypes, innerTypes, &MPI_TaskDataType);
	MPI_Type_commit(&MPI_TaskDataType);

}

int calculateTask(Task* task)
{
	int hits = 0;
	Point point;

	for (int i = 0; i < task->pointsToProcess; i++)
	{
		point = generatePoint(task->circleRadius);
		if (pointWithinBounds(&point, task->circleRadius)) {
			hits++;
		}
	}
	
	return hits;
}

Point generatePoint(double radius)
{
	Point point;
	point.x = getRandomDouble(-radius, radius);
	point.y = getRandomDouble(-radius, radius);
	return point;
}

double getRandomDouble(double max, double min)
{
	double random = (double)rand() / RAND_MAX;
	return min + random * (max - min);
}

bool pointWithinBounds(Point* point, double radius)
{
	double x_2 = pow(point->x, 2);
	double y_2 = pow(point->y, 2);
	double calculatedRadius = sqrt(x_2 + y_2);
	return calculatedRadius < radius;
}

