#include <stdlib.h>
#include <sys/time.h>

timeval StartingTime;

void setTime(){
	gettimeofday( &StartingTime, NULL );
}

double getTime(){
	timeval PausingTime, ElapsedTime;
	gettimeofday( &PausingTime, NULL );
	timersub(&PausingTime, &StartingTime, &ElapsedTime);
	return ElapsedTime.tv_sec*1000.0+ElapsedTime.tv_usec/1000.0;	// Returning in milliseconds.
}
