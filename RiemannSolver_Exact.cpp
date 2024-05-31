#include <stdio.h>
#include <cstdio>
#include <cstdlib>
//#include <mpi.h>
//#include <omp.h>

#include <exception>
#include <string>
#include <math.h>
#include <vector>
#include <time.h>

#define IsTestMode 0

using namespace std;

// ----------------- define functions -----------------//
void SetInitState(const string &argStr, vector<double> &state);
double ComputeF(const double gamma, const double pStar, const vector<double> &leftState, const vector<double> &rightState);
double ComputeDiffF(const double gamma, const double pStar, const vector<double> &leftState, const vector<double> &rightState);
double Computef(const double gamma, const double pStar, const vector<double> &leftState, const vector<double> &rightState);
void SaveData(const int NGrids, string &basicStr, const vector<double> &x, const vector<vector<double>> &U);
                
// -----------------------------------------------------------
// the main funtion with arguments:
// argv[1]: (int)NGrid
// argv[2]: (int)TimeStep
// argv[3]: (double)EndTime
// argv[4]: (int)NThreads
// argv[5]: (string)LeftState 
// argv[6]: (string)RightState
// ex. ./RiemannSolver_Exact.out {NGrid} {TimeStep} {EndTime} {NThreads} {LeftState} {RightState}
// -----------------------------------------------------------
int main(int argc, char *argv[]) 
{    
/********* set mpi *********/

/********* set openmp *********/
    
/********* get the arguments *********/
    string argv1, argv2, argv3, argv4, argv5, argv6;
    if (IsTestMode)
    {
        argv1 = "10";
        argv2 = "10";
        argv3 = "0.1";
        argv4 = "4";
        argv5 = "1.25e3,0.0,5.0e2";
        argv6 = "0.125,0.0,0.1";
    }else
    {
        argv1 = argv[1];
        argv2 = argv[2];
        argv3 = argv[3];
        argv4 = argv[4];
        argv5 = argv[5];
        argv6 = argv[6];  
    }

/********* set variables from arguments *********/
   int NGrid = 2000;                    // the number of grid(1D)
   int TimeStep = 1000;                 // the time steps
   double EndTime = 0.1;                // the end of time
   int NThread = 4;                     // the number of threads
   int NComponents = 3;                 // the number of the Euler equations
   vector<double> LeftState;            // [rho_L, u_L, p_L, c_L], take u=Vx for 1D case
   vector<double> RightState;           // [rho_R, u_R, p_R, c_R] 
   try {
       NGrid = stoi(argv1);
       TimeStep = stoi(argv2);
       EndTime = stod(argv3);
       NThread = stoi(argv4);
       SetInitState(argv5, LeftState);   // insert [rho_L, u_L, p_L]
       SetInitState(argv6, RightState);  // insert [rho_R, u_R, p_R]
       NComponents = LeftState.size();   // only for 1D case
   } catch (exception const &ex) {
       printf( "There are invalid numbers\n" );
       return EXIT_FAILURE;
   }

/********* set constants and variables *********/
    const double gamma = 1.4;               // the ratio of specific heats, Cp/Cv, gamma=1.4 for air
    const double dx = 1. / NGrid;           // the space interval, uniform over x, y and z
    const double dt = EndTime / TimeStep;   // the time interval
    const double x0 = 0.00025;              // the start grid of x
    const int halfGrid = NGrid / 2;         // the half of NGrid
    vector<vector<double>> U(NComponents);  // the vector collects the primitive variables, U_i(x) = [rho_i, u_i, P_i]^T(x)
    vector<double> x(NGrid);                // the grid points on x
    double t;                               // the current time
    double pStar;                           // the pressure in the middle region
    string str;                             // the string value for printed out

/********* initialization *********/
    clock_t tStart = clock();                                           // start the clock

//  set LeftState and RightState
    LeftState.push_back(sqrt(gamma * LeftState[2] / LeftState[0]));     // insert c_L
    RightState.push_back(sqrt(gamma * RightState[2] / RightState[0]));  // insert c_R

// verify the results: LeftState and RightState
    if (IsTestMode) 
    {
        printf("LeftState[rho_L, u_L, p_L, c_L]: %.4f\t%.4f\t%13.6e%13.6e\n", 
                LeftState[0], LeftState[1], LeftState[2], LeftState[3]);
        printf("RightState[rho_R, u_R, p_R, c_R]: %.4f\t%.4f\t%13.6e%13.6e\n\n", 
                RightState[0], RightState[1], RightState[2], RightState[3]);
    }
//  set U
    for (int i = 0 ; i < NComponents ; i++)
    {
        U[i] = vector<double>(NGrid);
        for (int j = 0 ; j < halfGrid ; j++)
        {
            U[i][j] = LeftState[i];
        }
        for (int j = halfGrid ; j < NGrid ; j++)
        {
            U[i][j] = RightState[i];
        }
    }

//  set x
    x[0] = x0;
    x[halfGrid] = x0 + halfGrid*dx;
    for (int i = 1 ; i < halfGrid ; i++)
        x[i] = x[i-1] + dx;
    for (int i = halfGrid + 1 ; i < NGrid ; i++)
        x[i] = x[i-1] + dx;
    
// verify the results: x and U
    if (IsTestMode) 
    {
        str = "";
        str.append("U:\n");
        for (int i = 0; i < NComponents; i++){
            for (int j = 0; j < NGrid; j++)
                str.append(to_string(U[i][j]) + "  ");
            str.append("\n");
        }
        
        printf("%s\n", str.c_str());
        
        str = "";
        str.append("x:\n");
        for (int i = 0; i < NGrid; i++)
            str.append(to_string(x[i]) + "  ");
        str.append("\n");
        printf("%s\n", str.c_str());
    }
    
// compute pStar
    pStar = CompPStarNewton(gamma, LeftState, RightState);
// verify the results: pStar
    if (IsTestMode) 
        printf("pStar: %.6f\n\n", pStar);
    
/********* compute U over time *********/
/*
    t = 0.0;
    while ( t <= EndTime)
    {
        for (int i = 0 ; i < NGrid ; i++)
        {
            % update U[i][j]
        }
    
        t += dt;
    }
*/

    clock_t tEnd = clock();         // finish the clock
    
/********* save U *********/
    t = (double)(tEnd - tStart)/CLOCKS_PER_SEC;         // execution time
    str = "# This table provides a solution of the Riemann problem at t=";
    str.append(to_string(EndTime) + "\n# with (NGrid, TimeStep, ExecutionTime) = (" +
               to_string(NGrid) + "," + to_string(TimeStep) + "," + to_string(t) + ")\n" +
               "# Initial condition of the strong shock problem:\n" + 
               "#   left  state: (rho, Vx, P) = (" + argv5 + ")\n" + 
               "#   right state: (rho, Vx, P) = (" + argv6 + ")\n\n");

    SaveData(NGrid, str, x, U);
    
    return EXIT_SUCCESS;
}

// ----------------- implement functions -----------------//
// -----------------------------------------------------------
// set the initial states of left and right sides
// -----------------------------------------------------------
void SetInitState(const string &argStr, vector<double> &state) {
	int curr = 0;       // the current position
	int len = argStr.size();
	
	try
    {
    	for (int i = 0 ; i <= len ; )
    	{
    	    if(argStr[i] == ',' || i == len)
    	    {
    	        state.push_back(stod(argStr.substr(curr, i)));
    	        //printf("%.5f", stod(argStr.substr(curr, i)));
    	        curr = ++i;
    	    }else
    	        ++i;
    	}
    }catch(exception const &ex)
    {
        throw ex;
    }
}

// -----------------------------------------------------------
// compute the pStar from the Newton Method
// -----------------------------------------------------------
double CompPStarNewton(const double gamma, const vector<double> &leftState, const vector<double> &rightState)
{
    return 0.1;
}

// -----------------------------------------------------------
// compute F(pStar, gamma, LeftState, RightState)
//         = f(pStar, gamma, p_L, rho_L) +  f(pStar, gamma, p_R, rho_R) - (u_L - u_R)
// -----------------------------------------------------------
double ComputeF(const double gamma, const double pStar, 
         const vector<double> &leftState, const vector<double> &rightState)
{
    return 0.1;
}

// -----------------------------------------------------------
// compute dF/dpStar
// -----------------------------------------------------------
double ComputeDiffF(const double gamma, const double pStar, 
           const vector<double> &leftState, const vector<double> &rightState)
{
    return 0.1;
}

// -----------------------------------------------------------
// compute f(pStar, gamma, p, rho)
// -----------------------------------------------------------
double Computef(const double gamma, const double pStar, 
         const vector<double> &leftState, const vector<double> &rightState)
{
    return 0.1;
}

// -----------------------------------------------------------
// save data
// -----------------------------------------------------------
void SaveData(const int NGrid, string &headerStr, const vector<double> &x, const vector<vector<double>> &U)
{
    headerStr.append("#   \tr   \t\t\t\tRho   \t\t\t\tVx   \t\t\t\tVy   \t\t\t\tVz   \t\t\t\tPres\n");
    //printf("%s", headerStr.c_str());
    
//  create the file of results and mark the NGrid
    char FileName[100];
    sprintf( FileName, "results_%d.txt", NGrid );
    FILE *File = fopen( FileName, "w" );
    
    fprintf( File, "%s", headerStr.c_str());
//   U_i = [rho_i, Vx_i, P_i]^T, each row with data = [x, v_x, v_y, v_z, P]    
    for (int i = 0; i < NGrid; i++)
        fprintf( File, "%.7f  %.13e  %.13e  %.13e  %.13e  %.13e\n",
                 x[i], U[0][i], U[1][i], 0.0, 0.0, U[2][i] );

    fclose( File );
}

