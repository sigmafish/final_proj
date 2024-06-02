#include <stdio.h>
#include <cstdio>
#include <cstdlib>

#include <exception>
#include <string>
#include <math.h>
#include <vector>
#include <time.h>

#define IsTestMode 0

using namespace std;

// ----------------- define classes ----------------- //

// the class for store the gamma factors
class GammaFacts {
 public:
    double ga, oneOverGa, gaP1, gaM1, gaM1Over2, gaM1Over2Ga, gaP1Over2Ga, 
           twoOverGaM1, twoOverGaP1, twoGaOverGaM1, gaM1OverGaP1;

    GammaFacts(){}
    GammaFacts(const double gamma)
    {
        ga = gamma;
        gaP1 = gamma + 1;
        gaM1 = gamma - 1;
        gaM1Over2 = gaM1 / 2;
        gaM1Over2Ga = gaM1Over2 / gamma;
        gaP1Over2Ga = gaP1 / 2 / gamma;
        oneOverGa = 1 / gamma;
        twoOverGaM1 = 1 / gaM1Over2;
        twoOverGaP1 = 2 / gaP1;
        twoGaOverGaM1 = 1 / gaM1Over2Ga;
        gaM1OverGaP1 = gaM1 / gaP1;

        //printf("GammaFacts: %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f\n",  
        //       ga, gaP1, gaM1, gaM1Over2, gaM1Over2Ga, gaP1Over2Ga, oneOverGa, twoOverGaM1, twoOverGaP1, twoGaOverGaM1, gaM1OverGaP1);
    }
};

// the class for the left and right middle states
class MidState {
 public:
    vector<double> state;    // [rhoStar, uStar, pStar, cStar]
    vector<double> spSet;    // the Set of speeds

    MidState(){}
//   sign = +1:sideState=RightState ; -1:sideState=LeftState
    MidState(const vector<double> &sideState, const vector<double> &tempState, const GammaFacts &GaF, const int sign)
    {
        state = vector<double>(sideState.size());
//       set pStar
        state[2] = tempState[0];
//       set uStar
        state[1] = tempState[1];

        spSet = {0, 0, 0};
//       set contact speed when it is on the same side
        spSet[0] = sign * state[1] > 0 ? state[1] : 0;
        if (state[2] > sideState[2]) // pStar > pSide => shock
        {
//         compute and set shock speed
          spSet[1] = sideState[1] + sign * sideState[3]
                            * sqrt(GaF.gaP1Over2Ga * state[2]/sideState[2] + GaF.gaM1Over2Ga);
//         compute and set rhoStar
          state[0] = sideState[0] * (sideState[2] * GaF.gaM1 + state[2] * GaF.gaP1)
                       / (state[2] * GaF.gaM1 + sideState[2] * GaF.gaP1);
//         compute and set cStar
          state[3] = sqrt( GaF.ga * state[2] / state[0] );
        }else                       // pStar < pSide => rarefaction
        {
//         compute and set rhoStar
          state[0] = sideState[0] * pow(state[2]/sideState[2], GaF.oneOverGa);
//         compute and set cStar
          state[3] = sqrt( GaF.ga * state[2] / state[0] );
//         compute and set tail speed
          spSet[1] = state[1] + sign * state[3];
//         compute and set head speed
          spSet[2] = sideState[1] + sign * sideState[3];
        }
    }
};

// ----------------- define functions -----------------//
void SetInitState(const string &argStr, vector<double> &state);
double CompPStarBisection(const double gamma, const vector<double> &leftState, const vector<double> &rightState);
double ComputeF(const double gamma, const double pStar, const vector<double> &leftState, const vector<double> &rightState);
double Computef(const GammaFacts &GaF, const double pStar, const vector<double> &state);
void SaveData(const int NGrids, string &basicStr, const vector<double> &x, const vector<vector<double>> &U);

// -----------------------------------------------------------
// the main funtion with arguments:
// argv[1]: (double)gamma
// argv[2]: (int)NGrid
// argv[3]: (int)TimeStep
// argv[4]: (double)EndTime
// argv[5]: (int)NThreads
// argv[6]: (string)LeftState
// argv[7]: (string)RightState
// ex. ./RiemannSolver_Exact.out {gamma} {NGrid} {TimeStep} {EndTime} {NThreads} {LeftState} {RightState}
// -----------------------------------------------------------
int main(int argc, char *argv[])
{
	
/********* get the arguments *********/
    string argv1, argv2, argv3, argv4, argv5, argv6, argv7;
    if (IsTestMode)
    {
		argv1 = "1.4";
        argv2 = "20";
        argv3 = "20";
        argv4 = "0.25";
        argv5 = "4";
        argv6 = "0.125,0.0,0.1";
        argv7 = "1.0,0.0,1.0";
    }else
    {
        argv1 = argv[1];
        argv2 = argv[2];
        argv3 = argv[3];
        argv4 = argv[4];
        argv5 = argv[5];
        argv6 = argv[6];
		argv7 = argv[7];
    }

/********* set variables from arguments *********/
   double gamma = 1.4;                 // the ratio of specific heats, Cp/Cv, gamma=1.4 for air
   int NGrid = 100;                    // the number of grid(1D)
   int TimeStep = 20;                   // the time steps
   double EndTime = 0.1;                // the end of time
   int NThread = 4;                     // the number of threads
   int NComponents = 3;                 // the number of the Euler equations  
   vector<double> LeftState;            // [rho_L, u_L, P_L, c_L], take u=Vx for 1D case
   vector<double> RightState;           // [rho_R, u_R, P_R, c_R]
   try {
	   gamma = stod(argv1);
       NGrid = stoi(argv2);
       TimeStep = stoi(argv3);
       EndTime = stod(argv4);
       NThread = stoi(argv5);
       SetInitState(argv6, LeftState);   // insert [rho_L, u_L, P_L]
       SetInitState(argv7, RightState);  // insert [rho_R, u_R, P_R]
       NComponents = LeftState.size();   // only for 1D case
   } catch (exception const &ex) {
       printf( "There are invalid numbers\n" );
       return EXIT_FAILURE;
   }

/********* set constants and variables *********/
    const GammaFacts GaF(gamma);            // collect the factors related with gamma
    const double dx = 1. / NGrid;           // the space interval, uniform over x, y and z
    const double dt = EndTime / TimeStep;   // the time interval
    const double x0 = 0.00025;              // the start grid of x
    const int halfGrid = NGrid / 2;         // the half of NGrid
    vector<vector<double>> U(NComponents);  // the vector collects the primitive variables, U_i(x) = [rho_i, u_i, P_i]^T(x)
    vector<double> x(NGrid);                // the grid points on x
    vector<double> tempMidState;            // a temp vector to save the computed uStar and pStar
    MidState leftMidState, rightMidState;   // the left and right middle states
    double t;                               // the current time
    double dist_0, dist_1, dist_2;          // the distances computed by speeds, dist_i=MidState.spSet[i]*dt
    double tempVel, temp;                   // the variables save temp data 
    string str;                             // the string value for printed out

/********* initialization *********/
    clock_t tStart = clock();               // start the clock

//  set LeftState and RightState
    LeftState.push_back(sqrt(gamma * LeftState[2] / LeftState[0]));     // insert c_L
    RightState.push_back(sqrt(gamma * RightState[2] / RightState[0]));  // insert c_R

// verify the results: LeftState and RightState
    if (IsTestMode)
    {
        printf("x0=%.5f, halfGrid=%d, dx=%.5f, dt=%.5f\n",x0, halfGrid, dx, dt);
        printf("LeftState[rho_L, u_L, p_L, c_L]: %.4f\t%.4f\t%13.6e%13.6e\n",
                LeftState[0], LeftState[1], LeftState[2], LeftState[3]);
        printf("RightState[rho_R, u_R, p_R, c_R]: %.4f\t%.4f\t%13.6e%13.6e\n\n",
                RightState[0], RightState[1], RightState[2], RightState[3]);
    }
//  set U
    for (int i = 0 ; i < NComponents ; ++i)
    {
        U[i] = vector<double>(NGrid);
        for (int j = 0 ; j < halfGrid ; ++j)
        {
            U[i][j] = LeftState[i];
        }
        for (int j = halfGrid ; j < NGrid ; ++j)
        {
            U[i][j] = RightState[i];
        }
    }

//  set x
    x[0] = x0;
    x[halfGrid] = x0 + halfGrid*dx;
    for (int i = 1 ; i < halfGrid ; ++i)
        x[i] = x[i-1] + dx;
    for (int i = halfGrid + 1 ; i < NGrid ; ++i)
        x[i] = x[i-1] + dx;

// verify the results: x and U
    if (IsTestMode)
    {
        str = "";
        str.append("U:\n");
        for (int i = 0; i < NComponents; ++i){
            for (int j = 0; j < NGrid; ++j)
                str.append(to_string(U[i][j]) + "  ");
            str.append("\n");
        }

        printf("%s\n", str.c_str());

        str = "";
        str.append("x:\n");
        for (int i = 0; i < NGrid; ++i)
            str.append(to_string(x[i]) + "  ");
        str.append("\n");
        printf("%s\n", str.c_str());

        printf("x[halfGrid]=%.5f\n", x[halfGrid]);
    }

// compute pStar and uStar
    tempMidState.push_back(CompPStarBisection(gamma, LeftState, RightState));
    tempMidState.push_back(RightState[1] + Computef(GaF, tempMidState[0], RightState));
// verify the results: pStar
    if (IsTestMode)
        printf("pStar: %.6f\n\n", tempMidState[0]);

/********* compute U over time *********/

    rightMidState = MidState(RightState, tempMidState, GaF, 1);
    leftMidState = MidState(LeftState, tempMidState, GaF, -1);
// verify the results: rightMidState and leftMidState
    if (IsTestMode)
    {
        printf("rightMidState:\n");
        str = "  rhoStar, uStar, pStar, cStar: ";
        for (int i = 0; i < rightMidState.state.size() ; ++i)
            str.append(to_string(rightMidState.state[i]) + "  ");
        printf("%s\n", str.c_str());

        str = "  speed set: ";
        for (int i = 0; i < rightMidState.spSet.size() ; ++i)
            str.append(to_string(rightMidState.spSet[i]) + "  ");
        printf("%s\n", str.c_str());

        printf("leftMidState:\n");
        str = "  rhoStar, uStar, pStar, cStar: ";
        for (int i = 0; i < leftMidState.state.size() ; ++i)
            str.append(to_string(leftMidState.state[i]) + "  ");
        printf("%s\n", str.c_str());

        str = "  speed set: ";
        for (int i = 0; i < leftMidState.spSet.size() ; ++i)
            str.append(to_string(leftMidState.spSet[i]) + "  ");
        printf("%s\n", str.c_str());
    }
    t = dt;

    while (t <= EndTime)
    {
//       update data on the left side
        dist_0 = leftMidState.spSet[0] * t + x[halfGrid];
        dist_1 = leftMidState.spSet[1] * t + x[halfGrid];
        dist_2 = leftMidState.spSet[2] * t + x[halfGrid];
        for (int j = halfGrid - 1 ; j >= 0  ; --j)
        {
            if (x[j] > dist_0)
            {
                for (int i = 0; i < NComponents ; ++i)
                    U[i][j] = rightMidState.state[i];
            }else if ( dist_0 >= x[j] && x[j] > dist_1 )
            {
                for (int i = 0; i < NComponents ; ++i)
                    U[i][j] = leftMidState.state[i];
            }else if ( dist_1 >= x[j] && x[j] > dist_2 )
            {
                tempVel = (x[j] - x[halfGrid]) / t;
                temp = GaF.twoOverGaP1 + GaF.gaM1OverGaP1 * (LeftState[1] - tempVel) / LeftState[3];
                U[0][j] = LeftState[0] * pow(temp, GaF.twoOverGaM1);
                U[1][j] = GaF.twoOverGaP1 * (LeftState[3] + GaF.gaM1Over2 * LeftState[1] + tempVel);
                U[2][j] = LeftState[2] * pow(temp, GaF.twoGaOverGaM1);
            }else
                break;
        }

//       update data on the right side
        dist_0 = rightMidState.spSet[0] * t + x[halfGrid];
        dist_1 = rightMidState.spSet[1] * t + x[halfGrid];
        dist_2 = rightMidState.spSet[2] * t + x[halfGrid];
        for (int j = halfGrid ; j < NGrid  ; ++j)
        {
            if (x[j] < dist_0)
            {
                for (int i = 0; i < NComponents ; ++i)
                    U[i][j] = leftMidState.state[i];
            }else if ( dist_0 <= x[j] && x[j] < dist_1 )
            {
                for (int i = 0; i < NComponents ; ++i)
                    U[i][j] = rightMidState.state[i];
            }else if ( dist_1 <= x[j] && x[j] < dist_2 )
            {
                tempVel = (x[j] - x[halfGrid]) / t;
                temp = GaF.twoOverGaP1 - GaF.gaM1OverGaP1 * (RightState[1] - tempVel) / RightState[3];
                U[0][j] = RightState[0] * pow(temp, GaF.twoOverGaM1);
                U[1][j] = GaF.twoOverGaP1 * (-RightState[3] + GaF.gaM1Over2 * RightState[1] + tempVel);
                U[2][j] = RightState[2] * pow(temp, GaF.twoGaOverGaM1);
            }else
                break;
        }

        t += dt;
    }

    clock_t tEnd = clock();         // finish the clock

/********* save U *********/
    t = (double)(tEnd - tStart)/CLOCKS_PER_SEC;         // execution time

// verify the results: t and U
    if (IsTestMode)
    {
        printf("ExecutionTime: %.5f\n", t);
        str = "";
        str.append("U:\n");
        for (int i = 0; i < NComponents; ++i){
            for (int j = 0; j < NGrid; ++j)
                str.append(to_string(U[i][j]) + "  ");
            str.append("\n");
        }

        printf("%s\n", str.c_str());
    }

    str = "# This table provides a solution of the Riemann problem at t=";
    str.append(to_string(EndTime) + "\n# with (NGrid, TimeStep, ExecutionTime) = (" +
               to_string(NGrid) + "," + to_string(TimeStep) + "," + to_string(t) + ")\n" +
               "# Initial condition of the strong shock problem:\n" +
               "#   left  state: (rho, Vx, P) = (" + argv6 + ")\n" +
               "#   right state: (rho, Vx, P) = (" + argv7 + ")\n\n");
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
// compute the pStar from the Bisection Method
// -----------------------------------------------------------
double CompPStarBisection(const double gamma, const vector<double> &leftState, const vector<double> &rightState)
{
    double pStar, pStar0, compuF, compuF_L, compuF_R;
    double p_L0 = leftState[2], p_R0 = rightState[2], p_L=0, p_R= (p_L0+p_R0)/2 * 5;

    while(true)
    {
        pStar0 = (p_L+p_R)/2;
        pStar = pStar0;
        if (p_L == pStar || p_R == pStar) break;
		compuF = ComputeF(gamma, pStar, leftState, rightState);
        compuF_L = ComputeF(gamma, p_L, leftState, rightState);
        compuF_R = ComputeF(gamma, p_R, leftState, rightState);
		if (compuF == 0) break;
		if (compuF_L * compuF < 0) {p_R = pStar; compuF_R = compuF;}
		else {p_L = pStar0; compuF_L = compuF;}
    }
    return pStar; 
}

// -----------------------------------------------------------
// compute F(pStar, gamma, LeftState, RightState)
//         = f(pStar, gamma, p_L, rho_L) +  f(pStar, gamma, p_R, rho_R) - (u_L - u_R)
// -----------------------------------------------------------
double ComputeF(const double gamma, const double pStar, 
         const vector<double> &leftState, const vector<double> &rightState)
{   
    double rho_L = leftState[0], rho_R = rightState[0];
    double u_L = leftState[1], u_R = rightState[1];
    double p_L = leftState[2], p_R = rightState[2];
    double c_L = leftState[3], c_R = rightState[3];
    double f_L, f_R, y;

    if( pStar >= p_L )
    {
        f_L = (pStar - p_L)/rho_L/c_L/sqrt( (gamma+1)/2/gamma *(pStar/p_L) + (gamma-1)/2/gamma);
    }
    if( pStar < p_L )
    {
        f_L = 2*c_L/(gamma-1)*(pow((pStar/p_L),((gamma-1)/2/gamma)) - 1);
    }
    if( pStar >= p_R )
    {
        f_R = (pStar - p_R)/rho_R/c_R/sqrt( (gamma+1)/2/gamma *(pStar/p_R) + (gamma-1)/2/gamma);
    }
    if( pStar < p_R )
    {
        f_R = 2*c_R/(gamma-1)*(pow((pStar/p_R),((gamma-1)/2/gamma)) - 1);
    }
    y = f_L + f_R + u_R - u_L;

    return y;
}

// -----------------------------------------------------------
// compute f(pStar, gamma, p, rho)
// -----------------------------------------------------------
double Computef(const GammaFacts &GaF, const double pStar, const vector<double> &state)
{
//  state = [rho, u, P, c]
    if (pStar >= state[2]) // pStar >= pSide => shock wave
//      f = (pStar - P)/(rho*c*sqrt((gamma + 1)*pStar/P/2/gamma + (gamma - 1)/2/gamma ))
        return (pStar - state[2]) / (state[0] * state[3] * sqrt(GaF.gaP1Over2Ga * pStar / state[2] + GaF.gaM1Over2Ga));
    else                   // pStar < pSide => rarefaction wave
//      f = 2*c*((pStar/P)^((gamma-1)/2/gamma) - 1)/(gamma - 1)
        return 2 * state[3] * (pow(pStar/state[2], GaF.gaM1Over2Ga) - 1) / GaF.gaM1;
}

// -----------------------------------------------------------
// save data
// -----------------------------------------------------------
void SaveData(const int NGrid, string &headerStr, const vector<double> &x, const vector<vector<double>> &U)
{
    headerStr.append("#   \tr   \t\t\t\tRho   \t\t\t\tVx   \t\t\t\tVy   \t\t\t\tVz   \t\t\t\tPres\n");
    //printf("%s", headerStr.c_str());

//  create the file of results and mark the NGrid
    string FileName = "results_" + to_string(NGrid) + ".txt";
    //sprintf( FileName, "results_%d.txt", NGrid );
    FILE *File = fopen( FileName.c_str(), "w" );

    fprintf( File, "%s", headerStr.c_str());
//   U_i = [rho_i, Vx_i, P_i]^T, each row with data = [x, v_x, v_y, v_z, P]
    for (int i = 0; i < NGrid; ++i)
        fprintf( File, "%.7f  %.13e  %.13e  %.13e  %.13e  %.13e\n",
                 x[i], U[0][i], U[1][i], 0.0, 0.0, U[2][i] );

    fclose( File );
}