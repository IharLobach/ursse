#include <complex>
#include <cstdlib>
#include <iostream>
#include <map>
#include <string>
#include <cmath>
#include <iomanip>
#include <ctime>
#include <stdio.h>
#include <stdlib.h> /* atoi */
#include <random>
#define BOOST_PYTHON_MAX_ARITY 17
#include <boost/python.hpp>
#include <boost/python/tuple.hpp>
#include <boost/python/numpy.hpp>
#include <boost/math/special_functions/sinc.hpp>
#include "coefs.h"
namespace p = boost::python;
namespace np = boost::python::numpy;

double gamma_to_nav = 0.0661969 ;
double x3 = 0.9132260271183847;
double x4 = 2.4444485538746025480;
double x5 = 9.3830728608909477079;
double x6 = 33.122936966163038145;
double c = 2.9979e10;
double hbar_c = 0.1973269804e-4;
double rho = 70;

// std::random_device rd;
// std::mt19937 gen(rd());


// std::default_random_engine generator;
// std::uniform_real_distribution<double> distribution(0.0, 1.0);

std::default_random_engine generator;


np::ndarray RandomEnergyGammaDistribution(np::ndarray params)
{
    const double *prms = reinterpret_cast<double *>(params.get_data());
    double k = (double)(prms[0]);
    double theta = (double) (prms[1]);
    int size = (int)(prms[2]);
    int seed = (int)(prms[3]);
    std::gamma_distribution<double> dist(k, theta);
    std::srand(seed);
    std::vector<double> rand_res(size);
    for(int i=0;i<size;i++){
        rand_res[i] = dist(generator);
    }
    Py_intptr_t sh[1] = {rand_res.size()};
    np::ndarray result = np::zeros(1, sh, np::dtype::get_builtin<double>());
    std::copy(rand_res.begin(), rand_res.end(), reinterpret_cast<double *>(result.get_data()));
    return result;
}

double Cheb(double a, double b, double* C, int n, double x){
    double y = (2*x-a-b)/(b-a);
    double y2 = 2*y;
    double d = 0;
    double dd = 0;
    int j = n-1;
    while (j>=1){
        double sv = d;
        d = y2*d-dd+C[j];
        dd = sv;
        j = j-1;
    }
    return y*d-dd+0.5*C[0];
}

double InvSynchFractInt(double x){
    if (x < 0.7){
        return pow(x, 3) * Cheb(0, 0.7, C1, n1, x);
    }
    if (x < x3){
        return Cheb(0.7, x3, C2, n2, x);
    }
    if (x < (1-0.0000841363)){
        double y = -log(1-x);
        return y*Cheb(x4, x5, C3, n3, y);
    }
    if (x <= 1){
        double y = -log(1-x);
        return y*Cheb(x5, x6, C4, n4, y);
    }
    else{
        return -1;
    }
}


p::dict get_simulated_revolution_delay_data(np::ndarray params, np::ndarray revolutions)
{
    const double *prms = reinterpret_cast<double *>(params.get_data());
    const int64_t *revs = reinterpret_cast<int64_t *>(revolutions.get_data());
    double gamma = prms[0];
    double alpha = prms[1];
    double V = prms[2];
    double f = prms[3];
    int h = (int)(prms[4]);
    double JE = prms[5];
    double k = prms[6];
    double theta = prms[7];
    double phi0 = prms[8];
    double delta0 = prms[9];
    int seed = (int)(prms[10]);

    double E0 = gamma * 511000;
    double v0 = V / E0;
    double eta = alpha - 1 / pow(gamma, 2);
    double w = 2 * M_PI * h * eta;
    double eav = k * theta;
    std::gamma_distribution<double> distribution(k, theta);
    int64_t npoints = revolutions.get_shape()[0];
    std::vector<double> phis(npoints);
    std::vector<double> deltas(npoints);
    int64_t nper = revs[npoints-1]+1;

    std::srand(seed);
    double p_prev = phi0;
    double d_prev = delta0;
    int64_t rev_idx = 0;
    int64_t cur_rev = revs[rev_idx];
    double e, d_new, p_new;
    for (int i = 0; i < nper; i++)
    {
        if (i == cur_rev){
            phis[rev_idx] = p_prev;
            deltas[rev_idx] = d_prev;
            rev_idx++;
            cur_rev = revs[rev_idx]; 
        }
        e = distribution(generator);
        d_new = d_prev + v0 * sin(p_prev);
        d_new = d_new - (e - eav * (1 - JE * d_new)) / E0;
        p_new = p_prev - w * d_new;
        d_prev = d_new;
        p_prev = p_new;
    }

    Py_intptr_t sh[1] = {phis.size()};
    np::ndarray result_phis = np::zeros(1, sh, np::dtype::get_builtin<double>());
    std::copy(phis.begin(), phis.end(), reinterpret_cast<double *>(result_phis.get_data()));
    np::ndarray result_deltas = np::zeros(1, sh, np::dtype::get_builtin<double>());
    std::copy(deltas.begin(), deltas.end(), reinterpret_cast<double *>(result_deltas.get_data()));

    p::dict d;
    d["phi"] = result_phis;
    d["delta"] = result_deltas;

    return d;
}

BOOST_PYTHON_MODULE(sync_motion_sim_cpp)
{
    using namespace boost::python;
    Py_Initialize();
    np::initialize();
    def("get_simulated_revolution_delay_data", get_simulated_revolution_delay_data);
    def("InvSynchFractInt", InvSynchFractInt);
    def("RandomEnergyGammaDistribution", RandomEnergyGammaDistribution);
}