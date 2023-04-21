#include <cxsom-rules.hpp>
#include <cxsom-builder.hpp>
using namespace cxsom::rules;
context* cxsom::rules::ctx = nullptr;

#define CACHE 2
#define TRACE 250000
#define FORGET 0
#define BETA 0.5
#define SIGMA_CONV 0.01
#define MAP_SIZE 500
#define WALLTIME -1
#define DEADLINE 1000
#define OPENED true
#define FROZEN_TRACE 20000


void build_relax_step(int walltime, int timestep, std::string test_name, cxsom::builder::Map& map){
  name_space ns(map.map_name);
  {
    timeline t(std::string("out-rlx")+std::string(test_name));
    kwd::type("Ae", "Map1D<Scalar>="+std::to_string(map.map_size), CACHE, FORGET, false);
    kwd::type("Input" , "Scalar", CACHE, TRACE, false);
    kwd::type("Am ", std::string("Map1D<Scalar>=")+std::to_string(map.map_size), CACHE, TRACE, false);
    "Input" << fx::copy(kwd::prev("Input"))| kwd::use("walltime", walltime);
    //2 - Calcul de l'ae,ac,am pour chacun
    "Ae"<< fx::match_gaussian("Input", kwd::at(kwd::ith(kwd::var("wgt","We"),0),timestep)) | map.p_external;
    kwd::type("P", "Pos1D", CACHE, TRACE, false);
    //Input has a fixed value for all the tests.
    //std::cout<<input<<std::endl;

    kwd::type("Ac", std::string("Map1D<Scalar>=")+std::to_string(map.map_size), CACHE, FORGET, false);
    "Ac"<< fx::match_gaussian("P", kwd::at(kwd::ith(kwd::var("wgt","Wc"),0),timestep)) | map.p_contextual;
    "Am"<< fx::merge("Ae", "Ac")| kwd::use("walltime", walltime), kwd::use("epsilon", 1e-3), kwd::use("beta",  BETA );
    kwd::type("BMU", "Pos1D", CACHE, TRACE, false);
    "BMU" << map.argmax("Am") | kwd::use("walltime", walltime), kwd::use("epsilon", 1e-3), kwd::use("sigma", SIGMA_CONV);
  }
}
