
#include <cxsom-builder.hpp>
#include <string>

using namespace cxsom::rules;
context* cxsom::rules::ctx = nullptr;
#define PI 3.14159
#define CACHE 2
#define TRACE 250000
#define FORGET 0
#define BETA 0.5
#define SIGMA_CONV 0.01
#define MAP_SIZE 1000
#define WALLTIME -1
#define DEADLINE 200
#define OPENED true
#define FROZEN_TRACE 20000
#define ACTIVITY_TRACE 1000
#define PARTIAL_RECORD true
#define RE 0.2
#define RC 0.02

//Create the 4 maps architecture for drone control

auto make_architecture(){

  //parameters definition
  kwd::parameters p_main, p_match, p_learn, p_learn_e, p_learn_c1,p_learn_c2, p_external, p_contextual, p_global;
  p_main       | kwd::use("walltime", WALLTIME), kwd::use("epsilon", 0);
  p_match      | p_main,  kwd::use("sigma", .1);
  p_learn      | p_main,  kwd::use("alpha", .1);
  p_learn_e    | p_learn, kwd::use("r", RE );
  p_learn_c1    | p_learn, kwd::use("r", RC);
  p_learn_c2    | p_learn, kwd::use("r", RC);
  p_external   | p_main;
  p_contextual | p_main;
  p_global     | p_main,  kwd::use("random-bmu", 1), kwd::use("beta", BETA), kwd::use("delta", .1), kwd::use("deadline", DEADLINE);

  auto map_settings = cxsom::builder::map::make_settings();
  map_settings.map_size      = MAP_SIZE;
  map_settings.cache_size    = CACHE;
  map_settings.weights_file_size     = TRACE;
  map_settings.kept_opened   = OPENED;
  map_settings               = {p_external, p_contextual, p_global};
  map_settings.argmax        = fx::argmax;
  map_settings.toward_argmax = fx::toward_argmax;
  map_settings.internals_file_size   = ACTIVITY_TRACE;
  auto archi = cxsom::builder::architecture();

 //inputs definition
  auto CMD = cxsom::builder::variable("in", cxsom::builder::name("cmdy"), "Scalar", CACHE, TRACE, OPENED);
  auto PHI = cxsom::builder::variable("in", cxsom::builder::name("ct"), "Scalar", CACHE, TRACE, OPENED);
  auto V = cxsom::builder::variable("in", cxsom::builder::name("odomy"), "Scalar", CACHE, TRACE, OPENED);
  auto X = cxsom::builder::variable("in", cxsom::builder::name("vpx"), "Scalar", CACHE, TRACE, OPENED);
  CMD->definition();
  PHI->definition();
  V->definition();
  X->definition();

  //Architecture definition
  auto CMD_map = cxsom::builder::map::make_1D("M1");
  auto PHI_map = cxsom::builder::map::make_1D("M2");
  auto V_map = cxsom::builder::map::make_1D("M3");
  auto X_map = cxsom::builder::map::make_1D("M4");

  CMD_map->external  (CMD,    fx::match_gaussian, p_match, fx::learn_triangle, p_learn_e);
  CMD_map->contextual(PHI_map, fx::match_gaussian, p_match, fx::learn_triangle, p_learn_c1);
  CMD_map->contextual(V_map, fx::match_gaussian, p_match, fx::learn_triangle, p_learn_c2);
  CMD_map->contextual(X_map, fx::match_gaussian, p_match, fx::learn_triangle, p_learn_c2);

  PHI_map->external  (PHI,    fx::match_gaussian, p_match, fx::learn_triangle, p_learn_e);
  PHI_map->contextual(CMD_map, fx::match_gaussian, p_match, fx::learn_triangle, p_learn_c1);
  PHI_map->contextual(V_map, fx::match_gaussian, p_match, fx::learn_triangle, p_learn_c2);
  PHI_map->contextual(X_map, fx::match_gaussian, p_match, fx::learn_triangle, p_learn_c2);


  V_map->external  (V,    fx::match_gaussian, p_match, fx::learn_triangle, p_learn_e);
  V_map->contextual(PHI_map, fx::match_gaussian, p_match, fx::learn_triangle, p_learn_c1);
  V_map->contextual(CMD_map, fx::match_gaussian, p_match, fx::learn_triangle, p_learn_c2);
  V_map->contextual(X_map, fx::match_gaussian, p_match, fx::learn_triangle, p_learn_c2);

  X_map->external  (X,    fx::match_gaussian, p_match, fx::learn_triangle, p_learn_e);
  X_map->contextual(PHI_map, fx::match_gaussian, p_match, fx::learn_triangle, p_learn_c1);
  X_map->contextual(CMD_map, fx::match_gaussian, p_match, fx::learn_triangle, p_learn_c2);
  X_map->contextual(V_map, fx::match_gaussian, p_match, fx::learn_triangle, p_learn_c2);


  archi << CMD_map << PHI_map<<V_map<<X_map;


  *archi = map_settings;
  archi->relax_count = "Cvg";

  return archi;

}

//creation of the same architecture as make_architecture, with map CMD (command) being closed.
auto make_closed_architecture(){
  //parameters
  kwd::parameters p_main, p_match, p_learn, p_learn_e, p_learn_c1,p_learn_c2, p_external, p_contextual, p_global;
  p_main       | kwd::use("walltime", WALLTIME), kwd::use("epsilon", 0);
  p_match      | p_main,  kwd::use("sigma", .1);
  p_learn      | p_main,  kwd::use("alpha", .1);
  p_learn_e    | p_learn, kwd::use("r", RE );
  p_learn_c1    | p_learn, kwd::use("r", RC);
  p_learn_c2    | p_learn, kwd::use("r", RC);
  p_external   | p_main;
  p_contextual | p_main;
  p_global     | p_main,  kwd::use("random-bmu", 1), kwd::use("beta", BETA), kwd::use("delta", .1), kwd::use("deadline", DEADLINE);

  auto map_settings = cxsom::builder::map::make_settings();
  map_settings.map_size      = MAP_SIZE;
  map_settings.cache_size    = CACHE;
  map_settings.weights_file_size     = TRACE;
  map_settings.kept_opened   = OPENED;
  map_settings               = {p_external, p_contextual, p_global};
  map_settings.argmax        = fx::argmax;
  map_settings.toward_argmax = fx::toward_argmax;
  map_settings.internals_file_size   = ACTIVITY_TRACE;
  auto archi = cxsom::builder::architecture();


  //inputs
  auto CMD = cxsom::builder::variable("in", cxsom::builder::name("cmdy"), "Scalar", CACHE, TRACE, OPENED);
  auto PHI = cxsom::builder::variable("in", cxsom::builder::name("ct"), "Scalar", CACHE, TRACE, OPENED);
  auto V = cxsom::builder::variable("in", cxsom::builder::name("odomy"), "Scalar", CACHE, TRACE, OPENED);
  auto X = cxsom::builder::variable("in", cxsom::builder::name("vpx"), "Scalar", CACHE, TRACE, OPENED);
  CMD->definition();
  PHI->definition();
  V->definition();
  X->definition();

  auto CMD_map = cxsom::builder::map::make_1D("M1");
  auto PHI_map = cxsom::builder::map::make_1D("M2");
  auto V_map = cxsom::builder::map::make_1D("M3");
  auto X_map = cxsom::builder::map::make_1D("M4");

  //CMD_map->external  (CMD,    fx::match_gaussian, p_match, fx::learn_triangle, p_learn_e); // Pas de poids externe pour CMD !
  CMD_map->contextual(PHI_map, fx::match_gaussian, p_match, fx::learn_triangle, p_learn_c1);
  CMD_map->contextual(V_map, fx::match_gaussian, p_match, fx::learn_triangle, p_learn_c2);
  CMD_map->contextual(X_map, fx::match_gaussian, p_match, fx::learn_triangle, p_learn_c2);

  PHI_map->external  (PHI,    fx::match_gaussian, p_match, fx::learn_triangle, p_learn_e);
  PHI_map->contextual(CMD_map, fx::match_gaussian, p_match, fx::learn_triangle, p_learn_c1);
  PHI_map->contextual(V_map, fx::match_gaussian, p_match, fx::learn_triangle, p_learn_c2);
  PHI_map->contextual(X_map, fx::match_gaussian, p_match, fx::learn_triangle, p_learn_c2);


  V_map->external  (V,    fx::match_gaussian, p_match, fx::learn_triangle, p_learn_e);
  V_map->contextual(PHI_map, fx::match_gaussian, p_match, fx::learn_triangle, p_learn_c1);
  V_map->contextual(CMD_map, fx::match_gaussian, p_match, fx::learn_triangle, p_learn_c2);
  V_map->contextual(X_map, fx::match_gaussian, p_match, fx::learn_triangle, p_learn_c2);

  X_map->external  (X,    fx::match_gaussian, p_match, fx::learn_triangle, p_learn_e);
  X_map->contextual(PHI_map, fx::match_gaussian, p_match, fx::learn_triangle, p_learn_c1);
  X_map->contextual(CMD_map, fx::match_gaussian, p_match, fx::learn_triangle, p_learn_c2);
  X_map->contextual(V_map, fx::match_gaussian, p_match, fx::learn_triangle, p_learn_c2);


  archi << CMD_map << PHI_map<<V_map<<X_map;


  *archi = map_settings;
  archi->relax_count = "Cvg";

  return archi;
}


int main(int argc, char* argv[]){
context c(argc, argv);

//learning option
if(c.user_argv[0]== "main"){
  auto archi = make_architecture();
  archi->realize();
  for(auto map : archi->maps) map->internals_random_at(0);
}

//frozen : if you have test inputs ztest-in.
else if(c.user_argv[0]=="frozen"){

  std::string analysis_prefix = c.user_argv[1];
  std::cout<<analysis_prefix;
  unsigned int timestep = stoi(c.user_argv[2]);

  auto archi = make_architecture();

  auto CMD = cxsom::builder::variable(analysis_prefix + "-in", cxsom::builder::name("cmdy"), "Scalar", CACHE, TRACE, OPENED);
  auto PHI = cxsom::builder::variable(analysis_prefix + "-in", cxsom::builder::name("ct"), "Scalar", CACHE, TRACE, OPENED);
  auto V = cxsom::builder::variable(analysis_prefix + "-in", cxsom::builder::name("odomy"), "Scalar", CACHE, TRACE, OPENED);
  auto X = cxsom::builder::variable(analysis_prefix + "-in", cxsom::builder::name("vpx"), "Scalar", CACHE, TRACE, OPENED);

  CMD->definition();
  PHI->definition();
  V->definition();
  X->definition();

  archi->frozen({analysis_prefix,analysis_prefix, CACHE, FROZEN_TRACE, true, timestep, PARTIAL_RECORD});

  kwd::rename(kwd::var(analysis_prefix+"-in", "cmdy"),kwd::var("ztest-in", "cmdy"));
  kwd::rename(kwd::var(analysis_prefix+"-in", "ct"),kwd::var("ztest-in", "ct"));
  kwd::rename(kwd::var(analysis_prefix+"-in", "odomy"),kwd::var("ztest-in", "odomy"));
  kwd::rename(kwd::var(analysis_prefix+"-in", "vpx"),kwd::var("ztest-in", "vpx"));

}

//command prediction
else if(c.user_argv[0] == "closed"){

  std::string analysis_prefix = c.user_argv[1];
  std::cout<<analysis_prefix;

  unsigned int timestep = stoi(c.user_argv[2]);
  auto archi = make_closed_architecture();

  auto CMD = cxsom::builder::variable(analysis_prefix + "-in", cxsom::builder::name("cmdy"), "Scalar", CACHE, TRACE, OPENED);
  auto PHI = cxsom::builder::variable(analysis_prefix + "-in", cxsom::builder::name("ct"), "Scalar", CACHE, TRACE, OPENED);
  auto V = cxsom::builder::variable(analysis_prefix + "-in", cxsom::builder::name("odomy"), "Scalar", CACHE, TRACE, OPENED);
  auto X = cxsom::builder::variable(analysis_prefix + "-in", cxsom::builder::name("vpx"), "Scalar", CACHE, TRACE, OPENED);
  CMD->definition();
  PHI->definition();
  V->definition();
  X->definition();

  archi->frozen({analysis_prefix,analysis_prefix, CACHE, FROZEN_TRACE, false, timestep,PARTIAL_RECORD});

  kwd::rename(kwd::var(analysis_prefix+"-in", "cmdy"),kwd::var("ztest-in", "cmdy"));
  kwd::rename(kwd::var(analysis_prefix+"-in", "ct"),kwd::var("ztest-in", "ct"));
  kwd::rename(kwd::var(analysis_prefix+"-in", "odomy"),kwd::var("ztest-in", "odomy"));
  kwd::rename(kwd::var(analysis_prefix+"-in", "vpx"),kwd::var("ztest-in", "vpx"));
}

return 0;

}
