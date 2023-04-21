
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
#define MAP_SIZE 500
#define WALLTIME -1
#define DEADLINE 2
#define OPENED true
#define FROZEN_TRACE 20000

auto make_architecture(int nmaps){
  kwd::parameters p_main, p_match, p_learn, p_learn_e, p_learn_c1,p_learn_c2, p_external, p_contextual, p_global;
  p_main       | kwd::use("walltime", WALLTIME), kwd::use("epsilon", 0);
  p_match      | p_main,  kwd::use("sigma", .1);
  p_learn      | p_main,  kwd::use("alpha", .1);
  p_learn_e    | p_learn, kwd::use("r", .2 );
  p_learn_c1    | p_learn, kwd::use("r", .02);
  p_learn_c2    | p_learn, kwd::use("r", .02);
  p_external   | p_main;
  p_contextual | p_main;
  p_global     | p_main,  kwd::use("random-bmu", 1), kwd::use("beta", .5), kwd::use("delta", .1), kwd::use("deadline", DEADLINE);

  auto map_settings = cxsom::builder::map::make_settings();
  map_settings.map_size      = MAP_SIZE;
  map_settings.cache_size    = CACHE;
  map_settings.file_size     = TRACE;
  map_settings.kept_opened   = OPENED;
  map_settings               = {p_external, p_contextual, p_global};
  map_settings.argmax        = fx::argmax;
  map_settings.toward_argmax = std::bind(fx::argmax,std::placeholders::_1);
  //map_settings.toward_argmax = fx::toward_argmax;
  auto archi = cxsom::builder::architecture();
  if(nmaps==2){
  auto X = cxsom::builder::variable("in", cxsom::builder::name("I1"), "Scalar", CACHE, TRACE, OPENED);
  auto Y = cxsom::builder::variable("in", cxsom::builder::name("I2"), "Scalar", CACHE, TRACE, OPENED);
  X->definition();
  Y->definition();
  auto Xmap = cxsom::builder::map::make_1D("M1");
  auto Ymap = cxsom::builder::map::make_1D("M2");

  Xmap->external  (X,    fx::match_gaussian, p_match, fx::learn_triangle, p_learn_e);
  Ymap->external  (Y,    fx::match_gaussian, p_match, fx::learn_triangle, p_learn_e);
  archi << Xmap << Ymap;
}else if(nmaps==3){
  auto X = cxsom::builder::variable("in", cxsom::builder::name("I1"), "Scalar", CACHE, TRACE, OPENED);
  auto Y = cxsom::builder::variable("in", cxsom::builder::name("I2"), "Scalar", CACHE, TRACE, OPENED);
  auto Z = cxsom::builder::variable("in", cxsom::builder::name("I3"), "Scalar", CACHE, TRACE, OPENED);
  X->definition();
  Y->definition();
  Z->definition();
  auto Xmap = cxsom::builder::map::make_1D("M1");
  auto Ymap = cxsom::builder::map::make_1D("M2");
  auto Zmap = cxsom::builder::map::make_1D("M3");

  Xmap->external  (X,    fx::match_gaussian, p_match, fx::learn_triangle, p_learn_e);
  Ymap->external  (Y,    fx::match_gaussian, p_match, fx::learn_triangle, p_learn_e);
  Zmap->external  (Z,    fx::match_gaussian, p_match, fx::learn_triangle, p_learn_e);
  archi << Xmap << Ymap<<Zmap;
}
*archi = map_settings;
archi->relax_count = "Cvg";

return archi;

}


int main(int argc, char* argv[]){
context c(argc, argv);

//std::string config_file = "configs/2som.config";

if(c.user_argv[0]== "main"){
  int nmaps = stoi(c.user_argv[1]);
  auto archi = make_architecture(nmaps);
  archi->realize();
  for(auto map : archi->maps) map->internals_random_at(0);
}

else if(c.user_argv[0]=="frozen"){
  std::string analysis_prefix = c.user_argv[1];
  std::cout<<analysis_prefix;
  unsigned int timestep = stoi(c.user_argv[2]);
  int nmaps = stoi(c.user_argv[3]);
  auto archi = make_architecture(nmaps);

  auto X = cxsom::builder::variable(analysis_prefix + "-in", cxsom::builder::name("I1"), "Scalar", 1, FROZEN_TRACE, OPENED);
  auto Y = cxsom::builder::variable(analysis_prefix + "-in", cxsom::builder::name("I2"), "Scalar", 1, FROZEN_TRACE, OPENED);
  auto Z = cxsom::builder::variable(analysis_prefix + "-in", cxsom::builder::name("I3"), "Scalar", 1, FROZEN_TRACE, OPENED);
  auto U = cxsom::builder::variable(analysis_prefix + "-in", cxsom::builder::name("U"), "Scalar", 1, FROZEN_TRACE, OPENED);
  X->definition();
  Y->definition();
  Z->definition();

  //U->definition();

  archi->frozen({analysis_prefix,analysis_prefix, CACHE, FROZEN_TRACE, false, timestep});

  kwd::rename(kwd::var(analysis_prefix+"-in", "I1"),kwd::var("ztest-in", "I1"));
  kwd::rename(kwd::var(analysis_prefix+"-in", "I2"),kwd::var("ztest-in", "I2"));
  if(nmaps==3) kwd::rename(kwd::var(analysis_prefix+"-in", "I3"),kwd::var("ztest-in", "I3"));
}

return 0;

}
