
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
#define DEADLINE 200
#define OPENED true
#define FROZEN_TRACE 20000
#define ACTIVITY_TRACE 1000
#define PARTIAL_RECORD true

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
  map_settings.weights_file_size     = TRACE;
  map_settings.internals_file_size   = ACTIVITY_TRACE;
  map_settings.kept_opened   = OPENED;
  map_settings               = {p_external, p_contextual, p_global};
  map_settings.argmax        = fx::argmax;
  //map_settings.toward_argmax = std::bind(fx::argmax,std::placeholders::_1);
  map_settings.toward_argmax = fx::toward_argmax;
  auto archi = cxsom::builder::architecture();
  std::vector<std::shared_ptr<cxsom::builder::Map>> map_list = {};

  for(int i = 0; i < nmaps ; i++){
    auto X = cxsom::builder::variable("in", cxsom::builder::name(std::string("I")+std::to_string(i+1)), "Scalar", CACHE, TRACE, OPENED);
    X->definition();
    auto Xmap = cxsom::builder::map::make_1D(std::string("M")+std::to_string(i+1));
    Xmap->external  (X,    fx::match_gaussian, p_match, fx::learn_triangle, p_learn_e);
    map_list.push_back(Xmap);
  }
  for (int i = 0; i <nmaps ; i++){
    for(int j = 0; j < nmaps; j++){
        if (j != i) map_list[i]->contextual(map_list[j], fx::match_gaussian, p_match, fx::learn_triangle, p_learn_c1);
      }
      archi <<map_list[i];
    }
*archi = map_settings;
archi->relax_count = "Cvg";
return archi;

}

auto make_closed_architecture(int nmaps,int map_closed){
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
  map_settings.weights_file_size     = TRACE;
  map_settings.kept_opened   = OPENED;
  map_settings.internals_file_size   = ACTIVITY_TRACE;
  map_settings               = {p_external, p_contextual, p_global};
  map_settings.argmax        = fx::argmax;
  //map_settings.toward_argmax = std::bind(fx::argmax,std::placeholders::_1);
  map_settings.toward_argmax = fx::toward_argmax;
  auto archi = cxsom::builder::architecture();
  std::vector<std::shared_ptr<cxsom::builder::Map>> map_list = {};
  for(int i = 0; i < nmaps ; i++){
    auto X = cxsom::builder::variable("in", cxsom::builder::name(std::string("I")+std::to_string(i+1)), "Scalar", CACHE, TRACE, OPENED);
    X->definition();
    auto Xmap = cxsom::builder::map::make_1D(std::string("M")+std::to_string(i+1));
    if (i != (map_closed - 1)) Xmap->external(X,    fx::match_gaussian, p_match, fx::learn_triangle, p_learn_e);
    map_list.push_back(Xmap);
  }
  for (int i = 0; i <nmaps ; i++){
    for(int j = 0; j < nmaps; j++){
      if (j != i) map_list[i]->contextual(map_list[j], fx::match_gaussian, p_match, fx::learn_triangle, p_learn_c1);
      }
      archi << map_list[i];
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

else if(c.user_argv[0] == "relax-traj"){
  //a lancer pour N_TRAJ
  //usage : --relax-traj analysis_prefix timestep
  std::string analysis_prefix = c.user_argv[1];
  unsigned int timestep = stoi(c.user_argv[2]);
  int nmaps = stoi(c.user_argv[3]);
  auto archi = make_architecture(nmaps);
  archi->expand_relax({std::string(),analysis_prefix, CACHE, DEADLINE + 1, false, timestep,PARTIAL_RECORD});
}

else if(c.user_argv[0]=="frozen"){
  std::string analysis_prefix = c.user_argv[1];
  std::cout<<analysis_prefix;
  unsigned int timestep = stoi(c.user_argv[2]);
  int nmaps = stoi(c.user_argv[3]);
  auto archi = make_architecture(nmaps);

  archi->frozen({analysis_prefix,analysis_prefix, CACHE, FROZEN_TRACE, false, timestep,PARTIAL_RECORD});
  for(int i = 0; i<nmaps; i++){
  kwd::rename(kwd::var(analysis_prefix+"-in", std::string("I")+std::to_string(i+1)),kwd::var("ztest-in", std::string("I") + std::to_string(i+1)));
  }
}
else if(c.user_argv[0] == "closed"){
  std::string analysis_prefix = c.user_argv[1];
  std::cout<<analysis_prefix;
  unsigned int timestep = stoi(c.user_argv[2]);
  int nmaps = stoi(c.user_argv[3]);
  int(map_closed) = stoi(c.user_argv[4]);
  auto archi = make_closed_architecture(nmaps,map_closed);
  archi->frozen({analysis_prefix,analysis_prefix, CACHE, FROZEN_TRACE, false, timestep,PARTIAL_RECORD});
  for(int i = 0; i<nmaps; i++){
    kwd::rename(kwd::var(analysis_prefix+"-in", std::string("I")+std::to_string(i+1)),kwd::var("ztest-in", std::string("I") + std::to_string(i+1)));
  }
}
return 0;
}
