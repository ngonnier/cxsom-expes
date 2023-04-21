#include <cxsom-builder.hpp>
#include <fstream>
#include <sstream>


#define CACHE              2
#define TRACE         100000
#define FROZEN_TRACE    1000
#define OPENED          true
#define OPEN_AS_NEEDED false
#define FORGET             0
#define FOREVER           -1 // Infinite walltime
#define DEADLINE         1000

#define MAP_SIZE         50

// cxsom declarations
using namespace cxsom::rules;
context* cxsom::rules::ctx = nullptr;

auto make_architecture(int nmaps) {

  auto archi = cxsom::builder::architecture();

  kwd::parameters p_main, p_match, p_learn, p_learn_e, p_learn_c, p_external, p_contextual, p_global;
  p_main       | kwd::use("walltime", FOREVER), kwd::use("epsilon", 0);
  p_match      | p_main,  kwd::use("sigma", .2);
  p_learn      | p_main,  kwd::use("alpha", .05);
  p_learn_e    | p_learn, kwd::use("r", .2 );
  p_learn_c    | p_learn, kwd::use("r", .06);
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
  map_settings.toward_argmax = fx::toward_argmax;

  if(nmaps==2){
  auto Xmap = cxsom::builder::map::make_2D("M1");
  auto Ymap = cxsom::builder::map::make_2D("M2");

  auto X = cxsom::builder::variable("in", cxsom::builder::name("I1"), "Scalar", CACHE, TRACE, OPENED);
  auto Y = cxsom::builder::variable("in", cxsom::builder::name("I2"), "Scalar", CACHE, TRACE, OPENED);

  Xmap->external  (X,    fx::match_gaussian, p_match, fx::learn_triangle, p_learn_e);
  Xmap->contextual(Ymap, fx::match_gaussian, p_match, fx::learn_triangle, p_learn_c);
  Ymap->external  (Y,    fx::match_gaussian, p_match, fx::learn_triangle, p_learn_e);
  Ymap->contextual(Xmap, fx::match_gaussian, p_match, fx::learn_triangle, p_learn_c);

  archi << Xmap << Ymap;
  *archi = map_settings;
}

else{

  auto X = cxsom::builder::variable("in", cxsom::builder::name("I1"), "Scalar", CACHE, TRACE, OPENED);
  auto Y = cxsom::builder::variable("in", cxsom::builder::name("I2"), "Scalar", CACHE, TRACE, OPENED);
  auto Z = cxsom::builder::variable("in", cxsom::builder::name("I3"), "Scalar", CACHE, TRACE, OPENED);

  auto Xmap = cxsom::builder::map::make_2D("M1");
  auto Ymap = cxsom::builder::map::make_2D("M2");
  auto Zmap = cxsom::builder::map::make_2D("M3");

  Xmap->external  (X,    fx::match_gaussian, p_match, fx::learn_triangle, p_learn_e);
  Xmap->contextual(Zmap, fx::match_gaussian, p_match, fx::learn_triangle, p_learn_c);
  Xmap->contextual(Ymap, fx::match_gaussian, p_match, fx::learn_triangle, p_learn_c);
  Ymap->external  (Y,    fx::match_gaussian, p_match, fx::learn_triangle, p_learn_e);
  Ymap->contextual(Xmap, fx::match_gaussian, p_match, fx::learn_triangle, p_learn_c);
  Ymap->contextual(Zmap, fx::match_gaussian, p_match, fx::learn_triangle, p_learn_c);
  Zmap->external  (Z,    fx::match_gaussian, p_match, fx::learn_triangle, p_learn_e);
  Zmap->contextual(Ymap, fx::match_gaussian, p_match, fx::learn_triangle, p_learn_c);
  Zmap->contextual(Xmap, fx::match_gaussian, p_match, fx::learn_triangle, p_learn_c);
  archi << Xmap << Ymap<<Zmap;
  *archi = map_settings;
}

  archi->relax_count = "Cvg";
  return archi;
}

enum class Mode : char {Main, Relax, Frozen};

int main(int argc, char* argv[]) {
  context c(argc, argv);
  Mode mode = Mode::Main;
  std::string analysis_prefix;
  unsigned int weight_time;
  int nmaps;

  // We analyse the arguments and identify the mode.
  std::ostringstream prefix;
  for(const auto& arg : c.argv) prefix << arg << ' ';
  prefix << "-- ";

  if(c.user_argv.size() == 0) {
    std::cout << "You have to provide user arguments." << std::endl
	      << "e.g:" << std::endl
	      << "  " << prefix.str() << "main <nmaps>                           <-- sends the main rules." << std::endl
	      << "  " << prefix.str() << "relax <nmaps> <timeline-prefix> <time>  <-- sends relaxation rules for weights at time." << std::endl
	      << "  " << prefix.str() << "frozen <nmaps> <timeline-prefix> <time> <-- sends 'frozen' rules for weights at time." << std::endl;
    c.notify_user_argv_error();
    return 0;
  }

  if(c.user_argv[0] == "main"){
    if(c.user_argv.size() != 2) {
      std::cout << "The 'main' mode expects 1 arguments"  << std::endl;
      c.notify_user_argv_error();
      return 0;
    }
    mode = Mode::Main;
    nmaps = stoi(c.user_argv[1]);
  }
  else if(c.user_argv[0] == "relax") {
    if(c.user_argv.size() != 4) {
      std::cout << "The 'relax' mode expects 3 arguments"  << std::endl;
      c.notify_user_argv_error();
      return 0;
    }
    mode = Mode::Relax;
    nmaps = stoi(c.user_argv[1]);
    analysis_prefix = c.user_argv[2];
    weight_time = stoul(c.user_argv[3]);
  }
  else if(c.user_argv[0] == "frozen") {
    if(c.user_argv.size() != 4) {
      std::cout << "The 'frozen' mode expects 3 arguments"  << std::endl;
      c.notify_user_argv_error();
      return 0;
    }
    mode = Mode::Frozen;
    nmaps = stoi(c.user_argv[1]);
    analysis_prefix = c.user_argv[2];
    weight_time = stoul(c.user_argv[3]);
  }

  else {
    std::cout << "Bad user arguments." << std::endl;
    c.notify_user_argv_error();
    return 0;
  }


  // Now, according to the mode, let us send rules.

  if(mode == Mode::Main) {
    auto archi = make_architecture(nmaps);
    archi->realize();
    for(auto map : archi->maps) map->internals_random_at(0);

    if(c.argv[1] == "graph" || c.argv[1] == "graph-full") {
      std::ofstream dot_file("archi.dot");
      dot_file << archi->write_dot;
    }
  }


  if(mode == Mode::Relax) {
    auto archi = make_architecture(nmaps);
    archi->expand_relax({analysis_prefix, CACHE, DEADLINE + 1, OPEN_AS_NEEDED, weight_time});
  }


  if(mode == Mode::Frozen) {
    auto archi = make_architecture(nmaps);
    auto X = cxsom::builder::variable(analysis_prefix+"-in", cxsom::builder::name("I1"), "Scalar", 1, 1, OPENED);
    auto Y = cxsom::builder::variable(analysis_prefix+"-in", cxsom::builder::name("I2"), "Scalar", 1, 1, OPENED);
    auto Z = cxsom::builder::variable(analysis_prefix+"-in", cxsom::builder::name("I3"), "Scalar", 1, 1, OPENED);

    archi->frozen({analysis_prefix, CACHE, FROZEN_TRACE, OPEN_AS_NEEDED, weight_time});
     if(nmaps==2){
    kwd::rename(kwd::var(analysis_prefix+"-in", "I1"),kwd::var("ztest-in", "I1"));
    kwd::rename(kwd::var(analysis_prefix+"-in", "I2"),kwd::var("ztest-in", "I2"));
    }else if(nmaps==3){
    kwd::rename(kwd::var(analysis_prefix+"-in", "I1"),kwd::var("ztest-in", "I1"));
    kwd::rename(kwd::var(analysis_prefix+"-in", "I2"),kwd::var("ztest-in", "I2"));
    kwd::rename(kwd::var(analysis_prefix+"-in", "I3"),kwd::var("ztest-in", "I3"));
    }
  }

  return 0;
}
