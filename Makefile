# else ifeq ($(MODE),drone)
# 		RULES=$(RULESDRONE)
# 		FILE=main_drone.cpp
# Adapt this path if needed.
include /usr/share/cxsom/cxsom-makefile

RULES1D         = ./build/relax
RULES2D         = ./build/relax2D
RULESNC1D					= ./build/relaxnc
RULESNC2D					= ./build/relaxnc2D
RULESDRONE 				= ./build/drone

#1D maps or 2D maps
MODE 						= 1

ifeq ($(MODE),2)
	RULES=$(RULES2D)
	FILE = main2D.cpp
else
		RULES=$(RULES1D)
		FILE=main.cpp
endif

ifeq ($(MODE),2)
	RULESNC=$(RULESNC2D)
	FILENC = main2D_nc.cpp
else
	RULESNC=$(RULESNC1D)
	FILENC=main_nc.cpp
endif

#to be put in a config file
RELAX_PREFIX  = zrlx
FROZEN_PREFIX = zfrz
MAP_CLOSED = 1
CLOSED_PREFIX=	zclosed-$(MAP_CLOSED)

#number of relaxation trajectories to generate
NTRAJ = 10

#timestep range for send-film-rules
BEGIN=10000
END = 500000
STEP = 10000

#number of maps in the architecture

NMAPS=3

help:
	@echo
	@echo
	@echo "make cxsom-help                                    <-- help for common cxsom manipulations"
	@echo
	@echo "make inputs TYPE=<cercle,anneau,lissajoux,	plan> DIM=<dim>	N=<n samples> NTEST=<n_test>	<-- generate inputs of type TYPE, with N samples and NTEST test samples"
	@echo "make send-main-rules                               <-- sends the main rules"
	@echo "make compile																				<-- compile the cpp file in RULES"
	@echo "make send-relax-rules 								              <-- sends NTRAJ relaxation trajectories at timestep TIMESTEP"
	@echo "make send-frozen-rules TIMESTEP=<t>        	<-- sends a test phase at timestep TIMESTEP"
	@echo "make send-film-rules 															<-- sends several test phases for timesteps between BEGIN, END, with step STEP"
	@echo "make send-nc-rules 																<-- sends main rules on unconnected maps"
	@echo "make send-frozen-nc-rules 													<-- sends frozen rules for timestep TIMESTEP on unconnected maps"


	@echo "make ights             										  			<-- plot weigths for timestep"
	@echo "make plt-field               										  <-- plot BMU field"
	@echo

.PHONY: compile
compile:
	g++ -std=c++17 -o $(RULES) $(FILE) -lboost_system -lstdc++fs $$(pkg-config --cflags --libs cxsom-rules cxsom-builder)

compilenc:
	g++ -std=c++17 -o $(RULESNC)  $(FILENC) -lboost_system -lstdc++fs $$(pkg-config --cflags --libs cxsom-rules cxsom-builder)

clear:
	rm -r `cat .cxsom-rootdir-config`/${FROZEN_PREFIX}-* `cat .cxsom-rootdir-config`/${CLOSED_PREFIX}-* `cat .cxsom-rootdir-config`/rlx `cat .cxsom-rootdir-config`/wgt `cat .cxsom-rootdir-config`/out

.PHONY: inputs
inputs:
	python3 inputs/inputs_ND.py `cat .cxsom-rootdir-config` ${TYPE} ${DIM} ${NMAPS} ${N} ${NTEST}

# .PHONY: graph-main-rules
# graph-main-rules:
# 	@${RULES} graph relax -- main
#
# .PHONY: graph-relax-traj
# graph-relax-traj:
# 		for i in `seq ${NTRAJ}`;do \
# 			${RULES} graph relaxt$${i} -- relax-traj $(RELAX_PREFIX)-`printf %04d ${TIMESTEP}`-`printf %04d ${TIME_INP}`-`printf %03d $${i}` $(TIMESTEP);\
# 			dot -Tpdf relaxt$${i}-updates.dot -o relaxt$${i}-updates.pdf ; \
# 		done

.PHONY: send-main-rules
send-main-rules:
	@${RULES} send `cat .cxsom-hostname-config` `cat .cxsom-port-config` -- main $(NMAPS)

send-nc-rules:
	@${RULESNC} send `cat .cxsom-hostname-config` `cat .cxsom-port-config` -- main $(NMAPS)

send-closed-rules:
	@${RULES} send `cat .cxsom-hostname-config` `cat .cxsom-port-config` -- closed ${CLOSED_PREFIX}-`printf %04d ${TIMESTEP}` $(TIMESTEP) $(NMAPS) $(MAP_CLOSED)

# .PHONY: feed-main-inputs
# feed-main-inputs:
# 	@python3 fill_inputs.py `cat .cxsom-rootdir-config` ${N_INPUTS} ${N_TEST}
#
# .PHONY: feed-main-inputs-3D
# feed-main-inputs-3D:
# 	@python3 fill_inputs_3D.py `cat .cxsom-rootdir-config` ${N_INPUTS} ${N_TEST}


# .PHONY: send-relax-traj
# send-relax-traj:
# 	for i in `seq ${NTRAJ}`;do \
# 		${RULES} send `cat .cxsom-hostname-config` `cat .cxsom-port-config` -- relax-traj $(RELAX_PREFIX)-`printf %04d ${TIMESTEP}`-`printf %04d ${TIME_INP}`-`printf %03d $${i}` $(TIMESTEP) $(NMAPS);\
# 	done
#
# .PHONY: feed-relax-inputs
# feed-relax-inputs:
# 	for i in `seq ${NTRAJ}`;do \
# 		python3 feed_relax.py `cat .cxsom-rootdir-config` `printf %04d ${TIMESTEP}` `printf %04d ${TIME_INP}` `printf %03d $${i}`; \
# 	done

.PHONY: send-frozen-rules
send-frozen-rules:
	@${RULES} send `cat .cxsom-hostname-config` `cat .cxsom-port-config` -- frozen ${FROZEN_PREFIX}-`printf %04d ${TIMESTEP}` $(TIMESTEP) $(NMAPS)

.PHONY: send-frozen-rules
send-frozen-nc-rules:
	@${RULESNC} send `cat .cxsom-hostname-config` `cat .cxsom-port-config` -- frozen ${FROZEN_PREFIX}-`printf %04d ${TIMESTEP}` $(TIMESTEP) $(NMAPS)

.PHONY: send-film-rules
send-film-rules:
	for i in `seq ${BEGIN} ${STEP} ${END} ` ;do \
		${RULES} send `cat .cxsom-hostname-config` `cat .cxsom-port-config` -- frozen ${FROZEN_PREFIX}-`printf %04d $${i}` $${i} $(NMAPS);\
	done

.PHONY: send-film-closed-rules
send-film-closed-rules:
	for i in `seq ${BEGIN} ${STEP} ${END} ` ;do \
		${RULES} send `cat .cxsom-hostname-config` `cat .cxsom-port-config` -- closed ${CLOSED_PREFIX}-`printf %04d $${i}` $${i} $(NMAPS) $(MAP_CLOSED) ;\
	done

.PHONY: send-film-nc-rules
send-film-nc-rules:
	for i in `seq ${BEGIN} ${STEP} ${END} ` ;do \
		${RULESNC} send `cat .cxsom-hostname-config` `cat .cxsom-port-config` -- frozen ${FROZEN_PREFIX}-`printf %04d $${i}` $${i} $(NMAPS);\
	done
.PHONY: clear-relax-traj
clear-relax-traj:
	rm -r `cat .cxsom-rootdir-config`/${RELAX_PREFIX}-*



#plots
.PHONY: plot-weights
plot-weights:
	 @python3 plots/plot_weigths.py ${TIMESTEP} 0 image 1 `cat .cxsom-rootdir-config` 'I1' 'I2'

.PHONY: plot-weights
plot-weights-2D:
	@python3 ../plots/plot_weights_2D.py ${TIMESTEP} 0 image 1 `cat .cxsom-rootdir-config` 0 0 0 'I1' 'I2' 'U'

.PHONY: plot-film-weights-2D
plot-film-weights-2D:
	@python3 ../plots/plot_weights_2D.py 0 0 film 0 `cat .cxsom-rootdir-config` $(BEGIN) $(END) $(STEP) 'I1' 'I2' 'U'
.PHONY: plot-uv
plot-uv:
	@python3 ../plots/plot_weigths.py ${TIMESTEP} 0 image ${TIME_INP} `cat .cxsom-rootdir-config` 'U' 'V'

plot-u-2D:
	python3 ../plots/plot_weights_2D.py ${TIMESTEP} 0 image ${TIME_INP} `cat .cxsom-rootdir-config` 'U'


plot-u:
	python3 ../plots/plot_weigths.py ${TIMESTEP} 0 image ${TIME_INP} `cat .cxsom-rootdir-config` 'U'

.PHONY: plot-weights
plot-weights-closed:
	@python3 ../plots/plot_weigths.py ${TIMESTEP} 1 image ${TIME_INP} `cat .cxsom-rootdir-config` 'I1' 'I2' 'U'

.PHONY: plot-weights-closed-drone
plot-weights-closed-drone:
	@python3 ../plots/plot_weigths.py ${TIMESTEP} 1 image ${TIME_INP} `cat .cxsom-rootdir-config` 'cmdy' 'ct' 'odomy' 'vpx'

.PHONY: plot-error
plot-error:
	@python3 plot_error.py ${FROZEN_PREFIX}-`printf $${TIMESTEP}` ${TIMESTEP} `cat .cxsom-rootdir-config` 1 'I1' 'I2'

.PHONY: plot-error-2D
plot-error-2D:
	@python3 plot_error.py ${FROZEN_PREFIX}-`printf $${TIMESTEP}` ${TIMESTEP} `cat .cxsom-rootdir-config` 2 'I1' 'I2'

.PHONY: plot-error-closed
plot-error-closed:
	@python3 plot_error.py ${CLOSED_PREFIX}-`printf %04d $${TIMESTEP}` ${TIMESTEP} `cat .cxsom-rootdir-config` 1 'I1' 'I2' 'I3'

.PHONY: plot-error-closed-drone
plot-error-closed-drone:
	@python3 plot_error.py ${CLOSED_PREFIX}-`printf %04d $${TIMESTEP}` ${TIMESTEP} `cat .cxsom-rootdir-config` 1 'cmdy' 'ct' 'odomy' 'vpx'

.PHONY: plot-error-closed-2D
plot-error-closed-2D:
	@python3 plot-error-2D.py ${CLOSED_PREFIX}-`printf $${TIMESTEP}` ${TIMESTEP} `cat .cxsom-rootdir-config` 'I1' 'I2' 'I3'

.PHONY: view-rlx-stat
view-rlx-stat:
	@python3 view-relax-stat.py `cat .cxsom-rootdir-config` ${TIMESTEP}

plot-disto:
	@python3 ../indicateurs/distortion_maps.py `cat .cxsom-rootdir-config` ${FROZEN_PREFIX}-`printf $${TIMESTEP}` ${TIMESTEP} 0 ${M} 'M1' 'M2'
