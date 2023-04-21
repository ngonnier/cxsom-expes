help:

# Adapt this path if needed.
include /usr/share/cxsom/cxsom-makefile

RULES         = ./relax
RELAX_PREFIX  = zrlx
FROZEN_PREFIX = zfrz
NTRAJ = 10
BEGIN=200
END = 5000
STEP = 50
NMAPS=2
help:
	@echo
	@echo
	@echo "make cxsom-help                                    <-- help for common cxsom manipulations"
	@echo "make graph-main-rules                              <-- generate graphs for main rules"
	@echo "make graph-relax-rules                             <-- generate graphs for expanded relaxation rules"
	@echo
	@echo "make send-main-rules                               <-- sends the main rules"
	@echo "make send-relax-rules 								              <-- sends the relax rules"
	@echo "make send-frozen-rules              								<-- sends the frozen rules"
	@echo
	@echo "make feed-main-inputs                              <-- feeds X and Y, main and frozen"
	@echo "make feed-relax-inputs 										        <-- feeds the X,Y pair for relaxation fields and trajectories; and init BMU"
	@echo "make clear-relax-traj                          	  <-- clear all relaxation trajectories, leaves relaxation field"
	@echo "make plot-weights             										  <-- plot weigths for timestep"
	@echo "make plt-field               										  <-- plot BMU field"
	@echo

compile:
	g++ -std=c++17 -o relax relaxation.cpp -lboost_system -lstdc++fs $$(pkg-config --cflags --libs cxsom-rules cxsom-builder)

.PHONY: graph-main-rules
graph-main-rules:
	@${RULES} graph relax -- main

graph-relax-rules:
	@${RULES} send `cat .cxsom-hostname-config` `cat .cxsom-port-config` relax -- relax-field $(TIMESTEP) $(TIME_INP)
	dot -Tpdf relax-updates.dot -o relax-updates.pdf

.PHONY: graph-relax-traj
graph-relax-traj:
		for i in `seq ${NTRAJ}`;do \
			${RULES} graph relaxt$${i} -- relax-traj $(RELAX_PREFIX)-`printf %04d ${TIMESTEP}`-`printf %04d ${TIME_INP}`-`printf %03d $${i}` $(TIMESTEP);\
			dot -Tpdf relaxt$${i}-updates.dot -o relaxt$${i}-updates.pdf ; \
		done

.PHONY: send-main-rules
send-main-rules:
	@${RULES} send `cat .cxsom-hostname-config` `cat .cxsom-port-config` -- main $(NMAPS)

.PHONY: feed-main-inputs
feed-main-inputs:
	@python3 fill_inputs.py `cat .cxsom-rootdir-config` ${N_INPUTS} ${N_TEST}

.PHONY: feed-main-inputs-3D
feed-main-inputs-3D:
	@python3 fill_inputs_3D.py `cat .cxsom-rootdir-config` ${N_INPUTS} ${N_TEST}

.PHONY: send-relax-rules
send-relax-rules:
	@${RULES} send `cat .cxsom-hostname-config` `cat .cxsom-port-config` -- relax-field $(TIMESTEP) $(TIME_INP)

.PHONY: send-relax-traj
send-relax-traj:
	for i in `seq ${NTRAJ}`;do \
		${RULES} send `cat .cxsom-hostname-config` `cat .cxsom-port-config` -- relax-traj $(RELAX_PREFIX)-`printf %04d ${TIMESTEP}`-`printf %04d ${TIME_INP}`-`printf %03d $${i}` $(TIMESTEP) $(NMAPS);\
	done

.PHONY: feed-relax-inputs
feed-relax-inputs:
	for i in `seq ${NTRAJ}`;do \
		python3 feed_relax.py `cat .cxsom-rootdir-config` `printf %04d ${TIMESTEP}` `printf %04d ${TIME_INP}` `printf %03d $${i}`; \
	done

.PHONY: feed-bmu-field
feed-bmu-field:
	@python3 fill_bmus.py `cat .cxsom-rootdir-config`

.PHONY: send-frozen-rules
send-frozen-rules:
	@${RULES} send `cat .cxsom-hostname-config` `cat .cxsom-port-config` -- frozen ${FROZEN_PREFIX}-`printf %04d ${TIMESTEP}` $(TIMESTEP) $(NMAPS)

.PHONY: send-film-rules
send-film-rules:
	for i in `seq ${BEGIN} ${STEP} ${END} ` ;do \
		${RULES} send `cat .cxsom-hostname-config` `cat .cxsom-port-config` -- frozen ${FROZEN_PREFIX}-`printf %04d $${i}` $${i} $(NMAPS);\
	done

.PHONY: clear-relax-traj
clear-relax-traj:
	rm -r `cat .cxsom-rootdir-config`/${RELAX_PREFIX}-*

.PHONY: plot-weights
plot-weights:
	 @python3 ../plots/plot_weigths.py ${TIMESTEP} 0 image ${TIME_INP} `cat .cxsom-rootdir-config` 'I1' 'I2'

.PHONY: plot-field
plot-field:
	@python3 ./plot_champs.py `cat .cxsom-rootdir-config` zrlx-`printf %04d ${TIMESTEP}`-`printf %04d ${TIME_INP}` ${TIMESTEP} ${TIME_INP} ${NTRAJ}


.PHONY: view-rlx-stat
view-rlx-stat:
	@python3 view-relax-stat.py `cat .cxsom-rootdir-config` ${TIMESTEP}
