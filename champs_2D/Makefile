help:
	@echo
	@echo
	@echo "make cxsom-help                                    <-- help for common cxsom manipulations"
	@echo
	@echo "make main-figs                                     <-- generate figures for main rules"
	@echo "make relax-figs                                    <-- generate figures for expanded relaxation rules"
	@echo "make frozen-figs                                   <-- generate figures for frozen rules"
	@echo
	@echo "make send-main-rules                               <-- sends the main rules"
	@echo "make send-relax-rules TIMESTEP=<time>              <-- sends the relax rules"
	@echo "make send-frozen-rules TIMESTEP=<time>             <-- sends the frozen rules"
	@echo
	@echo "make feed-main-inputs                              <-- feeds X and Y"
	@echo "make feed-relax-inputs TIMESTEP=<time>...          <-- feeds the X,Y pair for relaxation at <time>"
	@echo "     ... THETA=<deg> [XBMU=<xbmu> YBMU=<ybmu>]"
	@echo "make feed-frozen-inputs TIMESTEP=<time>            <-- feeds the X,Y pair for relaxation at <time>"
	@echo
	@echo "make view-weights                                  <-- shows the weight evolution."
	@echo "make view-relaxation TIMESTEP=<time>               <-- shows the relaxation expansion."
	@echo "make view-frozen TIMESTEP=<time>                   <-- shows the frozen analysis."
	@echo

# Adapt this path if needed.
include /usr/share/cxsom/cxsom-makefile

RULES         = ./relax2D
RELAX_PREFIX  = zrlx
FROZEN_PREFIX = zfrz
NTRAJ = 10
BEGIN = 0
END= 29500
STEP=500
NMAPS = 2


.PHONY: compile
compile:
	g++ -std=c++17 -o $(RULES) relaxation2D.cpp -lboost_system -lstdc++fs $$(pkg-config --cflags --libs cxsom-rules cxsom-builder)

.PHONY: main-figs
main-figs:
	@${RULES} graph main -- main ${NMAPS}
	@dot -Tpdf main-inits.dot -o main-inits.pdf
	@dot -Tpdf main-updates.dot -o main-updates.pdf
	@dot -Tpdf archi.dot -o archi.pdf
	@rm main-inits.dot main-updates.dot archi.dot
	@echo
	@echo
	@echo "main-inits.pdf, main-updates.pdf and archi.pdf have been generated."
	@echo

.PHONY: relax-figs
relax-figs:
	@${RULES} graph relax -- relax ${NMAPS} $(RELAX_PREFIX) 1234
	@dot -Tpdf relax-inits.dot -o relax-inits.pdf
	@dot -Tpdf relax-updates.dot -o relax-updates.pdf
	@rm relax-inits.dot relax-updates.dot
	@echo
	@echo
	@echo "relax-inits.pdf and relax-updates.pdf have been generated."
	@echo

.PHONY: frozen-figs
frozen-figs:
	@${RULES} graph frozen -- frozen ${NMAPS} $(FROZEN_PREFIX) 1234
	@dot -Tpdf frozen-inits.dot -o frozen-inits.pdf
	@dot -Tpdf frozen-updates.dot -o frozen-updates.pdf
	@rm frozen-inits.dot frozen-updates.dot
	@echo
	@echo
	@echo "frozen-inits.pdf and frozen-updates.pdf have been generated."
	@echo

.PHONY: send-main-rules
send-main-rules:
	@${RULES} send `cat .cxsom-hostname-config` `cat .cxsom-port-config` -- main ${NMAPS}

.PHONY: feed-main-inputs
feed-main-inputs:
	@python3 feed_main.py `cat .cxsom-rootdir-config` ${N_INPUTS} ${N_TEST}

.PHONY: feed-main-inputs-3D
feed-main-inputs-3D:
	@python3 feed_main_3D.py `cat .cxsom-rootdir-config` ${N_INPUTS} ${N_TEST}

.PHONY: view-weights
view-weights:
	@python3 view-weights.py `cat .cxsom-rootdir-config` ${TIMESTEP} $(FROZEN_PREFIX)-`printf %08d ${TIMESTEP}` I1 I2

.PHONY: send-relax-rules
send-relax-rules:
	for i in `seq ${NTRAJ}`;do \
		${RULES} send `cat .cxsom-hostname-config` `cat .cxsom-port-config` -- relax ${NMAPS} $(RELAX_PREFIX)-`printf %08d ${TIMESTEP}`-`printf %04d ${TIMEINP}`-`printf %03d $${i}` ${TIMESTEP} ; \
	done

.PHONY: feed-relax-inputs
feed-relax-inputs:
	for i in `seq ${NTRAJ}`;do \
		python3 feed-relax.py `cat .cxsom-rootdir-config` $(RELAX_PREFIX)-`printf %08d ${TIMESTEP}`-`printf %04d ${TIMEINP}`-`printf %03d $${i}` ${TIMESTEP} ${TIMEINP} ${NMAPS}; \
	done

.PHONY: view-relaxation
view-relaxation:
	@python3 view-relaxation.py `cat .cxsom-rootdir-config` ${TIMESTEP}

.PHONY: view-relaxation-weight
view-relaxation-weight:
	@python3 view-relaxation-weights.py `cat .cxsom-rootdir-config` ${TIMESTEP} $(RELAX_PREFIX)-`printf %08d ${TIMESTEP}`-`printf %04d ${TIMEINP}` ${NTRAJ} ${TIMEINP}


.PHONY: send-frozen-rules
send-frozen-rules:
	@${RULES} send `cat .cxsom-hostname-config` `cat .cxsom-port-config` -- frozen ${NMAPS} $(FROZEN_PREFIX)-`printf %08d ${TIMESTEP}` ${TIMESTEP}

.PHONY: feed-frozen-inputs
feed-frozen-inputs:
	@python3 feed-frozen.py `cat .cxsom-rootdir-config` `cat .cxsom-hostname-config` `cat .cxsom-port-config` $(FROZEN_PREFIX)-`printf %08d ${TIMESTEP}`

.PHONY: view-frozen
view-frozen:
	@python3 view-frozen.py `cat .cxsom-rootdir-config` ${TIMESTEP} $(FROZEN_PREFIX)-`printf %08d ${TIMESTEP}` I1 I2

.PHONY: view-rlx-stat
view-rlx-stat:
	@python3 view-relax-stats.py `cat .cxsom-rootdir-config` ${TIMESTEP}


.PHONY: send-film-rules
send-film-rules:
	for i in `seq ${BEGIN} ${STEP} ${END} ` ;do \
		${RULES} send `cat .cxsom-hostname-config` `cat .cxsom-port-config` -- frozen ${NMAPS} ${FROZEN_PREFIX}-`printf %08d $${i}` $${i} ;\
	done
