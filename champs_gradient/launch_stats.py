import pycxsom as cx
import os
import sys
import subprocess
import time

#100 experiences a lancer !
for i in range(0,5):
    path = f"./stats3D/cercle{i}"
    os.mkdir(path)
    subprocess.run(["python3","fill_inputs_3D.py",f"{path}","10000","1000"])
    #lance cxsom processor
    command = ["cxsom-processor", path, "4", "10000"]
    p = subprocess.Popen(command)
    #lance l'apprentissage
    subprocess.run(["../bin/relax", "send", "localhost", "10000", "--","main", "3"])
    #lance les tests (films ...)
    for j in range(0,2500,50):
        subprocess.run(["../bin/relax", "send", "localhost", "10000", "--", "frozen", "zfrz-%04d"%j, "%d"%j,"3"])
    time.sleep(0.02)
    #On vérifie que le dernier test a fini de tourner:
    #si sa valeur de la timeline d'une variable n'avance pas, c'est que c'est bon. (par ex on prend BMU)
    varpath = os.path.join(path,'zfrz-2450-out','M1','BMU.var')
    with cx.variable.Realize(varpath) as x :
        x.sync_init() # we record the current time, from which we wait for a modification.
        r = x.time_range()
        while x.time_range() is None or  x.time_range()[1]<999:
            x.wait_next(sleep_duration=.01) # We scan every 10ms to check for a value change. # We scan every 10ms to check for a value change.

        time.sleep(0.05)# on attend encore un poil
    #si ok, on peut fermer le processeur et lancer le suivant.
    print(f"folder {i}")
    p.kill()
