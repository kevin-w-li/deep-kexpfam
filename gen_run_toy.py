import subprocess
import time

env   = "local"
device = 0

dnames = ["ring","funnel" ,"spiral","uniform", "cosine","multiring", "grid", "banana"]

ps = []
njob =1

for dset in dnames:

    base_command = "python run_toy.py %s" % dset

    with open("run_toy.sh", "w") as f:
        
        f.write("#!/bin/bash\n")
        if env=="cpu":

            f.write('avx2="$(grep avx2 /proc/cpuinfo)"\n')
            f.write("if [[ -z $avx3 ]] ; then source activate tensorflow_cpu; else source activate tensorflow_avx; fi\n" )
            f.write(base_command)

        elif env=="gpu":

            f.write("source activate tensorflow_gpu\n" )
            f.write("CUDA_VISIBLE_DEVICES=%d %s" % (device,base_command))

        elif env=="local":
            f.write("CUDA_VISIBLE_DEVICES=%d %s" % (device,base_command))


    time.sleep(1)
    p = subprocess.Popen(["./run_toy.sh"])
    time.sleep(1)
    ps.append(p)

    if len(ps) == njob:
        for p in ps:
            p.wait()
        ps = []

