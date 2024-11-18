import time
import torch
import sys
import subprocess

argslist = list(sys.argv)[1:]
num_gpus = torch.cuda.device_count()
argslist.append('--n_gpus={}'.format(num_gpus))
workers = []
job_id = time.strftime("%Y_%m_%d-%H%M%S")
argslist.append("--group_name=group_{}".format(job_id))

for i in range(num_gpus):
    argslist.append('--rank={}'.format(i))
    logdir = argslist[2].split('=')[1]
    stdout = None if i == 0 else open("{}/{}_GPU_{}.log".format(logdir, job_id, i),
                                      "w")
    print(argslist)
    p = subprocess.Popen([str(sys.executable)]+argslist, stdout=stdout)
    workers.append(p)
    argslist = argslist[:-1]

returncode = 0

for i, p in enumerate(workers):
    p.wait()
    if p.returncode != 0:
        returncode = 1

exit(returncode)