#!/pbs/home/x/xtian/.conda/envs/grandlib2304/bin/python3
from grand.grandlib_classes.grandlib_classes import *
import grand.dataio.root_trees as rt
import numpy as np
import os
import sys
# import glob
from scipy.fft import rfftfreq, rfft, irfft

fpath = sys.argv[1]
fname = fpath.split('/')[-1]
out_path = "/sps/grand/xtian/grand_offline_T3_trigger/coincidence_table/" + fname + '/'

file_size = os.path.getsize(out_path + "DU_id.txt")
if file_size == 0:
  print(f"Empty file: {fname}")
  exit(1)

list_entry_number = np.genfromtxt(out_path + "DU_id.txt").reshape((-1, 4))[:,3]
traces = np.zeros((len(list_entry_number), 4, 1024), dtype=np.int16)

file_root = rt.DataFile(fpath)
for i, event_i in enumerate(list_entry_number):
  file_root.tadc.get_entry(int(event_i))
  traces[i,0] = file_root.tadc.trace_ch[0][0]
  traces[i,1] = file_root.tadc.trace_ch[0][1]
  traces[i,2] = file_root.tadc.trace_ch[0][2]
  traces[i,3] = file_root.tadc.trace_ch[0][3]
np.savez(out_path + "trace.npz", traces)
