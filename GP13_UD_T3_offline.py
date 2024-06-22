#!/pbs/home/x/xtian/.conda/envs/grandlib2304/bin/python3
from grand.grandlib_classes.grandlib_classes import *
import grand.dataio.root_trees as rt
import numpy as np
import os
import sys
# import glob
# from scipy.fft import rfftfreq, rfft, irfft
from grand import ECEF, Geodetic, GRANDCS, LTP

coord_1078 = Geodetic(latitude=40.99368437530295, longitude=93.95411072589444, height=1205.9284000000027)
def get_DU_coord(lat, long, alt, obstime, origin=coord_1078):
  # From GPS to Cartisian coordinates
  geod = Geodetic(latitude=lat, longitude=long, height=alt)
  gcs = GRANDCS(geod, obstime=obstime, location=origin)
  return gcs

timewindow_ns = 7e3 # the coincidence timewindow
nDU = 3

# fname = "data/GP13_UD/GP13_20240613_164741_RUN127_UD_RAW_ChanXYZ_20dB_11DUs_001.root"
fpath = sys.argv[1]
fname = fpath.split('/')[-1]
out_path = "coincidence_table/" + fname + '/'
os.makedirs(out_path, exist_ok=True)
file_GP13_UD = rt.DataFile(fpath)
n = file_GP13_UD.tadc.get_number_of_entries()
list_du_id = np.zeros(n, dtype=int)
list_du_n = np.zeros(n, dtype=int)
list_du_nanoseconds = np.zeros(n, dtype=np.int64)
list_du_seconds = np.zeros(n, dtype=np.int64)
list_traces = np.zeros((n, 4, 1024), dtype=int)
list_lat = np.zeros(n, float)
list_lon = np.zeros(n, float)
list_alt = np.zeros(n, float)
# plt.figure(figsize=(12, 4))
_list_traces = np.zeros((4, 1024), dtype=int)
for k in range(n):
  file_GP13_UD.tadc.get_entry(k)
  file_GP13_UD.trawvoltage.get_entry(k)
  _list_du_n = len(file_GP13_UD.tadc.du_id)
  _list_du_id = file_GP13_UD.tadc.du_id[0]
  _list_lat = file_GP13_UD.trawvoltage.gps_lat[0]
  _list_lon = file_GP13_UD.trawvoltage.gps_long[0]
  _list_alt = file_GP13_UD.trawvoltage.gps_alt[0]
  _list_du_nanoseconds = file_GP13_UD.tadc.du_nanoseconds[0]
  _list_du_seconds = file_GP13_UD.tadc.du_seconds[0]
  _t_len = len(file_GP13_UD.tadc.trace_ch[0][1])
  if _t_len != 1024:
    # File is corrupted, exit
    os.rmdir(out_path)
    print(f"Trace length is not 1024 but {_t_len}.: " + fname)
    exit(1)
  _list_traces[0] = file_GP13_UD.tadc.trace_ch[0][0]
  _list_traces[1] = file_GP13_UD.tadc.trace_ch[0][1]
  _list_traces[2] = file_GP13_UD.tadc.trace_ch[0][2]
  _list_traces[3] = file_GP13_UD.tadc.trace_ch[0][3]
  # plt.clf()
  # plt.plot(high_pass_filter(_list_traces[i][1,:]), marker='.', alpha=.5)
  # plt.plot(high_pass_filter(_list_traces[i][2,:]), marker='.', alpha=.5)
  # plt.tight_layout()
  # plt.savefig(f"imgs/Filtered_{v}_{j}.pdf")
  list_du_id[k] = _list_du_id
  list_du_n[k] = _list_du_n
  list_du_nanoseconds[k] = _list_du_nanoseconds
  list_du_seconds[k] = _list_du_seconds
  list_traces[k] = _list_traces
  list_lat[k] = _list_lat
  list_lon[k] = _list_lon
  list_alt[k] = _list_alt


def safe_substraction(sec1, sec2):
  # Substract seconds in nanosecond
  # to avoid the rounding errors
  nanosec1 = np.round(sec1 * 1e9)
  nanosec2 = np.round(sec2 * 1e9)
  return (nanosec1 - nanosec2) / 1e9


# Offline T3 trigger
def grand_T3_trigger(arr_time_sorted, width, nDU):
   n = len(arr_time_sorted)
   arr_index = np.arange(n)
   triggr_time = []
   i = 0
   t = arr_time_sorted[0] + width
   while t < arr_time_sorted[-1]:
    mask_coin = np.abs(arr_time_sorted - t) <= width
    if np.sum(mask_coin) >= nDU:
      # Possible timing coincidence,
      # search around this time for the most DU triggered
      # Update the central timestamp to the trigger time of first DU + window
      triggr_time.append(t)
      # Jump to the next one, if goes to the end, exit
      # the 'window' here is half of the actual coincidence window (+-) 
      mask_t_next = arr_time_sorted > (t + 1 * width)
      if np.sum(mask_t_next):
        i = arr_index[mask_t_next][0]
        t = arr_time_sorted[i] + width
      else:
        # the end of the file, exit
        break
    else:
      i += 1
      t = arr_time_sorted[i] + width
   return triggr_time

# Sort the time
ref_sec = np.min(list_du_seconds)
ref_nanosec = list_du_nanoseconds[np.argmin(list_du_seconds)]
# Get the time elapsed wrt the first time point
list_sec0 = list_du_seconds - ref_sec
list_nanosec0 = list_du_nanoseconds - ref_nanosec
list_time0 = list_sec0.astype(np.float64) + list_nanosec0.astype(np.float64) / 1e9
list_time0_sorted = np.sort(list_time0)
mask_time0_sort = np.argsort(list_time0)
list_du_id_sorted = list_du_id[mask_time0_sort]
list_trigger_time = grand_T3_trigger(list_time0_sorted, timewindow_ns / 1e9, nDU)
list_trigger_du_ids = [list_du_id_sorted[(np.abs(list_time0_sorted - time) <= (timewindow_ns / 1e9))] for time in list_trigger_time]
list_n_DU = np.array([len(i) for i in list_trigger_du_ids])
list_trigger_du_ids_flatted = np.array([du for dus in list_trigger_du_ids for du in dus])

# Save the generated data in file for later (faster) searches.
# Takes too much space.
# np.savez(f"{out_path}fname.npz",
#          list_du_id=list_du_id,
#          list_du_seconds=list_du_seconds,
#          list_du_nanoseconds=list_du_nanoseconds,
#          list_lat=list_lat,
#          list_lon=list_lon,
#          list_alt=list_alt,
#          list_traces=list_traces,
#          list_trigger_time=list_trigger_time,
#          t3_trigger_pars=[timewindow_ns, nDU])

n = 0
i_event = 0
with open(f"{out_path}/Rec_coinctable.txt", 'w') as f:
  with open(f"{out_path}/coord_antennas.txt", 'w') as f_coord:
    with open(f"{out_path}/DU_id.txt", 'w') as f_duid:
      for t in list_trigger_time:
        # Coincidence timewindow
        mask_time_conincidence = (np.abs(list_time0_sorted - t) <= (timewindow_ns / 1e9))
        for i, du_id in enumerate(list_du_id[mask_time0_sort][mask_time_conincidence]):
          # Use the GPS timestamp as trigger time for reconstruction
          # Use the unfiltered Y as the trigger channel
          index_peak = np.argmax(list_traces[mask_time0_sort][mask_time_conincidence,2,:][i])
          time_peak = index_peak * 2 # ns
          # Use ChY as the peak amplitude
          amp_peak = list_traces[mask_time0_sort][mask_time_conincidence,2,:][i][index_peak]
          # amp_peak = 1
          # f.write(f"{n} {i_event} {second_with_nano[du_mask][mask_time_conincidence][i]:.9f} {amp_peak}\n") # LineNumber, EventID, TriggerTime, PeakAmplitude
          # Use the first triggered DU as the time origin
          f.write(f"{n} {i_event} {list_time0_sorted[mask_time_conincidence][i]:.9f} {amp_peak}\n") # LineNumber, EventID, TriggerTime, PeakAmplitude
          # Coordinates in meter
          date = datetime.datetime.utcfromtimestamp(list_time0_sorted[mask_time_conincidence][i])
          gcs = get_DU_coord(list_lat[mask_time0_sort][mask_time_conincidence][i],
                            list_lon[mask_time0_sort][mask_time_conincidence][i],
                            list_alt[mask_time0_sort][mask_time_conincidence][i],
                            str(date)[:10])
          f_coord.write(f"{n} {gcs.x[0]} {gcs.y[0]} {gcs.z[0] + coord_1078.height[0]}\n")
          f_duid.write(f"{list_du_id[mask_time0_sort][mask_time_conincidence][i]} {ref_sec} {ref_nanosec}\n")
          n += 1
        i_event += 1


def dist_delta_t():
  list_du_seconds_0 = list_du_seconds - list_du_seconds[0]
  seconds_with_nano = list_du_seconds_0 + list_du_nanoseconds / 1e9
  list_du_id_unique = np.unique(list_du_id)
  bin_edges = np.logspace(-6, 3)
  bin_width = (bin_edges[1:] - bin_edges[:-1])
  count = np.zeros((len(list_du_id_unique), len(bin_edges) - 1))
  for i, du in enumerate(list_du_id_unique):
      n = np.sum(list_du_id==du)
      count[i], edges = np.histogram(np.diff(np.sort(seconds_with_nano[list_du_id == du])), bin_edges, weights=np.ones(n-1)/n)
  np.savez(f"data/{fpath.split('/')[-1]}.binned.npz", du_seconds=list_du_seconds, du_nanoseconds=list_du_nanoseconds,
            du_id=list_du_id, du_id_unique=list_du_id_unique,
            count=count, edges=bin_edges)
