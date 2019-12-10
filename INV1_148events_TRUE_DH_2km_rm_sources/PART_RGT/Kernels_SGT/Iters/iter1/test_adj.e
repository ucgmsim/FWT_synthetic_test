Traceback (most recent call last):
  File "../../SGT_read_all.py", line 172, in <module>
    nRec, R, statnames = read_stat_name(station_file)
  File "../../SGT_read_all.py", line 64, in read_stat_name
    with open(station_file, 'r') as f:
FileNotFoundError: [Errno 2] No such file or directory: '../../StatInfo/STATION.txt'
srun: error: nid00020: task 0: Exited with exit code 1
srun: Terminating job step 743290.0
