"""Replay sub-101 with bounded range reads and progress logging."""
from scripts import stream_nottingham_zenodo as stream

original = stream.HttpRangeFile.readinto
read_count = 0


def bounded_read(self, buffer):
    global read_count
    read_count += 1
    if read_count % 10 == 0:
        print(f'HTTP read {read_count}; archive offset {self.pos}', flush=True)
    # RawIOBase permits short reads. Limit the request rather than asking the
    # server for the full 480 MB compressed member in one read.
    return original(self, memoryview(buffer)[:4 * 1024 * 1024])


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--archive', help='Local sub-101 ZIP verified against source CRC')
args = parser.parse_args()
if args.archive:
    stream.HttpRangeFile = lambda url: open(args.archive, 'rb', buffering=0)
else:
    stream.HttpRangeFile.readinto = bounded_read
stream.stream_nottingham_subjects(['sub-101'], out_dir='data/cache/validation/nottingham')
