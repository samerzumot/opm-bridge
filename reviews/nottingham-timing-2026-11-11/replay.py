"""Replay the corrected timing from the previously checksum-verified raw archive."""
from scripts import stream_nottingham_zenodo as stream
archive = 'data/validation/nottingham_download/sub-101.zip'
stream.HttpRangeFile = lambda url: open(archive, 'rb', buffering=0)
stream.stream_nottingham_subjects(['sub-101'], out_dir='data/cache/validation/nottingham_timing_corrected')
