"""Fetch only sub-101 using bounded parallel ranges; verify ZIP CRC and size."""
import concurrent.futures
import io
import struct
import urllib.request
import zipfile
import zlib
from pathlib import Path
from scripts.stream_nottingham_zenodo import HttpRangeFile, ZENODO_ADULTS_ZIP_URL

root=Path('data/validation/nottingham_download'); root.mkdir(parents=True,exist_ok=True)
name='Adults/sub-101/ses-001/meg/sub-101_ses-001_task-braille_run-001_meg.dat'
url=ZENODO_ADULTS_ZIP_URL

def fetch(start,end):
    request=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}'})
    with urllib.request.urlopen(request,timeout=120) as response:
        if response.status != 206 or not response.headers['Content-Range'].startswith(f'bytes {start}-{end}/'):
            raise ValueError('Server did not honor requested range')
        data=response.read()
    if len(data)!=end-start+1: raise ValueError('Truncated range')
    return data

with zipfile.ZipFile(io.BufferedReader(HttpRangeFile(url),buffer_size=4194304)) as z:
    info=z.getinfo(name)
header=fetch(info.header_offset,info.header_offset+29)
assert header[:4]==b'PK\x03\x04'
name_len,extra_len=struct.unpack('<HH',header[26:30])
start=info.header_offset+30+name_len+extra_len
block=8*1024*1024
ranges=[(i,start+i,min(start+i+block-1,start+info.compress_size-1)) for i in range(0,info.compress_size,block)]
def download(item):
    index,first,last=item
    path=root/f'{index:010d}.part'
    if not path.exists() or path.stat().st_size!=last-first+1:
        for attempt in range(3):
            try:
                path.write_bytes(fetch(first,last)); break
            except Exception:
                if attempt==2: raise
    return path
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    for i,path in enumerate(pool.map(download,ranges),1):
        print(f'Compressed blocks {i}/{len(ranges)}',flush=True)
output=root/'sub-101.zip'
crc=0; total=0; decompressor=zlib.decompressobj(-15)
with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as archive:
    for path in Path('data/validation/source_metadata').iterdir():
        archive.write(path,str(Path(name).parent/path.name))
    with archive.open(name,'w',force_zip64=True) as target:
        for index,_,_ in ranges:
            data=decompressor.decompress((root/f'{index:010d}.part').read_bytes())
            target.write(data); total+=len(data); crc=zlib.crc32(data,crc)
        data=decompressor.flush();target.write(data);total+=len(data);crc=zlib.crc32(data,crc)
assert decompressor.eof and total==info.file_size and crc==info.CRC, (total,crc,info.file_size,info.CRC)
print(f'Verified original ZIP CRC {crc:08x}, {total} uncompressed bytes; {output}',flush=True)
