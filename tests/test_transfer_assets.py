import os,hashlib
from pathlib import Path
import pytest
from src.transfer_assets import pack,restore,safe
from src.rescoring import sourcehash

@pytest.mark.parametrize('pointer',[False,True])
def test_transfer_roundtrip_preserves_hardlinks(tmp_path,monkeypatch,pointer):
 src=tmp_path/'source';src.mkdir();monkeypatch.chdir(src)
 (src/'reports/improvements').mkdir(parents=True);(src/'dataset').mkdir();(src/'work').mkdir()
 a=src/'dataset/record.tsv';a.write_bytes(b'name\tcountry\ncafe\tFrance\n');os.link(a,src/'work/shared.tsv')
 (src/'work/interrupted.partial').write_bytes(b'incomplete')
 bundle=src/'transfer/assets.zip';pack(bundle)
 dest=tmp_path/'destination';dest.mkdir();monkeypatch.chdir(dest)
 if pointer:
  (dest/'dataset').mkdir();(dest/'dataset/record.tsv').write_text('version https://git-lfs.github.com/spec/v1\noid sha256:'+hashlib.sha256(a.read_bytes()).hexdigest()+'\nsize '+str(a.stat().st_size)+'\n')
 restore(bundle)
 assert (dest/'dataset/record.tsv').read_bytes()==a.read_bytes()
 assert os.path.samefile(dest/'dataset/record.tsv',dest/'work/shared.tsv')
 assert not (dest/'work/interrupted.partial').exists()
 assert (dest/'dataset/record.tsv').stat().st_mtime_ns==a.stat().st_mtime_ns
 (dest/'dataset/record.tsv').write_bytes(b'changed')
 with pytest.raises(ValueError,match='Existing file differs'):restore(bundle)

def test_transfer_rejects_path_escape(tmp_path):
 with pytest.raises(ValueError,match='Unsafe archive path'):safe(tmp_path,'../outside')

def test_portable_source_signature(tmp_path):
 a=tmp_path/'a.py';b=tmp_path/'b.py';a.write_bytes(b'x = 1\ny = 2\n');b.write_bytes(b'\xef\xbb\xbfx = 1\r\ny = 2\r\n')
 assert sourcehash(a)==sourcehash(b)
