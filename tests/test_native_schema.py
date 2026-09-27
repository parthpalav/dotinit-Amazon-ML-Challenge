import json,os
from pathlib import Path
from types import SimpleNamespace
import pytest
from src.native_schema import supplement_source,validate_supplement_schema
from src.supplement import build_supplement

def test_native_variants_are_explicit_and_mismatches_fail(tmp_path,monkeypatch):
 monkeypatch.delenv('AMAZON_SUPPLEMENT_VARIANT',raising=False);assert supplement_source().name=='supplement.cpp'
 monkeypatch.setenv('AMAZON_SUPPLEMENT_VARIANT','phase8');assert supplement_source().name=='supplement_phase8.cpp'
 store=tmp_path/'train';store.mkdir();(store/'supplement_manifest.json').write_text(json.dumps({'native_sha256':'legacy-hash'}))
 with pytest.raises(ValueError,match='key layout'):validate_supplement_schema(tmp_path/'native',store/'supplement_index.bin','phase8-hash')
 validate_supplement_schema(tmp_path/'native',store/'supplement_index.bin','legacy-hash')
 monkeypatch.setenv('AMAZON_SUPPLEMENT_VARIANT','typo')
 with pytest.raises(ValueError):supplement_source()

def test_rebuild_cannot_truncate_shared_packed_data(tmp_path,monkeypatch):
 monkeypatch.delenv('AMAZON_SUPPLEMENT_VARIANT',raising=False);store=tmp_path/'train';store.mkdir();source=tmp_path/'preserved.bin';source.write_bytes(b'preserved original data');os.link(source,store/'entity_id.bin')
 with pytest.raises(ValueError,match='shared packed asset'):build_supplement(SimpleNamespace(working_dir=str(tmp_path)),'train')
 assert source.read_bytes()==b'preserved original data'
