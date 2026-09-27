"""Explicit native-key variants; never query an index with another key layout."""
import hashlib,json,os
from pathlib import Path

def supplement_source():
 variant=os.environ.get('AMAZON_SUPPLEMENT_VARIANT','legacy')
 if variant not in ('legacy','phase8'):raise ValueError('AMAZON_SUPPLEMENT_VARIANT must be legacy or phase8')
 return Path(__file__).parent/'native'/('supplement.cpp' if variant=='legacy' else 'supplement_phase8.cpp')

def validate_supplement_schema(directory,index_path,signature):
 paths=[Path(index_path).parent/'supplement_manifest.json'] if index_path else [Path(directory).parent/s/'supplement_manifest.json' for s in ('train','test')]
 for path in paths:
  if path.exists() and json.loads(path.read_text(encoding='utf-8'))['native_sha256']!=signature:
   raise ValueError(f'Supplement key layout differs from {path}. Select the matching AMAZON_SUPPLEMENT_VARIANT (legacy or phase8), or use a fresh working directory; never mix key layouts.')
