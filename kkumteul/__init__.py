"""세계 시뮬레이션 1단계 시제품 (2D 세계 + 몸 + 죽음/번식 + 뇌 비용 + 감각 한계)."""
from ._cache import refresh as _refresh

__version__ = "0.2.0"
STATE_VERSION = 1
SOURCE_HASH = _refresh()   # kernels를 불러오기 전에 낡은 numba 캐시를 지운다
