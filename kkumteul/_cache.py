"""numba 캐시 지키기.

numba는 함수가 정의된 파일의 수정 시각만 보고 캐시를 다시 쓴다. 그래서 `rng.py`나 `config.py`의
상수를 고쳐도 `kernels.py`의 캐시는 옛 코드를 계속 쓴다(독립 검토 2026-11-04에서 확인).
계산에 쓰이는 소스 다섯 파일의 해시가 바뀌면 이 패키지의 numba 캐시를 지운다.
"""
import hashlib
from pathlib import Path

_SOURCES = ("rng.py", "config.py", "kernels.py", "stages.py", "world.py")


def source_hash() -> str:
    here = Path(__file__).parent
    m = hashlib.sha256()
    for name in _SOURCES:
        m.update((here / name).read_bytes())
    return m.hexdigest()


def refresh() -> str:
    h = source_hash()
    cache = Path(__file__).parent / "__pycache__"
    try:
        cache.mkdir(exist_ok=True)
        mark = cache / "kkumteul-src.hash"
        if not mark.exists() or mark.read_text().strip() != h:
            for p in list(cache.glob("*.nbi")) + list(cache.glob("*.nbc")):
                p.unlink(missing_ok=True)
            mark.write_text(h)
    except OSError:
        pass   # 읽기 전용 설치 등: 캐시를 못 지우면 그대로 둔다
    return h
