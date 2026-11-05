"""시험의 임시 폴더를 레포의 runs/ 아래(진짜 디스크)에 둔다.

/tmp가 메모리 디스크(tmpfs)인 기계에서는 시험이 만드는 체크포인트가 메모리를 먹고, 그 여유가 2GB 아래로
내려가면 실행기의 디스크 안전장치(min_free_gb)가 걸려 시험이 엉뚱하게 실패한다(2026-11-05에 실제로 겪었다).
--basetemp을 직접 주면 그쪽을 쓴다. 여기서 만든 폴더는 시험이 끝나면 지운다.
"""
import os
import shutil
from pathlib import Path

import pytest

_own: Path | None = None


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    global _own
    if config.option.basetemp is None and not hasattr(config, "workerinput"):
        _own = Path(__file__).resolve().parents[1] / "runs" / "_pytest_tmp" / str(os.getpid())
        _own.parent.mkdir(parents=True, exist_ok=True)
        config.option.basetemp = str(_own)


def pytest_unconfigure(config):
    if _own is not None:
        shutil.rmtree(_own, ignore_errors=True)
