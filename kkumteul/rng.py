"""상태 없는 난수.

(seed, stream, a, b) 네 정수를 섞어 값을 만든다. 내부 상태가 없으므로
스레드 수·호출 순서와 무관하게 같은 값이 나오고, 체크포인트에는 seed와 step만 있으면 된다.
"""
import math

import numpy as np
from numba import njit

_U = np.uint64
_C1 = _U(0x9E3779B97F4A7C15)
_C2 = _U(0xC2B2AE3D27D4EB4F)
_C3 = _U(0x165667B19E3779F9)
_M1 = _U(0xBF58476D1CE4E5B9)
_M2 = _U(0x94D049BB133111EB)
_S30 = _U(30)
_S27 = _U(27)
_S31 = _U(31)
_S11 = _U(11)
_INV53 = 1.0 / 9007199254740992.0

# 용도별 stream 번호 (값이 바뀌면 재현성이 깨지므로 고치지 말고 뒤에 덧붙인다)
S_INIT_POS = 1
S_INIT_GENOME = 2
S_FOOD = 3
S_PATCH = 4
S_NOISE = 5
S_MUT_W = 6
S_MUT_GENE = 7
S_BIRTH_POS = 8
S_RESEED = 9
S_PASSIVE = 10
S_ASSAY = 11
S_INIT_AGE = 12
S_BIRTH_ORDER = 13
S_SCENARIO = 14     # 각본의 재난(운석): 떨어지는 자리와 죽는 개체
S_FERTILE = 15      # 거름 땅에 나는 먹이의 자리
S_KIND = 16         # 먹이 종류와, 어느 종류가 독인지
S_ATTACK_ORDER = 17 # 공격을 처리하는 순서의 시작 칸(스텝마다 돌린다)


@njit(cache=True, inline="always")
def _mix(z):
    z = (z ^ (z >> _S30)) * _M1
    z = (z ^ (z >> _S27)) * _M2
    return z ^ (z >> _S31)


@njit(cache=True, inline="always")
def hash_u64(seed, stream, a, b):
    z = _mix(_U(seed) + _C1)
    z = _mix(z ^ (_U(stream) * _C2))
    z = _mix(z + _U(a) * _C3)
    z = _mix(z ^ (_U(b) * _C1 + _C2))
    return z


@njit(cache=True, inline="always")
def u01(seed, stream, a, b):
    """[0, 1) 균등."""
    return (hash_u64(seed, stream, a, b) >> _S11) * _INV53


@njit(cache=True, inline="always")
def normal(seed, stream, a, b):
    """표준 정규 (Box-Muller)."""
    z = hash_u64(seed, stream, a, b)
    u1 = ((z >> _S11) + 0.5) * _INV53
    z2 = _mix(z + _C1)
    u2 = (z2 >> _S11) * _INV53
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(6.283185307179586 * u2)


@njit(cache=True)
def randint(seed, stream, a, b, n):
    """[0, n) 정수."""
    return int(u01(seed, stream, a, b) * n)
