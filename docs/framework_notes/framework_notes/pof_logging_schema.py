"""
POF Logging Template (structure-first, no meaning/intent attribution)

이 파일은 실행을 위한 알고리즘이 아니다.
또한 결과를 산출하거나 전이를 검증하지 않는다.

POF에서 정의한 관측 좌표(Sₙ, Xₙ, Aₙ, J, Rₙ, T, d)를
기록 가능한 형식으로 옮긴 "개념적 로깅 템플릿"이다.

- 의미/의도/감정 귀속 금지
- 예측/유도/최적화 규칙 제시 금지
- 사후 기록 원칙 유지
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple


# ============================================================
# 0) 기본 타입 (관측 기록용)
# ============================================================

Payload = Dict[str, Any]
Distribution = Dict[str, float]  # (선택) 출력 분포 스냅샷: 토큰/라벨 -> 확률


# ============================================================
# 1) POF 관측 좌표: S, X, A, J, R, T, d (정의 비고정)
# ============================================================

@dataclass(frozen=True)
class State:
    """
    Sₙ: 인간 입력 상태의 '관측 스냅샷'
    - 내부 상태를 가정하지 않음
    - 해석 변수를 포함하지 않음
    """
    payload: Payload


@dataclass(frozen=True)
class SystemView:
    """
    Aₙ: 시스템 활성 상태의 '관측 가능 대체표현'
    - 개발자 수준 실측이 아니라, 관측자가 남길 수 있는 표지/메타만 허용
    - 예: 모델/세션 정보, 출력 형식 특성, 구조적 요약 형태 등
    """
    payload: Payload


@dataclass(frozen=True)
class Junction:
    """
    J: 정합 판단 지점(판정이 아니라 '관측 표지')
    - 정합/오해 가능성이 갈라지는 지점에 대한 태그/메모
    - 숫자 점수 강제하지 않음
    """
    payload: Payload


@dataclass(frozen=True)
class RelationalOutcome:
    """
    Rₙ: 관계적 결과(사후 관측 지표)
    - 참/거짓 판단이나 원인 귀속 없이,
      '사후적으로 관측된 변화/반응'만 기록
    """
    payload: Payload


@dataclass(frozen=True)
class TransitionRecord:
    """
    단일 전이 기록: Sₙ -> Sₙ₊₁
    - "전이 발생"을 관측 표식(T)으로만 기록
    - d(·,·)는 정의 비고정: '차이가 있었다'는 표기만 허용
    """
    n: int

    # 입력/출력(관측자 관점)
    x_n: str                      # 사용자가 입력한 원문(또는 요약)
    y_n: str                      # 시스템 출력 원문(또는 요약)

    # 관측 좌표
    S_n: State
    X_n: Optional[Payload] = None       # 입력 구성 요소(E, Δ, C 등) '표기' (선택)
    A_n: Optional[SystemView] = None    # 시스템 관측 대체표현 (선택)
    J: Optional[Junction] = None        # 정합 지점 표지 (선택)
    R_n: Optional[RelationalOutcome] = None  # 사후 결과 (선택)

    # (선택) 분포 스냅샷 & 구조 변화량
    P_n: Optional[Distribution] = None
    P_n1: Optional[Distribution] = None
    delta_phi: Optional[float] = None   # Δφ: 분포 구조 변화량(선택)

    # 관측 표식/메타 (태그)
    tags: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# 2) d(·,·) / Δφ: "구조 변화"를 기록하는 최소 도구 (선택)
#    - 측정값이 아니라, 기록 가능한 최소 연산자
# ============================================================

def l1_distance(a: Distribution, b: Distribution) -> float:
    """Dist(P, Q): 분포 간 L1 거리(선택)."""
    keys = set(a) | set(b)
    return sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


def compute_delta_phi(P_n: Distribution, P_n1: Distribution,
                      dist: Callable[[Distribution, Distribution], float] = l1_distance) -> float:
    """
    Δφₙ = Dist(Pₙ₊₁, Pₙ)
    - 의미/감정 변화가 아니라 '출력 분포의 구조 변화량'만 기록
    """
    return dist(P_n1, P_n)


# ============================================================
# 3) 기호 레이어: 문서 기호 ↔ 코드 안전 이름
#    - 문서에서는 기호를 유지
#    - 코드에서는 ASCII 함수로 '표식'만 찍음
# ============================================================

SYMBOL_MAP = {
    "⟡": "mark_observed",    # 관측 발생 표식
    "Δφ": "delta_phi",       # 구조 변화량(선택)
    "❖": "mark_coherent",    # 정합성 표식(판정 아님)
    "❂": "mark_impact",      # 영향 흔적 표식
    "⟁Σ": "aggregate",       # 누적/집계
    "∞⊕": "stream_join",     # 스트림 결합
}


def mark_observed(tr: TransitionRecord, *, key: str = "T", value: bool = True) -> TransitionRecord:
    """
    ⟡ / T : 전이 발생(관측) 표식
    - 발생 여부를 '관측했다'는 기록
    - 원인/의도/정서 귀속 없음
    """
    tags = dict(tr.tags)
    tags[key] = value
    return replace(tr, tags=tags)


def mark_coherent(tr: TransitionRecord, *, key: str = "R_coherence", value: Any = True) -> TransitionRecord:
    """
    ❖ : 정합성 '판정'이 아니라, 정합성으로 해석될 수 있는 관측 표지(태그)
    예) True/False, 또는 "high/medium/low" 같은 비정량 라벨도 허용
    """
    tags = dict(tr.tags)
    tags[key] = value
    return replace(tr, tags=tags)


def mark_impact(tr: TransitionRecord, *, key: str = "R_impact", value: Any = True) -> TransitionRecord:
    """
    ❂ : 전이 이후 파급/여파에 대한 관측 표지(태그)
    """
    tags = dict(tr.tags)
    tags[key] = value
    return replace(tr, tags=tags)


# ============================================================
# 4) POF 로거: "기록 컨테이너"
#    - 분석/검증 도구가 아니라 '로그 보관'이 목적
# ============================================================

@dataclass
class POFLogger:
    """
    POF 관측 기록 컨테이너
    - trace: 전이 기록 목록
    """
    trace: List[TransitionRecord] = field(default_factory=list)

    def record(self, tr: TransitionRecord) -> None:
        self.trace.append(tr)

    def last(self) -> Optional[TransitionRecord]:
        return self.trace[-1] if self.trace else None

    def export_min(self) -> List[Dict[str, Any]]:
        """
        외부 전달용 최소 포맷(접속을 위한 형태)
        - 다른 프레임(심리/UX/기술)에 넘길 때 사용 가능
        """
        out: List[Dict[str, Any]] = []
        for t in self.trace:
            out.append({
                "n": t.n,
                "x_n": t.x_n,
                "y_n": t.y_n,
                "S_n": t.S_n.payload,
                "X_n": t.X_n,
                "A_n": None if t.A_n is None else t.A_n.payload,
                "J": None if t.J is None else t.J.payload,
                "R_n": None if t.R_n is None else t.R_n.payload,
                "T": t.tags.get("T"),
                "d": t.tags.get("d"),
                "delta_phi": t.delta_phi,
                "tags": t.tags,
            })
        return out


# ============================================================
# 5) "전이 기록"을 만드는 보조 함수 (실행 알고리즘 아님)
#    - 입력/출력/스냅샷을 받아 '기록 객체'로 정리만 함
# ============================================================

def make_transition(
    *,
    n: int,
    x_n: str,
    y_n: str,
    S_n_payload: Payload,
    S_n1_payload: Optional[Payload] = None,
    X_n: Optional[Payload] = None,
    A_n: Optional[Payload] = None,
    J: Optional[Payload] = None,
    R_n: Optional[Payload] = None,
    P_n: Optional[Distribution] = None,
    P_n1: Optional[Distribution] = None,
    dist: Callable[[Distribution, Distribution], float] = l1_distance,
) -> TransitionRecord:
    """
    기록 생성 헬퍼
    - Sₙ₊₁이 없으면, 관측자가 나중에 채우도록 비워둘 수 있음
    - Δφ는 P 스냅샷이 둘 다 있을 때만 계산(선택)
    """
    S_n = State(payload=S_n_payload)

    # S_{n+1}은 '관측자 기록'이므로 선택(없어도 됨)
    if S_n1_payload is None:
        S_n1_payload = {}

    tr = TransitionRecord(
        n=n,
        x_n=x_n,
        y_n=y_n,
        S_n=S_n,
        X_n=X_n,
        A_n=None if A_n is None else SystemView(payload=A_n),
        J=None if J is None else Junction(payload=J),
        R_n=None if R_n is None else RelationalOutcome(payload=R_n),
        P_n=P_n,
        P_n1=P_n1,
    )

    # Δφ(선택): 분포 스냅샷이 있을 때만
    if P_n is not None and P_n1 is not None:
        tr = replace(tr, delta_phi=compute_delta_phi(P_n, P_n1, dist))

    # 기본 표식: 전이 발생 관측(T)
    tr = mark_observed(tr, key="T", value=True)

    # d(·,·)는 "차이가 있었다"는 표기만(값은 비고정)
    # - 관측자에게 강제하지 않기 위해 tags로만 제공
    tr = replace(tr, tags={**tr.tags, "d": "observed_difference"})
    return tr


# ============================================================
# 6) (옵션) 간단 데모
#    - 이 파일이 동작함을 보여주기 위한 예시일 뿐
# ============================================================

if __name__ == "__main__":
    logger = POFLogger()

    tr0 = make_transition(
        n=0,
        x_n="오늘 대화가 좀 달라졌어.",
        y_n="어떤 점에서 달라졌는지 관측 항목으로 분리해볼까요?",
        S_n_payload={"self_report": "changed", "context": "after_update"},
        X_n={"E": "low", "Δ": "style_shift", "C": "meta_entry"},  # 예시 라벨(비정량)
        A_n={"output_style": "structured", "verbosity": "medium"},
        J={"misread_risk": "low"},
        R_n={"after_effect": "stabilized"},
    )

    tr0 = mark_coherent(tr0, value="medium")
    tr0 = mark_impact(tr0, value=True)

    logger.record(tr0)

    # 최소 포맷 출력
    for row in logger.export_min():
        print(row)
