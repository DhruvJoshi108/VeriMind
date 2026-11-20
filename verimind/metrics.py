"""Thread-safe in-process operational metrics with Prometheus exposition."""

from __future__ import annotations

import math
from collections import Counter
from threading import Lock


class Metrics:
    """Small dependency-free registry; use a shared backend for multi-process production."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._requests: Counter[tuple[str, str, int]] = Counter()
        self._verdicts: Counter[str] = Counter()
        self._latencies: list[float] = []
        self._retrieval_latencies: list[float] = []
        self._judge_latencies: list[float] = []
        self._judge_errors = 0
        self._flagged_claims = 0
        self._claims = 0

    def observe_request(self, method: str, path: str, status: int, latency_ms: float) -> None:
        with self._lock:
            self._requests[(method, path, status)] += 1
            self._latencies.append(latency_ms)

    def observe_report(self, report: dict[str, object]) -> None:
        claims = report.get("claims", [])
        if not isinstance(claims, list):
            return
        with self._lock:
            self._claims += len(claims)
            for claim in claims:
                if not isinstance(claim, dict):
                    continue
                self._verdicts[str(claim.get("verdict", "unknown"))] += 1
                if float(claim.get("risk_score", 0)) >= 60:
                    self._flagged_claims += 1
                retrieval_latency = claim.get("retrieval_latency_ms")
                if isinstance(retrieval_latency, (int, float)) and math.isfinite(retrieval_latency):
                    self._retrieval_latencies.append(float(retrieval_latency))
                judge_latency = claim.get("judge_latency_ms")
                if isinstance(judge_latency, (int, float)) and math.isfinite(judge_latency):
                    self._judge_latencies.append(float(judge_latency))
                self._judge_errors += int(claim.get("judge_error") is True)

    @staticmethod
    def _percentile(values: list[float], quantile: float) -> float:
        ordered = sorted(values)
        position = (len(ordered) - 1) * quantile
        lower = math.floor(position)
        upper = math.ceil(position)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    def render_prometheus(self) -> str:
        with self._lock:
            lines = ["# TYPE verimind_http_requests_total counter"]
            for (method, path, status), count in sorted(self._requests.items()):
                lines.append(
                    f'verimind_http_requests_total{{method="{method}",path="{path}",status="{status}"}} {count}'
                )
            lines.append("# TYPE verimind_verdicts_total counter")
            for verdict, count in sorted(self._verdicts.items()):
                lines.append(f'verimind_verdicts_total{{verdict="{verdict}"}} {count}')
            total = len(self._latencies)
            average = sum(self._latencies) / total if total else 0.0
            lines.extend(("# TYPE verimind_request_latency_ms gauge", f"verimind_request_latency_ms {average:.3f}"))
            if self._latencies:
                lines.append("# TYPE verimind_request_latency_percentile_ms gauge")
                for quantile in (0.50, 0.95, 0.99):
                    value = self._percentile(self._latencies, quantile)
                    lines.append(
                        f'verimind_request_latency_percentile_ms{{quantile="{quantile:.2f}"}} {value:.3f}'
                    )
            for name, latencies in (
                ("verimind_retrieval_latency_ms", self._retrieval_latencies),
                ("verimind_judge_latency_ms", self._judge_latencies),
            ):
                if latencies:
                    lines.extend(
                        (
                            f"# TYPE {name} gauge",
                            f"{name} {sum(latencies) / len(latencies):.3f}",
                        )
                    )
            lines.extend(
                (
                    "# TYPE verimind_judge_errors_total counter",
                    f"verimind_judge_errors_total {self._judge_errors}",
                )
            )
            ratio = self._flagged_claims / self._claims if self._claims else 0.0
            lines.extend(("# TYPE verimind_flagged_claim_ratio gauge", f"verimind_flagged_claim_ratio {ratio:.6f}"))
            return "\n".join(lines) + "\n"


metrics = Metrics()
