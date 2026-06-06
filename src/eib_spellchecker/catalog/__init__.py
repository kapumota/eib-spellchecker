# Codigo base de EIB Spellchecker.
# Implementa componentes principales del paquete, la API, la CLI y los backends.

from .legacy import inventory_from_zips
from .excels import inventory_from_zip as inventory_excels_zip, parse_score_file, parse_score_text, summarize_scores

__all__ = [
    "inventory_from_zips",
    "inventory_excels_zip",
    "parse_score_file",
    "parse_score_text",
    "summarize_scores",
    "inventory_research_runs",
    "benchmark_research_runs",
    "summarize_research_benchmarks",
    "benchmark_markdown",
]


def inventory_research_runs(*args, **kwargs):
    from .research import inventory_research_runs as _fn
    return _fn(*args, **kwargs)


def benchmark_research_runs(*args, **kwargs):
    from .research import benchmark_research_runs as _fn
    return _fn(*args, **kwargs)


def summarize_research_benchmarks(*args, **kwargs):
    from .research import summarize_research_benchmarks as _fn
    return _fn(*args, **kwargs)


def benchmark_markdown(*args, **kwargs):
    from .research import benchmark_markdown as _fn
    return _fn(*args, **kwargs)
