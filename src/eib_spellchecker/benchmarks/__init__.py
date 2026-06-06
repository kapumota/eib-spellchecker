# Codigo base de EIB Spellchecker.
# Implementa componentes principales del paquete, la API, la CLI y los backends.

from .reporting import write_report
from .suite import benchmark_suite

__all__ = ["write_report", "benchmark_suite"]
