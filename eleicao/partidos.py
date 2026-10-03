"""Padronização de partidos e candidatos entre eleições.

* Siglas são normalizadas (maiúsculas, sem acento, sem espaços/pontuação).
* Cada sigla é mapeada para uma "linhagem" — a sigla atual que sucedeu o
  partido por mudança de nome, incorporação ou fusão (config/linhagem_partidos.csv).
  Isso permite que o desempenho anterior de um partido seja encontrado
  mesmo que ele tenha mudado de nome. Siglas sem entrada na tabela são sua
  própria linhagem. A tabela é editável e deve ser revisada quando houver
  novas fusões.
* Candidatos são identificados entre eleições pelo nome civil completo
  normalizado (NM_CANDIDATO), que é estável; o número e o sequencial mudam.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

import pandas as pd

from . import config


def sem_acento(txt: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", txt) if not unicodedata.combining(c))


def normalizar_sigla(sigla: str) -> str:
    if sigla is None or (isinstance(sigla, float) and pd.isna(sigla)):
        return ""
    return re.sub(r"[^A-Z0-9]", "", sem_acento(str(sigla)).upper())


def normalizar_nome(nome: str) -> str:
    if nome is None or (isinstance(nome, float) and pd.isna(nome)):
        return ""
    return re.sub(r"\s+", " ", re.sub(r"[^A-Z ]", " ", sem_acento(str(nome)).upper())).strip()


@lru_cache(maxsize=1)
def _tabela() -> dict[str, str]:
    t = pd.read_csv(config.ARQ_LINHAGEM, dtype=str).fillna("")
    return {normalizar_sigla(a): normalizar_sigla(b) for a, b in zip(t.sigla_historica, t.linhagem)}


def linhagem(sigla: str) -> str:
    s = normalizar_sigla(sigla)
    return _tabela().get(s, s)
