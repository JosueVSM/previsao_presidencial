"""Agregação das previsões municipais para UF e Brasil.

Critério: média das porcentagens municipais PONDERADA pelos votos válidos
esperados em cada município,

    p_BR(c) = Σ_m W_m · p_m(c) / Σ_m W_m,
    W_m = eleitorado_apto_m(ano-alvo) × (votos válidos / aptos)_m na eleição anterior.

Justificativa: a porcentagem nacional oficial é Σ votos(c) / Σ votos válidos,
que é exatamente a média das porcentagens municipais ponderada pelos votos
válidos de cada município. Como os votos válidos do ano-alvo não são
conhecidos antes da eleição, usamos a estimativa acima, que só depende do
cadastro eleitoral do ano-alvo e da taxa de votos válidos por eleitor apto
observada na eleição anterior. A média simples entre municípios NÃO é usada,
pois daria a um município de 1 mil eleitores o mesmo peso de São Paulo.
"""
from __future__ import annotations

import pandas as pd


def agregar(df: pd.DataFrame, col: str, por: list[str], peso: str = "peso_agregacao") -> pd.DataFrame:
    chaves = [c for c in ["ano", "turno"] if c in df.columns] + por
    tmp = df.assign(_pxw=df[col] * df[peso])
    g = tmp.groupby(chaves + ["candidato", "nome_urna", "partido"], as_index=False)[["_pxw", peso]].sum()
    g[col] = g["_pxw"] / g[peso].where(g[peso] > 0)
    # Renormaliza (só altera algo se houver municípios com peso zero / ausentes).
    tot = g.groupby(chaves)[col].transform("sum")
    g[col] = g[col] / tot
    return g.drop(columns=["_pxw"])


def oficial(resultados: pd.DataFrame, ano: int, turno: int, por: list[str]) -> pd.DataFrame:
    """Porcentagem oficial de votos válidos (soma de votos / soma de válidos)."""
    r = resultados[(resultados.ano == ano) & (resultados.turno == turno)]
    g = r.groupby(por + ["candidato"], as_index=False)["votos"].sum()
    g["pct_oficial"] = g["votos"] / g.groupby(por)["votos"].transform("sum") if por else g["votos"] / g["votos"].sum()
    return g
