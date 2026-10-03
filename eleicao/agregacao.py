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


ROTULOS = ["nome_urna", "partido"]


def _rotulos(df: pd.DataFrame, chaves: list[str], peso: str) -> pd.DataFrame:
    """Um nome de urna e um partido por candidato, para exibição.

    O candidato é identificado pelo nome civil normalizado, que é estável. Nome de
    urna e sigla são só rótulos e podem variar entre municípios num mesmo arquivo do
    TSE. Se entrassem na chave do agrupamento, a mesma pessoa viraria duas linhas,
    cada uma com parte dos votos, e o gráfico mostraria uma barra menor do que a
    real. Então eles são escolhidos depois: fica o rótulo do maior peso agregado.
    """
    cols = [c for c in ROTULOS if c in df.columns]
    if not cols:
        return pd.DataFrame(columns=chaves + ["candidato"])
    return (df.groupby(chaves + ["candidato"] + cols, as_index=False, dropna=False)[peso].sum()
              .sort_values(peso, ascending=False, kind="stable")
              .drop_duplicates(chaves + ["candidato"])
              .drop(columns=peso))


def agregar(df: pd.DataFrame, col: str, por: list[str], peso: str = "peso_agregacao") -> pd.DataFrame:
    chaves = [c for c in ["ano", "turno"] if c in df.columns] + por
    tmp = df.assign(_pxw=df[col] * df[peso])
    # Agrupa SÓ por candidato: ver _rotulos().
    g = tmp.groupby(chaves + ["candidato"], as_index=False)[["_pxw", peso]].sum()
    g[col] = g["_pxw"] / g[peso].where(g[peso] > 0)
    # Renormaliza (só altera algo se houver municípios com peso zero / ausentes).
    tot = g.groupby(chaves)[col].transform("sum")
    g[col] = g[col] / tot
    g = g.drop(columns=["_pxw"])
    if any(c in df.columns for c in ROTULOS):
        g = g.merge(_rotulos(df, chaves, peso), on=chaves + ["candidato"], how="left")
    return g


def oficial(resultados: pd.DataFrame, ano: int, turno: int, por: list[str]) -> pd.DataFrame:
    """Porcentagem oficial de votos válidos (soma de votos / soma de válidos)."""
    r = resultados[(resultados.ano == ano) & (resultados.turno == turno)]
    g = r.groupby(por + ["candidato"], as_index=False)["votos"].sum()
    g["pct_oficial"] = g["votos"] / g.groupby(por)["votos"].transform("sum") if por else g["votos"] / g["votos"].sum()
    return g
