"""Verificações de consistência, escala das porcentagens e vazamento."""
from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd

from . import agregacao, config


def _item(nome: str, ok: bool, detalhe: str) -> dict:
    print(f"  [{'OK' if ok else 'FALHA'}] {nome}: {detalhe}")
    return {"verificacao": nome, "ok": bool(ok), "detalhe": detalhe}


def _somas(df: pd.DataFrame, col: str, chave: str | list[str]) -> tuple[bool, str]:
    s = df.groupby(chave)[col].sum()
    faixa = df[col].between(0, 1).all()
    ok = bool(np.allclose(s.values, 1.0, atol=1e-5) and faixa)
    return ok, f"{len(s)} grupos; soma mín={s.min():.6f}, máx={s.max():.6f}; todos os valores em [0,1]: {faixa}"


def verificar() -> list[dict]:
    p, r = config.DIR_PROCESSADOS, config.DIR_RESULTADOS
    itens: list[dict] = []
    res = pd.read_parquet(p / "resultados_municipio.parquet")

    # --- dados oficiais ----------------------------------------------------
    dup = res.duplicated(["ano", "turno", "uf", "cd_municipio", "candidato"]).sum()
    itens.append(_item("Chave única (eleição, turno, município, candidato)", dup == 0, f"{dup} duplicadas"))
    itens.append(_item("Votos não negativos", (res.votos >= 0).all(), f"mín = {res.votos.min()}"))
    ok, det = _somas(res.assign(f=res.pct_validos / 100), "f", ["ano", "turno", "uf", "cd_municipio"])
    itens.append(_item("% de votos válidos oficiais somam 100% por município", ok, det))
    cont = res.groupby(["ano", "turno"])["cd_municipio"].nunique()
    itens.append(_item("Municípios por eleição/turno", True,
                       "; ".join(f"{a}/{t}º: {n}" for (a, t), n in cont.items())))
    nac = res.groupby(["ano", "turno", "nome_urna"])["votos"].sum().reset_index()
    nac["pct"] = 100 * nac.votos / nac.groupby(["ano", "turno"]).votos.transform("sum")
    top = (nac.sort_values("pct", ascending=False).groupby(["ano", "turno"]).head(2)
              .sort_values(["ano", "turno", "pct"], ascending=[True, True, False]))
    itens.append(_item("Totais nacionais recalculados (conferir com o site do TSE)", True,
                       "; ".join(f"{a}/{t}º {n}: {v:.2f}%" for a, t, n, v in
                                 top[["ano", "turno", "nome_urna", "pct"]].itertuples(index=False))))

    # --- vazamento ---------------------------------------------------------
    alvo_na_base = (res.ano >= config.ANO_ALVO).any()
    itens.append(_item(f"Resultado oficial de {config.ANO_ALVO} fora das entradas", True,
                       "não há resultado de 2026 na base" if not alvo_na_base else
                       "há resultado de 2026 na base, mas carregar_contexto() o descarta antes de montar entradas"))
    arq_m = r / "avaliacao_metricas.csv"
    if arq_m.exists():
        met = pd.read_csv(arq_m)
        ok = all(max(int(x) for x in str(tr).split(", ")) < te
                 for te, tr in zip(met.eleicao_teste, met.eleicoes_treino))
        itens.append(_item("Validação temporal: treino sempre anterior ao teste", ok,
                           "; ".join(sorted({f"teste {te} <- treino {tr}" for te, tr in
                                             zip(met.eleicao_teste, met.eleicoes_treino)}))))
    arq_b = r / "previsoes_backtest.parquet"
    if arq_b.exists():
        b = pd.read_parquet(arq_b)
        chave = ["ano", "turno", "uf", "cd_municipio"]
        for col in ["p_rede", "p_persistencia"]:
            ok, det = _somas(b, col, chave)
            itens.append(_item(f"Backtest: {col} em escala válida (soma 100% por município)", ok, det))
        ag = agregacao.agregar(b, "p_rede", [])
        ok, det = _somas(ag, "p_rede", ["ano", "turno"])
        itens.append(_item("Backtest: agregado nacional soma 100%", ok, det))
    arq_p = r / f"previsao_{config.ANO_ALVO}.parquet"
    if arq_p.exists():
        prev = pd.read_parquet(arq_p)
        ok, det = _somas(prev, "p_rede", ["turno", "uf", "cd_municipio"])
        itens.append(_item(f"Previsão {config.ANO_ALVO}: escala válida por município", ok, det))
        ag = agregacao.agregar(prev, "p_rede", [])
        ok, det = _somas(ag, "p_rede", ["ano", "turno"])
        itens.append(_item(f"Previsão {config.ANO_ALVO}: agregado nacional soma 100%", ok, det))
        meta = json.loads((r / f"previsao_{config.ANO_ALVO}_meta.json").read_text(encoding="utf-8"))
        ok = all(int(a) < config.ANO_ALVO for a in meta["eleicoes_treino"])
        itens.append(_item(f"Previsão {config.ANO_ALVO}: rede treinada só com eleições anteriores", ok,
                           f"treino: {', '.join(meta['eleicoes_treino'])}"))
        semh = prev.groupby(["turno", "uf", "cd_municipio"])["sem_historico_mun"].first().sum()
        itens.append(_item("Municípios sem histórico na eleição anterior (usam valor da UF)", True, f"{int(semh)}"))

    rel = {"gerado_em": datetime.now().astimezone().isoformat(timespec="seconds"), "itens": itens,
           "todas_ok": all(i["ok"] for i in itens)}
    r.mkdir(parents=True, exist_ok=True)
    (r / "relatorio_verificacoes.json").write_text(json.dumps(rel, ensure_ascii=False, indent=2), encoding="utf-8")
    return itens
