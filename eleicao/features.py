"""Construção das entradas da rede neural, sem vazamento temporal.

Unidade de previsão: uma DISPUTA = (eleição, turno, município). Dentro de
cada disputa há uma linha por candidato. Para prever a eleição do ano T
usamos SOMENTE informação disponível antes dela:

* resultados oficiais da eleição presidencial anterior (L = T − 4) e
  candidaturas de eleições anteriores (≤ L);
* a lista de candidatos de T (conhecida no registro das candidaturas);
* o eleitorado apto de T (cadastro fechado ~150 dias antes do pleito);
* opcionalmente, médias de pesquisas nacionais divulgadas antes de T.

O resultado de T (votos) só é usado como ALVO do treino e na avaliação —
nunca como entrada. Para 2026, nenhum dado de resultado de 2026 é lido.

Representação numérica de candidatos e partidos
------------------------------------------------
A rede não recebe o nome do candidato nem um "one-hot" de partido (isso
não generalizaria para candidatos novos). Cada candidato é descrito por
quanto o seu partido (pela linhagem, ver partidos.py) e ele próprio
obtiveram na eleição anterior no mesmo município, na UF e no país, por
indicadores de "concorreu antes", "foi eleito na anterior" (reeleição) e
"partido do eleito anterior". Características do município (região,
eleitorado, comparecimento, fragmentação) são repetidas em todas as linhas
da disputa — elas só influenciam a previsão via interação com as do candidato,
o que a MLP consegue aprender.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config
from .partidos import sem_acento

EPS = 0.005

COLS_CANDIDATO = [
    "partido_ant_mun_1t", "partido_ant_uf_1t", "partido_ant_br_1t", "log_partido_ant_mun_1t",
    "partido_concorreu_ant", "partido_ant_mun_2t", "partido_no_2t_ant",
    "cand_ant_mun_1t", "cand_ant_br_1t", "cand_concorreu_ant", "cand_concorreu_alguma_vez",
    "cand_eleito_ant", "partido_do_eleito_ant",
]
COLS_PESQUISA = ["pesquisa_br", "log_pesquisa_br", "cand_fora_pesquisa"]
COLS_DISPUTA = ["segundo_turno", "n_candidatos", "log_eleitorado", "comparecimento_ant",
                "fragmentacao_ant", "sem_historico_mun"] + [f"regiao_{r}" for r in config.LISTA_REGIOES]


@dataclass
class Contexto:
    resultados: pd.DataFrame            # resultados_municipio (todas as eleições históricas)
    comparecimento: pd.DataFrame | None  # aptos/comparecimento por município
    eleitorado_alvo: pd.DataFrame | None  # eleitorado do ano-alvo (2026)
    pesquisas_manuais: pd.DataFrame | None = None


def _norm(s: pd.Series) -> pd.Series:
    return s.fillna("").map(lambda x: sem_acento(str(x)).upper().strip())


def eleito_em(res: pd.DataFrame, ano: int) -> set[str]:
    r = res[res.ano == ano]
    if r.empty:
        return set()
    sit = _norm(r["situacao_turno"])
    eleitos = set(r.loc[sit == "ELEITO", "candidato"])
    if eleitos:
        return eleitos
    turno_final = r.turno.max()
    tot = r[r.turno == turno_final].groupby("candidato")["votos"].sum()
    return {tot.idxmax()}


def _shares(df: pd.DataFrame, chave: list[str], item: str) -> pd.DataFrame:
    v = df.groupby(chave + [item], as_index=False)["votos"].sum()
    tot = v.groupby(chave)["votos"].transform("sum") if chave else v["votos"].sum()
    v["share"] = v["votos"] / tot
    return v.drop(columns="votos")


def _taxas_ant(ctx: Contexto, L: int, turno: int) -> pd.DataFrame:
    """Comparecimento e votos válidos por município na eleição anterior."""
    res = ctx.resultados
    tt = turno if not res[(res.ano == L) & (res.turno == turno)].empty else 1
    val = (res[(res.ano == L) & (res.turno == tt)]
           .groupby(["uf", "cd_municipio"], as_index=False)["votos"].sum()
           .rename(columns={"votos": "validos_ant"}))
    if ctx.comparecimento is not None:
        c = ctx.comparecimento[(ctx.comparecimento.ano == L) & (ctx.comparecimento.turno == tt)]
        val = val.merge(c[["uf", "cd_municipio", "aptos", "comparecimento"]], on=["uf", "cd_municipio"], how="left")
        val["comparecimento_ant"] = val["comparecimento"] / val["aptos"].replace(0, np.nan)
        val["validos_por_apto_ant"] = val["validos_ant"] / val["aptos"].replace(0, np.nan)
        val = val.rename(columns={"aptos": "aptos_ant"}).drop(columns="comparecimento")
    else:
        val["comparecimento_ant"] = np.nan
        val["validos_por_apto_ant"] = np.nan
        val["aptos_ant"] = np.nan
    return val


def _eleitorado_alvo(ctx: Contexto, T: int, turno: int) -> pd.DataFrame | None:
    if T == config.ANO_ALVO and ctx.eleitorado_alvo is not None:
        return ctx.eleitorado_alvo[["uf", "cd_municipio", "eleitores"]]
    if ctx.comparecimento is not None:
        c = ctx.comparecimento[(ctx.comparecimento.ano == T) & (ctx.comparecimento.turno == turno)]
        if not c.empty:
            return c[["uf", "cd_municipio"]].assign(eleitores=c["aptos"].values)
    return None


def candidatos_da_disputa(ctx: Contexto, T: int, turno: int,
                          candidatos_alvo: pd.DataFrame | None) -> pd.DataFrame:
    if T == config.ANO_ALVO:
        if candidatos_alvo is None:
            return pd.DataFrame()
        c = candidatos_alvo[candidatos_alvo.na_urna]
        if turno == 2:
            sit = _norm(c["situacao_turno"])
            c = c[sit.str.startswith("2") & sit.str.contains("TURNO")]
        return c[["candidato", "nome_urna", "partido", "linhagem"]].drop_duplicates("candidato")
    r = ctx.resultados[(ctx.resultados.ano == T) & (ctx.resultados.turno == turno)]
    tot = r.groupby("candidato")["votos"].sum()
    validos = tot[tot > 0].index  # candidaturas com votos válidos
    return (r[r.candidato.isin(validos)][["candidato", "nome_urna", "partido", "linhagem"]]
            .drop_duplicates("candidato"))


def pesquisas_habilitadas(ctx: Contexto, alvos: list[tuple[int, int]]) -> bool:
    """A variável de pesquisa só é ligada se TODAS as disputas (treino e alvo) tiverem
    ao menos uma pesquisa nacional na janela pré-eleitoral."""
    m = ctx.pesquisas_manuais
    if m is None:
        return False
    for ano, turno in alvos:
        dia = config.DATAS_ELEICAO.get((ano, turno))
        sel = m[(m.ano == ano) & (m.turno == turno) & (m.abrangencia == "BR")]
        if dia is None or sel.empty:
            return False
        ini = pd.Timestamp(dia) - pd.Timedelta(days=config.JANELA_PESQUISAS_DIAS)
        if not ((sel.data_fim >= ini) & (sel.data_fim < pd.Timestamp(dia))).any():
            return False
    return True


def montar(ctx: Contexto, T: int, turno: int, candidatos_alvo: pd.DataFrame | None = None,
           usar_pesquisas: bool = False) -> pd.DataFrame:
    """Monta a tabela longa (disputa × candidato) para o ano T e turno dado."""
    L = T - config.INTERVALO
    res = ctx.resultados
    hist = res[res.ano <= L]  # tudo que é anterior a T
    if hist[hist.ano == L].empty:
        raise ValueError(f"Sem resultados da eleição anterior ({L}) para prever {T}.")
    cands = candidatos_da_disputa(ctx, T, turno, candidatos_alvo)
    if cands.empty:
        return pd.DataFrame()

    # --- municípios (disputas) -------------------------------------------------
    ant1 = hist[(hist.ano == L) & (hist.turno == 1)]
    mun_ant = ant1[["uf", "cd_municipio", "nm_municipio"]].drop_duplicates(["uf", "cd_municipio"])
    if T == config.ANO_ALVO:
        munis = mun_ant.copy()
        if ctx.eleitorado_alvo is not None:
            novos = ctx.eleitorado_alvo[["uf", "cd_municipio", "nm_municipio"]]
            munis = pd.concat([munis, novos]).drop_duplicates(["uf", "cd_municipio"], keep="last")
    else:
        rT = res[(res.ano == T) & (res.turno == turno)]
        munis = rT[["uf", "cd_municipio", "nm_municipio"]].drop_duplicates(["uf", "cd_municipio"])
    munis = munis.reset_index(drop=True)

    base = munis.merge(cands, how="cross")
    base["ano"], base["turno"] = T, turno

    # --- desempenho anterior do partido (linhagem) -----------------------------
    pm = _shares(ant1, ["uf", "cd_municipio"], "linhagem").rename(columns={"share": "partido_ant_mun_1t"})
    pu = _shares(ant1, ["uf"], "linhagem").rename(columns={"share": "partido_ant_uf_1t"})
    pb = _shares(ant1, [], "linhagem").rename(columns={"share": "partido_ant_br_1t"})
    base = (base.merge(pm, on=["uf", "cd_municipio", "linhagem"], how="left")
                .merge(pu, on=["uf", "linhagem"], how="left")
                .merge(pb, on=["linhagem"], how="left"))
    base["sem_historico_mun"] = (~base.set_index(["uf", "cd_municipio"]).index.isin(
        mun_ant.set_index(["uf", "cd_municipio"]).index)).astype(float)
    # Município sem histórico (criado depois): usa o valor da UF.
    sem = base["sem_historico_mun"] == 1
    base.loc[sem, "partido_ant_mun_1t"] = base.loc[sem, "partido_ant_uf_1t"]
    base["partido_concorreu_ant"] = base["linhagem"].isin(set(ant1.linhagem)).astype(float)
    for c in ["partido_ant_mun_1t", "partido_ant_uf_1t", "partido_ant_br_1t"]:
        base[c] = base[c].fillna(0.0)
    base["log_partido_ant_mun_1t"] = np.log(base["partido_ant_mun_1t"] + EPS)

    ant2 = hist[(hist.ano == L) & (hist.turno == 2)]
    if not ant2.empty:
        pm2 = _shares(ant2, ["uf", "cd_municipio"], "linhagem").rename(columns={"share": "partido_ant_mun_2t"})
        base = base.merge(pm2, on=["uf", "cd_municipio", "linhagem"], how="left")
        pu2 = _shares(ant2, ["uf"], "linhagem").rename(columns={"share": "_pu2"})
        base = base.merge(pu2, on=["uf", "linhagem"], how="left")
        base["partido_ant_mun_2t"] = base["partido_ant_mun_2t"].fillna(base["_pu2"]).fillna(0.0)
        base = base.drop(columns="_pu2")
        base["partido_no_2t_ant"] = base["linhagem"].isin(set(ant2.linhagem)).astype(float)
    else:
        base["partido_ant_mun_2t"] = 0.0
        base["partido_no_2t_ant"] = 0.0

    # --- desempenho anterior do próprio candidato ------------------------------
    cm = _shares(ant1, ["uf", "cd_municipio"], "candidato").rename(columns={"share": "cand_ant_mun_1t"})
    cb = _shares(ant1, [], "candidato").rename(columns={"share": "cand_ant_br_1t"})
    base = base.merge(cm, on=["uf", "cd_municipio", "candidato"], how="left").merge(cb, on="candidato", how="left")
    cu = _shares(ant1, ["uf"], "candidato").rename(columns={"share": "_cu"})
    base = base.merge(cu, on=["uf", "candidato"], how="left")
    base["cand_ant_mun_1t"] = base["cand_ant_mun_1t"].fillna(base["_cu"]).fillna(0.0)
    base = base.drop(columns="_cu")
    base["cand_ant_br_1t"] = base["cand_ant_br_1t"].fillna(0.0)
    base["cand_concorreu_ant"] = base["candidato"].isin(set(ant1.candidato)).astype(float)
    base["cand_concorreu_alguma_vez"] = base["candidato"].isin(set(hist.candidato)).astype(float)
    eleitos = eleito_em(hist, L)
    base["cand_eleito_ant"] = base["candidato"].isin(eleitos).astype(float)
    part_eleito = set(hist[(hist.ano == L) & hist.candidato.isin(eleitos)].linhagem)
    base["partido_do_eleito_ant"] = base["linhagem"].isin(part_eleito).astype(float)

    # --- características do município -----------------------------------------
    taxas = _taxas_ant(ctx, L, turno)
    # Fragmentação = número efetivo de candidatos (1 / Σ share²) na eleição anterior.
    frag = _shares(ant1, ["uf", "cd_municipio"], "candidato")
    frag = frag.assign(s2=frag.share ** 2).groupby(["uf", "cd_municipio"], as_index=False)["s2"].sum()
    frag["fragmentacao_ant"] = 1.0 / frag.pop("s2")
    mun = munis[["uf", "cd_municipio"]].merge(taxas, on=["uf", "cd_municipio"], how="left") \
                                       .merge(frag, on=["uf", "cd_municipio"], how="left")
    el = _eleitorado_alvo(ctx, T, turno)
    if el is not None:
        mun = mun.merge(el, on=["uf", "cd_municipio"], how="left")
    else:
        mun["eleitores"] = np.nan
    # Taxas ausentes: média da UF; depois média nacional.
    for c in ["comparecimento_ant", "validos_por_apto_ant", "fragmentacao_ant"]:
        mun[c] = mun[c].fillna(mun.groupby("uf")[c].transform("mean")).fillna(mun[c].mean())
    mun["eleitores"] = mun["eleitores"].fillna(mun["aptos_ant"])
    # Peso da agregação = votos válidos esperados (ver agregacao.py).
    esperado = mun["eleitores"] * mun["validos_por_apto_ant"]
    mun["peso_agregacao"] = esperado.fillna(mun["validos_ant"]).fillna(0.0)
    mun["eleitores"] = mun["eleitores"].fillna(mun["validos_ant"])
    mun["log_eleitorado"] = np.log1p(mun["eleitores"].fillna(0.0))
    base = base.merge(mun[["uf", "cd_municipio", "comparecimento_ant", "fragmentacao_ant",
                           "log_eleitorado", "peso_agregacao", "eleitores"]],
                      on=["uf", "cd_municipio"], how="left")
    base["comparecimento_ant"] = base["comparecimento_ant"].fillna(0.8)
    base["fragmentacao_ant"] = base["fragmentacao_ant"].fillna(2.0)
    base["segundo_turno"] = float(turno == 2)
    base["n_candidatos"] = float(len(cands))
    reg = base["uf"].map(config.regiao)
    for r in config.LISTA_REGIOES:
        base[f"regiao_{r}"] = (reg == r).astype(float)

    # --- pesquisas (opcional) --------------------------------------------------
    if usar_pesquisas:
        from .pesquisas import media_nacional
        from .partidos import normalizar_nome
        nomes = {c: {c, normalizar_nome(u)} for c, u in zip(cands.candidato, cands.nome_urna)}
        med = media_nacional(ctx.pesquisas_manuais, T, turno, nomes) or {}
        base["pesquisa_br"] = base["candidato"].map(med).fillna(0.0)
        base["cand_fora_pesquisa"] = (~base["candidato"].isin([k for k, v in med.items() if v > 0])).astype(float)
        base["log_pesquisa_br"] = np.log(base["pesquisa_br"] + EPS)

    # --- alvo (somente eleições já realizadas) ---------------------------------
    if T != config.ANO_ALVO:
        rT = res[(res.ano == T) & (res.turno == turno)][["uf", "cd_municipio", "candidato", "votos"]]
        base = base.merge(rT, on=["uf", "cd_municipio", "candidato"], how="left")
        base["votos"] = base["votos"].fillna(0)
        tot = base.groupby(["uf", "cd_municipio"])["votos"].transform("sum")
        base = base[tot > 0].copy()
        base["votos_validos_disputa"] = base.groupby(["uf", "cd_municipio"])["votos"].transform("sum")
        base["y"] = base["votos"] / base["votos_validos_disputa"]
    else:
        base["votos"] = np.nan
        base["votos_validos_disputa"] = np.nan
        base["y"] = np.nan

    base["id_disputa"] = (base["ano"].astype(str) + "-" + base["turno"].astype(str) + "-"
                          + base["uf"] + "-" + base["cd_municipio"].astype(str))
    return base.sort_values(["id_disputa", "candidato"]).reset_index(drop=True)


def colunas_entrada(usar_pesquisas: bool) -> list[str]:
    return COLS_CANDIDATO + (COLS_PESQUISA if usar_pesquisas else []) + COLS_DISPUTA


def baseline_persistencia(df: pd.DataFrame) -> np.ndarray:
    """Referência simples: repete a votação anterior do candidato/partido no município,
    renormalizada entre os candidatos atuais (com piso pequeno para quem não concorreu)."""
    s = np.maximum(df["partido_ant_mun_1t"].values, df["cand_ant_mun_1t"].values) + 0.001
    tot = pd.Series(s).groupby(df["id_disputa"].values).transform("sum").values
    return s / tot
