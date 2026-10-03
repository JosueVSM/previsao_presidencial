"""Pesquisas eleitorais.

O conjunto "Pesquisas eleitorais" do Portal de Dados Abertos do TSE traz o
REGISTRO das pesquisas (instituto, datas, amostra, metodologia, contratante),
mas NÃO os percentuais divulgados. Por isso:

1. O registro oficial é exibido no app como contexto (sem números de intenção).
2. Percentuais de pesquisa só entram se o usuário fornecer o arquivo
   dados/manual/pesquisas_resultados.csv (modelo em pesquisas_resultados_MODELO.csv),
   sempre rotulados como "Resultado de pesquisa" e nunca como resultado oficial.
3. A rede neural só usa pesquisas como entrada se esse arquivo cobrir TODAS as
   eleições de treino e de teste (senão ela aprenderia um peso para uma variável
   que nunca viu). Pesquisas nacionais entram como uma característica do
   candidato (igual para todos os municípios) — nunca como se fossem o
   resultado observado de cada município. Pesquisas estaduais não são usadas
   pela rede; aparecem apenas para comparação.
"""
from __future__ import annotations

import io
import zipfile
from datetime import timedelta

import numpy as np
import pandas as pd

from . import config, fontes
from .leitura import membros_csv
from .partidos import normalizar_nome

COLUNAS_MANUAIS = ["ano", "turno", "nr_registro_tse", "instituto", "data_inicio", "data_fim",
                   "abrangencia", "n_entrevistados", "margem_erro_pp", "candidato", "pct", "base"]


def _achar(cols: list[str], *padroes: str) -> str | None:
    for p in padroes:
        for c in cols:
            if p in c:
                return c
    return None


def preparar_registro(ano: int = config.ANO_ALVO) -> pd.DataFrame | None:
    arq = config.DIR_BRUTOS / f"pesquisa_eleitoral_{ano}.zip"
    if not arq.exists():
        print("  aviso: registro de pesquisas indisponível")
        return None
    partes = []
    with zipfile.ZipFile(arq) as zf:
        for m in membros_csv(zf):
            with zf.open(m) as f:
                partes.append(pd.read_csv(io.TextIOWrapper(f, encoding="latin-1"), sep=";", dtype=str,
                                          keep_default_na=False, on_bad_lines="warn"))
    if not partes:
        return None
    df = pd.concat(partes, ignore_index=True).drop_duplicates()
    df.columns = [c.strip().upper() for c in df.columns]
    cols = list(df.columns)
    c_cargo = _achar(cols, "DS_CARGO", "CARGO")
    if c_cargo:
        df = df[df[c_cargo].str.upper().str.contains("PRESIDENTE", na=False)]
    mapa = {
        "registro": _achar(cols, "NR_IDENTIFICACAO_PESQUISA", "NR_PROTOCOLO", "IDENTIFICACAO"),
        "instituto": _achar(cols, "NM_EMPRESA", "EMPRESA"),
        "data_registro": _achar(cols, "DT_REGISTRO"),
        "data_inicio": _achar(cols, "DT_INICIO_PESQUISA", "DT_INICIO"),
        "data_fim": _achar(cols, "DT_FIM_PESQUISA", "DT_FIM"),
        "data_divulgacao": _achar(cols, "DT_DIVULGACAO"),
        "abrangencia": _achar(cols, "NM_UE", "SG_UE", "SG_UF"),
        # O arquivo de 2026 traz QT_ENTREVISTADO (singular); anos anteriores, QT_ENTREVISTADOS.
        "n_entrevistados": _achar(cols, "QT_ENTREVISTADO", "ENTREVISTADO"),
        "metodologia": _achar(cols, "DS_METODOLOGIA", "METODOLOGIA"),
        "plano_amostral": _achar(cols, "DS_PLANO_AMOSTRAL", "PLANO_AMOSTRAL"),
        "margem_erro": _achar(cols, "MARGEM", "DS_MARGEM"),
    }
    out = pd.DataFrame({k: (df[v] if v else "") for k, v in mapa.items()})
    for c in ["data_registro", "data_inicio", "data_fim", "data_divulgacao"]:
        out[c] = pd.to_datetime(out[c], dayfirst=True, errors="coerce")
    out["n_entrevistados"] = pd.to_numeric(out["n_entrevistados"], errors="coerce")
    out = out.sort_values("data_fim", ascending=False).reset_index(drop=True)
    out.to_parquet(config.DIR_PROCESSADOS / f"pesquisas_registro_{ano}.parquet", index=False)
    fontes.registrar_uso(f"pesquisas_{ano}", [v for v in mapa.values() if v] + ([c_cargo] if c_cargo else []),
                         len(out), "Somente metadados de registro; o TSE não publica os percentuais.")
    return out


def carregar_manuais() -> pd.DataFrame | None:
    """Percentuais de pesquisas informados pelo usuário (opcional)."""
    arq = config.ARQ_PESQUISAS_MANUAIS
    if not arq.exists():
        return None
    df = pd.read_csv(arq, dtype={"nr_registro_tse": str})
    faltam = set(COLUNAS_MANUAIS) - set(df.columns)
    if faltam:
        raise ValueError(f"{arq.name}: faltam colunas {sorted(faltam)}")
    df["data_inicio"] = pd.to_datetime(df["data_inicio"], errors="coerce")
    df["data_fim"] = pd.to_datetime(df["data_fim"], errors="coerce")
    df["candidato_norm"] = df["candidato"].map(normalizar_nome)
    df["abrangencia"] = df["abrangencia"].astype(str).str.upper()
    df["pct"] = pd.to_numeric(df["pct"], errors="coerce")
    return df.dropna(subset=["pct", "data_fim"])


def media_nacional(manuais: pd.DataFrame | None, ano: int, turno: int,
                   nomes: dict[str, set[str]]) -> dict[str, float] | None:
    """Média de pesquisas NACIONAIS na janela pré-eleitoral, em base de votos válidos.

    `nomes` mapeia o id do candidato para os nomes aceitos (nome completo e de urna,
    normalizados). Cada pesquisa é renormalizada entre os candidatos da disputa
    (convertendo para base de válidos) e ponderada por √n_entrevistados.
    Retorna None se não houver pesquisa nacional na janela.
    """
    if manuais is None:
        return None
    dia = config.DATAS_ELEICAO.get((ano, turno))
    if dia is None:
        return None
    ini = pd.Timestamp(dia - timedelta(days=config.JANELA_PESQUISAS_DIAS))
    fim = pd.Timestamp(dia - timedelta(days=1))
    sel = manuais[(manuais.ano == ano) & (manuais.turno == turno) & (manuais.abrangencia == "BR")
                  & (manuais.data_fim >= ini) & (manuais.data_fim <= fim)]
    if sel.empty:
        return None
    inv = {n: cid for cid, ns in nomes.items() for n in ns}
    acum: dict[str, float] = {cid: 0.0 for cid in nomes}
    peso_total = 0.0
    for _, p in sel.groupby(["instituto", "nr_registro_tse", "data_fim"], dropna=False):
        p = p.assign(cid=p["candidato_norm"].map(inv)).dropna(subset=["cid"])
        tot = p["pct"].sum()
        if tot <= 0:
            continue
        w = float(np.sqrt(np.nan_to_num(p["n_entrevistados"].iloc[0], nan=1000.0)))
        for cid, v in p.groupby("cid")["pct"].sum().items():
            acum[cid] += w * v / tot
        peso_total += w
    if peso_total == 0:
        return None
    return {cid: v / peso_total for cid, v in acum.items()}
