"""Validação temporal (backtest) e previsão do ano-alvo.

Validação temporal: para cada eleição de teste T, a rede é treinada apenas
com eleições-alvo anteriores a T (as características dessas eleições, por
sua vez, só usam eleições ainda mais antigas) e avaliada em T, que fica
INTEIRAMENTE fora do treino — todos os municípios, os dois turnos. Nunca
misturamos municípios da mesma eleição entre treino e teste.
"""
from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd

from . import agregacao, config, features
from .modelo import ConjuntoMLP
from .pesquisas import carregar_manuais


def carregar_contexto() -> features.Contexto:
    p = config.DIR_PROCESSADOS
    res = pd.read_parquet(p / "resultados_municipio.parquet")
    comp = pd.read_parquet(p / "comparecimento.parquet") if (p / "comparecimento.parquet").exists() else None
    el = (pd.read_parquet(p / f"eleitorado_{config.ANO_ALVO}.parquet")
          if (p / f"eleitorado_{config.ANO_ALVO}.parquet").exists() else None)
    # Garantia contra vazamento: nenhum resultado do ano-alvo pode estar na base.
    res = res[res.ano < config.ANO_ALVO]
    return features.Contexto(res, comp, el, carregar_manuais())


def disputas_disponiveis(ctx: features.Contexto) -> list[tuple[int, int]]:
    anos = sorted(ctx.resultados.ano.unique())
    out = []
    for a in anos:
        if a < config.PRIMEIRO_ANO_ALVO or (a - config.INTERVALO) not in anos:
            continue
        for t in sorted(ctx.resultados[ctx.resultados.ano == a].turno.unique()):
            out.append((int(a), int(t)))
    return out


def _montar_varios(ctx, alvos, usar_pesq):
    partes = [features.montar(ctx, a, t, usar_pesquisas=usar_pesq) for a, t in alvos]
    return pd.concat([p for p in partes if len(p)], ignore_index=True)


def metricas(df: pd.DataFrame, col: str) -> dict:
    erro = (df[col] - df["y"]) * 100
    disp = df.groupby("id_disputa")
    w = disp["votos_validos_disputa"].first()
    mae_disp = (erro.abs()).groupby(df["id_disputa"]).mean()
    lider_prev = df.loc[df.groupby("id_disputa")[col].idxmax(), ["id_disputa", "candidato"]].set_index("id_disputa")
    lider_real = df.loc[df.groupby("id_disputa")["y"].idxmax(), ["id_disputa", "candidato"]].set_index("id_disputa")
    return {
        "mae_pp": float(erro.abs().mean()),
        "rmse_pp": float(np.sqrt((erro ** 2).mean())),
        "mae_pp_ponderado_votos": float(np.average(mae_disp, weights=w.loc[mae_disp.index])),
        "acerto_mais_votado_municipio_pct": float(100 * (lider_prev.candidato == lider_real.candidato).mean()),
        "n_municipios": int(disp.ngroups),
        "n_candidatos": int(df["candidato"].nunique()),
    }


def _nacional(df: pd.DataFrame, col: str, ctx, ano: int, turno: int) -> pd.DataFrame:
    prev = agregacao.agregar(df, col, [])
    of = agregacao.oficial(ctx.resultados, ano, turno, [])
    m = prev.merge(of[["candidato", "pct_oficial"]], on="candidato", how="left")
    m["erro_pp"] = (m[col] - m["pct_oficial"]) * 100
    return m


def backtest(verbose: bool = True) -> pd.DataFrame:
    ctx = carregar_contexto()
    disp = disputas_disponiveis(ctx)
    usar_pesq = features.pesquisas_habilitadas(ctx, disp)
    print(f"Pesquisas como entrada da rede: {'SIM' if usar_pesq else 'NÃO'}")
    colunas = features.colunas_entrada(usar_pesq)
    linhas, previsoes, nacionais = [], [], []
    for T in config.ANOS_TESTE_BACKTEST:
        teste = [(a, t) for a, t in disp if a == T]
        treino = [(a, t) for a, t in disp if a < T]
        if not teste or not treino:
            print(f"Backtest {T}: dados insuficientes — pulado")
            continue
        print(f"Backtest {T}: treino = {sorted({a for a, _ in treino})}, teste = {T}")
        df_tr = _montar_varios(ctx, treino, usar_pesq)
        df_te = _montar_varios(ctx, teste, usar_pesq)
        # Checagem de vazamento: nenhuma linha do ano de teste no treino.
        assert df_tr["ano"].max() < T and set(df_te["ano"]) == {T}
        conj = ConjuntoMLP(colunas).treinar(df_tr, verbose=verbose)
        df_te["p_rede"], df_te["p_rede_dp"] = conj.prever(df_te)
        df_te["p_persistencia"] = features.baseline_persistencia(df_te)
        df_te["p_uniforme"] = 1.0 / df_te.groupby("id_disputa")["candidato"].transform("count")
        for turno, g in df_te.groupby("turno"):
            for modelo, col in [("Rede neural (MLP)", "p_rede"),
                                ("Referência: persistência partidária", "p_persistencia"),
                                ("Referência: divisão igual", "p_uniforme")]:
                m = metricas(g, col)
                nac = _nacional(g, col, ctx, T, turno)
                m.update({
                    "eleicao_teste": T, "turno": int(turno), "modelo": modelo,
                    "eleicoes_treino": ", ".join(map(str, sorted({a for a, _ in treino}))),
                    "erro_nacional_mae_pp": float(nac["erro_pp"].abs().mean()),
                    "erro_nacional_max_pp": float(nac["erro_pp"].abs().max()),
                })
                if col == "p_rede":
                    m["epocas_medias"] = float(np.mean(conj.epocas))
                linhas.append(m)
                nacionais.append(nac.rename(columns={col: "pct_estimado"}).assign(modelo=modelo))
        previsoes.append(df_te[["ano", "turno", "uf", "cd_municipio", "nm_municipio", "candidato", "nome_urna",
                                "partido", "y", "p_rede", "p_rede_dp", "p_persistencia", "peso_agregacao",
                                "votos_validos_disputa"]])
    if not linhas:
        raise RuntimeError("Nenhum backtest pôde ser executado.")
    met = pd.DataFrame(linhas)
    cols = ["eleicao_teste", "turno", "modelo", "eleicoes_treino"]
    met = met[cols + [c for c in met.columns if c not in cols]]
    config.DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)
    met.to_csv(config.DIR_RESULTADOS / "avaliacao_metricas.csv", index=False)
    pd.concat(previsoes).to_parquet(config.DIR_RESULTADOS / "previsoes_backtest.parquet", index=False)
    pd.concat(nacionais).to_csv(config.DIR_RESULTADOS / "backtest_nacional.csv", index=False)
    return met


def prever_alvo(verbose: bool = True) -> pd.DataFrame | None:
    ctx = carregar_contexto()
    arq_c = config.DIR_PROCESSADOS / f"candidatos_{config.ANO_ALVO}.parquet"
    if not arq_c.exists():
        print("Candidatos do ano-alvo ausentes — previsão não realizada.")
        return None
    cands = pd.read_parquet(arq_c)
    disp = disputas_disponiveis(ctx)
    alvos = [(config.ANO_ALVO, 1)]
    if features.candidatos_da_disputa(ctx, config.ANO_ALVO, 2, cands).shape[0] == 2:
        alvos.append((config.ANO_ALVO, 2))
    usar_pesq = features.pesquisas_habilitadas(ctx, disp + alvos)
    colunas = features.colunas_entrada(usar_pesq)
    print(f"Treinando rede final com {sorted({a for a, _ in disp})}; pesquisas: {'SIM' if usar_pesq else 'NÃO'}")
    df_tr = _montar_varios(ctx, disp, usar_pesq)
    conj = ConjuntoMLP(colunas).treinar(df_tr, verbose=verbose)
    conj.salvar(config.DIR_MODELOS / f"mlp_{config.ANO_ALVO}")

    saidas = []
    for ano, turno in alvos:
        df = features.montar(ctx, ano, turno, candidatos_alvo=cands, usar_pesquisas=usar_pesq)
        assert df["y"].isna().all(), "Vazamento: há alvo observado no ano-alvo."
        df["p_rede"], df["p_rede_dp"] = conj.prever(df)
        df["p_persistencia"] = features.baseline_persistencia(df)
        saidas.append(df)
    prev = pd.concat(saidas, ignore_index=True)
    prev[["ano", "turno", "uf", "cd_municipio", "nm_municipio", "candidato", "nome_urna", "partido", "linhagem",
          "p_rede", "p_rede_dp", "p_persistencia", "peso_agregacao", "eleitores", "sem_historico_mun"]
         ].to_parquet(config.DIR_RESULTADOS / f"previsao_{config.ANO_ALVO}.parquet", index=False)
    meta = {
        "gerado_em": datetime.now().astimezone().isoformat(timespec="seconds"),
        "ano_alvo": config.ANO_ALVO,
        "turnos_estimados": [t for _, t in alvos],
        "pesquisas_como_entrada": usar_pesq,
        "eleicoes_treino": conj.eleicoes_treino,
        "eleicoes_validacao_interna": conj.eleicoes_validacao,
        "epocas_por_semente": conj.epocas,
        "colunas_entrada": colunas,
        "n_candidatos": int(prev[prev.turno == 1]["candidato"].nunique()),
        "n_municipios": int(prev[prev.turno == 1]["id_disputa"].nunique() if "id_disputa" in prev else 0),
    }
    (config.DIR_RESULTADOS / f"previsao_{config.ANO_ALVO}_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return prev
