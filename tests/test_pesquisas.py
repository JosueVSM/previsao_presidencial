import pandas as pd

from eleicao import config, features
from eleicao.avaliacao import carregar_contexto, disputas_disponiveis
from eleicao.partidos import normalizar_nome
from eleicao.pesquisas import COLUNAS_MANUAIS, media_nacional


def _pesquisas_ficticias(alvos, ctx):
    """Percentuais FICTÍCIOS, só para exercitar o código de pesquisas."""
    linhas = []
    for ano, turno in alvos:
        dia = pd.Timestamp(config.DATAS_ELEICAO[(ano, turno)])
        r = ctx.resultados[(ctx.resultados.ano == ano) & (ctx.resultados.turno == turno)]
        cands = r.drop_duplicates("candidato")
        for i, (_, c) in enumerate(cands.iterrows()):
            linhas.append({"ano": ano, "turno": turno, "nr_registro_tse": f"BR-{ano}{turno}", "instituto": "FICTICIO",
                           "data_inicio": dia - pd.Timedelta(days=10), "data_fim": dia - pd.Timedelta(days=8),
                           "abrangencia": "BR", "n_entrevistados": 2000, "margem_erro_pp": 2.0,
                           "candidato": c.nome_urna, "pct": 10.0 + i, "base": "totais"})
        # pesquisa fora da janela (deve ser ignorada)
        linhas.append({**linhas[-1], "data_fim": dia - pd.Timedelta(days=90), "pct": 99.0, "nr_registro_tse": "velha"})
    df = pd.DataFrame(linhas)[COLUNAS_MANUAIS]
    df["candidato_norm"] = df.candidato.map(normalizar_nome)
    return df


def test_pesquisas_opcionais(pipeline_completo):
    ctx = carregar_contexto()
    alvos = disputas_disponiveis(ctx)
    assert not features.pesquisas_habilitadas(ctx, alvos)  # sem arquivo -> desligado
    manuais = _pesquisas_ficticias(alvos, ctx)
    ctx.pesquisas_manuais = manuais
    assert features.pesquisas_habilitadas(ctx, alvos)
    assert not features.pesquisas_habilitadas(ctx, alvos + [(2026, 1)])  # sem pesquisa para 2026
    df = features.montar(ctx, 2014, 1, usar_pesquisas=True)
    por_cand = df.groupby("candidato").pesquisa_br.first()
    assert abs(por_cand.sum() - 1) < 1e-9  # renormalizado para base de válidos
    assert (por_cand < 0.5).all()           # a pesquisa "velha" (99%) ficou fora da janela
    nomes = {c: {normalizar_nome(u)} for c, u in zip(df.candidato, df.nome_urna)}
    assert media_nacional(manuais, 2014, 1, nomes) is not None
    assert set(features.COLS_PESQUISA) <= set(df.columns)
