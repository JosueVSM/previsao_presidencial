import numpy as np
import pandas as pd

from eleicao import config, features
from eleicao.avaliacao import carregar_contexto


def test_entradas_nao_dependem_do_resultado_previsto(pipeline_completo):
    """Vazamento: alterar os votos da eleição-alvo não pode mudar nenhuma entrada da rede."""
    ctx = carregar_contexto()
    a = features.montar(ctx, 2018, 1)
    rng = np.random.default_rng(0)
    res2 = ctx.resultados.copy()
    m = res2.ano == 2018
    res2.loc[m, "votos"] = (res2.loc[m, "votos"] * rng.uniform(0.1, 3, m.sum())).astype(int)
    ctx2 = features.Contexto(res2, ctx.comparecimento, ctx.eleitorado_alvo)
    b = features.montar(ctx2, 2018, 1)
    cols = features.colunas_entrada(False) + ["peso_agregacao"]
    pd.testing.assert_frame_equal(a[cols], b[cols])
    assert not np.allclose(a["y"], b["y"])  # só o alvo muda


def test_entradas_nao_usam_eleicoes_futuras(pipeline_completo):
    ctx = carregar_contexto()
    a = features.montar(ctx, 2014, 1)
    ctx2 = features.Contexto(ctx.resultados[ctx.resultados.ano <= 2014], ctx.comparecimento, None)
    b = features.montar(ctx2, 2014, 1)
    cols = features.colunas_entrada(False)
    pd.testing.assert_frame_equal(a[cols], b[cols])


def test_alvo_2026(pipeline_completo):
    ctx = carregar_contexto()
    cands = pd.read_parquet(config.DIR_PROCESSADOS / "candidatos_2026.parquet")
    df = features.montar(ctx, 2026, 1, candidatos_alvo=cands)
    assert df["y"].isna().all()
    assert set(df.nome_urna) == {"ALFA", "SIGMA", "OMEGA", "PSI"}  # INAPTO excluído
    novo = df[df.cd_municipio == 88888]
    assert len(novo) and (novo.sem_historico_mun == 1).all()
    # reeleição derivada dos dados oficiais da eleição anterior
    eleitos = features.eleito_em(ctx.resultados, 2022)
    assert set(df[df.cand_eleito_ant == 1].candidato) == eleitos & set(df.candidato)
    # 2º turno de 2026 ainda indefinido
    assert features.candidatos_da_disputa(ctx, 2026, 2, cands).empty


def test_resultado_do_ano_alvo_e_descartado(pipeline_completo, monkeypatch):
    """Mesmo que um resultado de 2026 apareça na base, ele não chega às entradas."""
    original = pd.read_parquet
    res = original(config.DIR_PROCESSADOS / "resultados_municipio.parquet")
    falso = res[res.ano == 2022].assign(ano=2026)

    def ler(caminho, *a, **k):
        if "resultados_municipio" in str(caminho):
            return pd.concat([res, falso])
        return original(caminho, *a, **k)

    monkeypatch.setattr(pd, "read_parquet", ler)
    ctx = carregar_contexto()
    assert ctx.resultados.ano.max() < 2026
