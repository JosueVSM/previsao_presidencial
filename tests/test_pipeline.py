import numpy as np
import pandas as pd

from eleicao import agregacao, config


def test_verificacoes_passam(pipeline_completo):
    falhas = [i for i in pipeline_completo["verificacoes"] if not i["ok"]]
    assert not falhas, falhas


def test_backtest_temporal(pipeline_completo):
    met = pipeline_completo["metricas"]
    rede = met[met.modelo == "Rede neural (MLP)"]
    assert set(rede.eleicao_teste) == set(config.ANOS_TESTE_BACKTEST)
    for te, tr in zip(rede.eleicao_teste, rede.eleicoes_treino):
        assert max(int(a) for a in tr.split(", ")) < te
    assert rede[["mae_pp", "rmse_pp", "erro_nacional_mae_pp"]].notna().all().all()


def test_agregacao_com_votos_reais_reproduz_oficial(pipeline_completo):
    """Ponderando as % municipais oficiais pelos votos válidos reais, a agregação
    tem de reproduzir exatamente a % nacional oficial (prova do critério de peso)."""
    b = pd.read_parquet(config.DIR_RESULTADOS / "previsoes_backtest.parquet")
    b = b[(b.ano == 2022) & (b.turno == 1)]
    ag = agregacao.agregar(b, "y", [], peso="votos_validos_disputa")
    res = pd.read_parquet(config.DIR_PROCESSADOS / "resultados_municipio.parquet")
    of = agregacao.oficial(res, 2022, 1, [])
    m = ag.merge(of, on="candidato")
    assert np.allclose(m["y"], m["pct_oficial"], atol=1e-9)


def test_previsao_2026(pipeline_completo):
    prev = pipeline_completo["previsao"]
    assert set(prev.turno) == {1}
    s = prev.groupby(["uf", "cd_municipio"]).p_rede.sum()
    assert np.allclose(s, 1, atol=1e-5)
    assert (prev.peso_agregacao > 0).all()


def test_app_streamlit(pipeline_completo):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(config.Path(__file__).resolve().parent.parent / "app.py"), default_timeout=120)
    at.run()
    assert not at.exception
    for nivel in ["Estado", "Município"]:
        at.sidebar.radio[1].set_value(nivel)
        at.run()
        assert not at.exception, nivel
    at.sidebar.radio[0].set_value(2)
    at.run()
    assert not at.exception
    at.sidebar.selectbox[0].set_value(at.sidebar.selectbox[0].options[-1])  # teste retrospectivo
    at.run()
    assert not at.exception
