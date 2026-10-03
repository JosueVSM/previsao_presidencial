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

def test_referencia_persistencia_usa_o_2o_turno_anterior(pipeline_completo):
    """No 2º turno, a referência tem de repetir o 2º turno anterior, não o 1º.

    Sem isso a comparação favorece a rede: a referência disputaria o 2º turno com
    as fatias de uma eleição de muitos candidatos.
    """
    from eleicao import features
    from eleicao.avaliacao import carregar_contexto

    ctx = carregar_contexto()
    anos2t = sorted({a for a, t in [(int(a), int(t)) for a, t in
                                    ctx.resultados[["ano", "turno"]].drop_duplicates().values]
                     if t == 2})
    alvo = next(a for a in anos2t if (a - config.INTERVALO) in anos2t)
    df = features.montar(ctx, alvo, 2)
    p = features.baseline_persistencia(df)

    usa_2t = (df.segundo_turno == 1) & (df.partido_no_2t_ant == 1)
    assert usa_2t.any(), "cenário não exercitado: nenhum partido veio do 2º turno anterior"

    # Onde o partido esteve no 2º turno anterior, a referência segue a coluna de 2º
    # turno — e não a de 1º, que é diferente.
    esperado = df.partido_ant_mun_2t.where(usa_2t,
                                           np.maximum(df.partido_ant_mun_1t, df.cand_ant_mun_1t)) + 0.001
    esperado = esperado / esperado.groupby(df.id_disputa).transform("sum")
    assert np.allclose(p, esperado)
    difere = ~np.isclose(df.partido_ant_mun_2t, np.maximum(df.partido_ant_mun_1t, df.cand_ant_mun_1t))
    assert (usa_2t & difere).any(), "as duas colunas são iguais: o teste não provaria nada"

    # Continua sendo uma distribuição válida por município.
    assert np.allclose(pd.Series(p).groupby(df.id_disputa.values).sum(), 1)


def test_agregacao_nao_parte_candidato_com_rotulo_divergente(pipeline_completo):
    """Nome de urna e sigla são rótulos: variar entre municípios não pode dividir a barra."""
    b = pd.read_parquet(config.DIR_RESULTADOS / "previsoes_backtest.parquet")
    b = b[(b.ano == b.ano.max()) & (b.turno == 1)].copy()
    certo = agregacao.agregar(b, "p_rede", [])

    # Simula o TSE grafando o nome de urna de um jeito em metade dos municípios.
    alvo = certo.sort_values("p_rede", ascending=False).candidato.iloc[0]
    muns = sorted(b.cd_municipio.unique())
    metade = b.cd_municipio.isin(muns[: len(muns) // 2])
    sujo = b.assign(nome_urna=b.nome_urna.where(~(metade & (b.candidato == alvo)),
                                                b.nome_urna + " JR"))
    obtido = agregacao.agregar(sujo, "p_rede", [])

    assert len(obtido) == len(certo), "candidato foi partido em duas linhas"
    assert obtido.candidato.is_unique
    m = certo[["candidato", "p_rede"]].merge(obtido[["candidato", "p_rede"]], on="candidato",
                                             suffixes=("_certo", "_obtido"))
    assert len(m) == len(certo)
    assert np.allclose(m.p_rede_certo, m.p_rede_obtido), "a porcentagem mudou com o rótulo"
    assert np.isclose(obtido.p_rede.sum(), 1)


def test_amostra_do_repositorio_cobre_o_que_o_app_abre(pipeline_completo):
    """Depois de gerada, a amostra tem de bastar para o app abrir sem dados/processados."""
    from eleicao import amostra

    amostra.gerar()
    assert amostra.disponivel()
    for nome in ["resultados_municipio.parquet", f"candidatos_{config.ANO_ALVO}.parquet",
                 "previsoes_backtest.parquet", "avaliacao_metricas.csv",
                 "backtest_nacional.csv", "relatorio_verificacoes.json",
                 f"previsao_{config.ANO_ALVO}.parquet", f"previsao_{config.ANO_ALVO}_meta.json",
                 "modelo.json", "AMOSTRA.json"]:
        assert (config.DIR_AMOSTRA / nome).exists(), nome

    # A tabela de métricas vai inteira; só o backtest navegável é recortado.
    met = pd.read_csv(config.DIR_AMOSTRA / "avaliacao_metricas.csv")
    assert set(met.eleicao_teste) == set(config.ANOS_TESTE_BACKTEST)
    bt = pd.read_parquet(config.DIR_AMOSTRA / "previsoes_backtest.parquet")
    assert bt.ano.nunique() == 1 and len(bt) > 0
    assert np.allclose(bt.groupby(["turno", "uf", "cd_municipio"]).p_rede.sum(), 1)
