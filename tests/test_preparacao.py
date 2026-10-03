import numpy as np
import pandas as pd

from eleicao import config
from eleicao.partidos import linhagem, normalizar_nome, normalizar_sigla


def test_linhagem_partidaria():
    assert linhagem("PMDB") == "MDB"
    assert linhagem("UNIÃO") == "UNIAO" and linhagem("PSL") == "UNIAO" and linhagem("PFL") == "UNIAO"
    assert linhagem("PC do B") == "PCDOB"
    assert linhagem("PR") == "PL" and linhagem("PRONA") == "PL"
    assert linhagem("NOVO") == "NOVO"  # sem entrada na tabela -> a própria sigla
    assert normalizar_sigla(" pt do b ") == "PTDOB"
    assert normalizar_nome("José  da Silva-Júnior") == "JOSE DA SILVA JUNIOR"


def test_preparacao(pipeline_completo, dados_sinteticos):
    res = pd.read_parquet(config.DIR_PROCESSADOS / "resultados_municipio.parquet")
    # Arquivo _BRASIL + arquivos por UF: votos não podem ser contados em dobro;
    # linhas de outros cargos devem ser descartadas.
    for ano, bruto in dados_sinteticos.items():
        pres = bruto[bruto.CD_CARGO == 1]
        esperado = pres.groupby("NR_TURNO").QT_VOTOS_NOMINAIS.sum()
        obtido = res[res.ano == ano].groupby("turno").votos.sum()
        assert (esperado.values == obtido.values).all(), ano
    # Arquivo de 1998 sem cabeçalho (layout antigo) foi lido e padronizado
    r98 = res[res.ano == 1998]
    assert len(r98) > 0 and "PL" in set(r98.linhagem)  # PRONA -> PL
    # % válidos somam 100 por município
    s = res.groupby(["ano", "turno", "uf", "cd_municipio"]).pct_validos.sum()
    assert np.allclose(s, 100)
    # zonas agregadas: chave única por município
    assert not res.duplicated(["ano", "turno", "uf", "cd_municipio", "candidato"]).any()


def test_candidatos_2026(pipeline_completo):
    c = pd.read_parquet(config.DIR_PROCESSADOS / "candidatos_2026.parquet")
    assert set(c[c.na_urna].nome_urna) == {"ALFA", "SIGMA", "OMEGA", "PSI"}
    assert "GOV SINTETICO" not in set(c.candidato)  # só cargo de Presidente


def test_fontes_registram_colunas(pipeline_completo):
    from eleicao.fontes import catalogo
    cat = catalogo()
    assert "QT_VOTOS_NOMINAIS_VALIDOS" in cat["votacao_2022"]["colunas_utilizadas"]
    assert cat["pesquisas_2026"]["linhas_utilizadas"] == 1  # só a de Presidente
