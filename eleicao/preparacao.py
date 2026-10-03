"""Preparação dos dados: dos .zip do TSE para tabelas limpas (parquet).

Fórmula da porcentagem de votos válidos (por disputa = eleição × turno × município):

    pct_validos(c) = 100 × votos_válidos(c) / Σ_k votos_válidos(k)

em que a soma percorre todos os candidatos da disputa. Votos brancos e nulos
não entram (é a mesma definição usada pelo TSE para votos válidos). Quando o
arquivo traz QT_VOTOS_NOMINAIS_VALIDOS, usamos essa coluna, que já exclui
votos dados a candidaturas anuladas; caso contrário usamos QT_VOTOS_NOMINAIS
e descartamos linhas cuja destinação indique voto anulado.
"""
from __future__ import annotations

import pandas as pd

from . import config, fontes
from .leitura import filtro_presidente, ler_zip_tse, para_inteiro
from .partidos import linhagem, normalizar_nome, normalizar_sigla

COLS_VOTACAO = [
    "ANO_ELEICAO", "NR_TURNO", "CD_TIPO_ELEICAO", "CD_CARGO", "DS_CARGO", "SG_UF",
    "CD_MUNICIPIO", "NM_MUNICIPIO", "NR_ZONA", "NR_CANDIDATO", "NM_CANDIDATO",
    "NM_URNA_CANDIDATO", "SG_PARTIDO", "QT_VOTOS_NOMINAIS", "QT_VOTOS_NOMINAIS_VALIDOS",
    "NM_TIPO_DESTINACAO_VOTOS", "DS_SIT_TOT_TURNO", "ST_VOTO_EM_TRANSITO",
]
COLS_DETALHE = ["ANO_ELEICAO", "NR_TURNO", "CD_TIPO_ELEICAO", "CD_CARGO", "DS_CARGO", "SG_UF",
                "CD_MUNICIPIO", "NR_ZONA", "QT_APTOS", "QT_COMPARECIMENTO"]
COLS_CAND = ["ANO_ELEICAO", "NR_TURNO", "CD_TIPO_ELEICAO", "CD_CARGO", "DS_CARGO", "NR_CANDIDATO",
             "NM_CANDIDATO", "NM_URNA_CANDIDATO", "SG_PARTIDO", "NM_PARTIDO", "SG_FEDERACAO",
             "DS_COMPOSICAO_COLIGACAO", "DS_SITUACAO_CANDIDATURA", "DS_DETALHE_SITUACAO_CAND",
             "DS_SIT_TOT_TURNO", "ST_REELEICAO"]
COLS_ELEITORADO = ["SG_UF", "CD_MUNICIPIO", "NM_MUNICIPIO", "QT_ELEITORES_PERFIL"]


def _ordinaria(df: pd.DataFrame) -> pd.DataFrame:
    # CD_TIPO_ELEICAO 2 = eleição ordinária (exclui suplementares).
    if "CD_TIPO_ELEICAO" in df.columns and df["CD_TIPO_ELEICAO"].isin(["2"]).any():
        return df[df["CD_TIPO_ELEICAO"] == "2"]
    return df


def preparar_votacao(ano: int) -> pd.DataFrame:
    arq = config.DIR_BRUTOS / f"votacao_candidato_munzona_{ano}.zip"
    if not arq.exists():
        raise FileNotFoundError(arq)
    df = ler_zip_tse(arq, COLS_VOTACAO, tipo_layout="votacao", filtro=filtro_presidente)
    df = _ordinaria(df)
    if "ANO_ELEICAO" in df.columns:
        df = df[df["ANO_ELEICAO"] == str(ano)]
    if df.empty:
        raise ValueError(f"Nenhuma linha de Presidente encontrada em {arq.name}")

    if "QT_VOTOS_NOMINAIS_VALIDOS" in df.columns:
        df["votos"] = para_inteiro(df["QT_VOTOS_NOMINAIS_VALIDOS"])
        col_votos = "QT_VOTOS_NOMINAIS_VALIDOS"
    else:
        if "NM_TIPO_DESTINACAO_VOTOS" in df.columns:
            dest = df["NM_TIPO_DESTINACAO_VOTOS"].str.upper()
            df = df[~dest.str.contains("ANULAD|NULO", na=False)]
        df["votos"] = para_inteiro(df["QT_VOTOS_NOMINAIS"])
        col_votos = "QT_VOTOS_NOMINAIS"

    df = df.assign(
        ano=ano,
        turno=para_inteiro(df["NR_TURNO"]),
        uf=df["SG_UF"].str.upper(),
        cd_municipio=para_inteiro(df["CD_MUNICIPIO"]),
        candidato=df["NM_CANDIDATO"].map(normalizar_nome),
        partido=df["SG_PARTIDO"].map(normalizar_sigla),
    )
    sit = df.get("DS_SIT_TOT_TURNO", pd.Series("", index=df.index)).fillna("").str.upper()
    df["situacao_turno"] = sit

    chave = ["ano", "turno", "uf", "cd_municipio", "candidato"]
    g = df.groupby(chave, as_index=False).agg(
        nm_municipio=("NM_MUNICIPIO", "first"),
        nr_candidato=("NR_CANDIDATO", "first"),
        nome_urna=("NM_URNA_CANDIDATO", "first"),
        partido=("partido", "first"),
        votos=("votos", "sum"),
    )
    # Situação final do candidato no turno (ELEITO / 2º TURNO / NÃO ELEITO): moda nacional.
    sit_cand = (df[df.situacao_turno != ""].groupby(["turno", "candidato"])["situacao_turno"]
                .agg(lambda s: s.mode().iat[0]).rename("situacao_turno").reset_index())
    g = g.merge(sit_cand, on=["turno", "candidato"], how="left")
    g["situacao_turno"] = g["situacao_turno"].fillna("")
    g["linhagem"] = g["partido"].map(linhagem)

    tot = g.groupby(["ano", "turno", "uf", "cd_municipio"])["votos"].transform("sum")
    g["votos_validos_disputa"] = tot
    g = g[tot > 0].copy()
    g["pct_validos"] = 100.0 * g["votos"] / g["votos_validos_disputa"]

    fontes.registrar_uso(f"votacao_{ano}",
                         [c for c in COLS_VOTACAO if c in df.columns and c != "QT_VOTOS_NOMINAIS"] + [col_votos],
                         len(df), "Filtro: CD_CARGO=1 (Presidente); eleição ordinária; agregado de zona para município.")
    return g


def preparar_detalhe(ano: int) -> pd.DataFrame | None:
    arq = config.DIR_BRUTOS / f"detalhe_votacao_munzona_{ano}.zip"
    if not arq.exists():
        return None
    try:
        df = ler_zip_tse(arq, COLS_DETALHE, tipo_layout="detalhe", filtro=filtro_presidente)
    except ValueError as e:
        print(f"  aviso: {e}")
        return None
    df = _ordinaria(df)
    if df.empty or "QT_APTOS" not in df.columns:
        return None
    df = df.assign(ano=ano, turno=para_inteiro(df["NR_TURNO"]), uf=df["SG_UF"].str.upper(),
                   cd_municipio=para_inteiro(df["CD_MUNICIPIO"]),
                   aptos=para_inteiro(df["QT_APTOS"]),
                   comparecimento=para_inteiro(df["QT_COMPARECIMENTO"]))
    g = df.groupby(["ano", "turno", "uf", "cd_municipio"], as_index=False)[["aptos", "comparecimento"]].sum()
    fontes.registrar_uso(f"detalhe_{ano}", [c for c in COLS_DETALHE if c in df.columns], len(df),
                         "Filtro: Presidente; agregado de zona para município.")
    return g


def preparar_candidatos(ano: int = config.ANO_ALVO) -> pd.DataFrame:
    arq = config.DIR_BRUTOS / f"consulta_cand_{ano}.zip"
    if not arq.exists():
        raise FileNotFoundError(arq)
    df = ler_zip_tse(arq, COLS_CAND, filtro=filtro_presidente)
    df = _ordinaria(df)
    if "ANO_ELEICAO" in df.columns:
        df = df[df["ANO_ELEICAO"] == str(ano)]
    df = df.assign(
        ano=ano,
        candidato=df["NM_CANDIDATO"].map(normalizar_nome),
        partido=df["SG_PARTIDO"].map(normalizar_sigla),
    )
    for c in ["DS_SITUACAO_CANDIDATURA", "DS_DETALHE_SITUACAO_CAND", "DS_SIT_TOT_TURNO",
              "SG_FEDERACAO", "DS_COMPOSICAO_COLIGACAO", "NM_PARTIDO", "ST_REELEICAO"]:
        if c not in df.columns:
            df[c] = ""
    df["linhagem"] = df["partido"].map(linhagem)
    situ = df["DS_SITUACAO_CANDIDATURA"].str.upper()
    # Candidaturas "APTO" constam da urna (inclui as sub judice com recurso).
    # Se o arquivo ainda não tiver nenhum "APTO" (fase inicial do registro),
    # aceitamos tudo que não esteja explicitamente INAPTO.
    df["na_urna"] = situ.eq("APTO") if situ.eq("APTO").any() else ~situ.eq("INAPTO")
    out = (df.sort_values("na_urna", ascending=False)
             .drop_duplicates("candidato")
             .rename(columns={"NR_CANDIDATO": "nr_candidato", "NM_URNA_CANDIDATO": "nome_urna",
                              "NM_PARTIDO": "nome_partido", "SG_FEDERACAO": "federacao",
                              "DS_COMPOSICAO_COLIGACAO": "coligacao",
                              "DS_SITUACAO_CANDIDATURA": "situacao_candidatura",
                              "DS_DETALHE_SITUACAO_CAND": "detalhe_situacao",
                              "DS_SIT_TOT_TURNO": "situacao_turno", "ST_REELEICAO": "reeleicao_tse"})
          )[["ano", "candidato", "nome_urna", "nr_candidato", "partido", "linhagem", "nome_partido",
             "federacao", "coligacao", "situacao_candidatura", "detalhe_situacao", "situacao_turno",
             "reeleicao_tse", "na_urna"]]
    fontes.registrar_uso(f"candidatos_{ano}", [c for c in COLS_CAND if c in df.columns], len(df),
                         "Filtro: cargo Presidente. Na urna = situação APTO.")
    return out.reset_index(drop=True)


def preparar_eleitorado(ano: int = config.ANO_ALVO) -> pd.DataFrame | None:
    """Eleitorado apto por município no ano-alvo (cadastro fechado antes do pleito).

    O arquivo tem uma linha por ESTRATO do perfil (zona, gênero, estado civil,
    faixa etária, escolaridade, raça/cor...) — são 11 milhões de linhas em 2026,
    2,2 GB descompactados. Como só precisamos do total por município, cada bloco
    lido já é somado por município: a soma das somas é idêntica à soma de tudo e
    a memória não cresce com o tamanho do arquivo.
    """
    arq = config.DIR_BRUTOS / f"perfil_eleitorado_{ano}.zip"
    if not arq.exists():
        return None
    lidas = {"n": 0}

    def somar_por_municipio(bloco: pd.DataFrame) -> pd.DataFrame:
        lidas["n"] += len(bloco)
        if "QT_ELEITORES_PERFIL" not in bloco.columns:
            return bloco
        b = bloco.assign(QT_ELEITORES_PERFIL=para_inteiro(bloco["QT_ELEITORES_PERFIL"]))
        return (b.groupby([c for c in ["SG_UF", "CD_MUNICIPIO", "NM_MUNICIPIO"] if c in b.columns],
                          as_index=False, dropna=False)["QT_ELEITORES_PERFIL"].sum())

    df = ler_zip_tse(arq, COLS_ELEITORADO, filtro=somar_por_municipio)
    if df.empty or "QT_ELEITORES_PERFIL" not in df.columns:
        return None
    df = df.assign(uf=df["SG_UF"].str.upper(), cd_municipio=para_inteiro(df["CD_MUNICIPIO"]),
                   eleitores=para_inteiro(df["QT_ELEITORES_PERFIL"]))
    g = df.groupby(["uf", "cd_municipio"], as_index=False).agg(
        nm_municipio=("NM_MUNICIPIO", "first"), eleitores=("eleitores", "sum"))
    fontes.registrar_uso(f"eleitorado_{ano}", COLS_ELEITORADO, lidas["n"],
                         "Soma de QT_ELEITORES (QT_ELEITORES_PERFIL até 2022) de todos os "
                         "estratos do perfil, por município.")
    return g


def preparar_tudo() -> None:
    config.garantir_pastas()
    res, det = [], []
    for ano in config.ANOS_HISTORICOS:
        print(f"Preparando {ano}...")
        try:
            res.append(preparar_votacao(ano))
        except FileNotFoundError:
            print(f"  ausente: votação {ano} (eleição ignorada)")
            continue
        d = preparar_detalhe(ano)
        if d is not None:
            det.append(d)
        else:
            print(f"  aviso: detalhe da apuração {ano} indisponível — comparecimento ficará ausente")
    if not res:
        raise RuntimeError("Nenhum arquivo de votação encontrado em dados/brutos/. Rode o download.")
    resultados = pd.concat(res, ignore_index=True)
    resultados.to_parquet(config.DIR_PROCESSADOS / "resultados_municipio.parquet", index=False)
    if det:
        pd.concat(det, ignore_index=True).to_parquet(
            config.DIR_PROCESSADOS / "comparecimento.parquet", index=False)

    print(f"Preparando candidatos {config.ANO_ALVO}...")
    try:
        preparar_candidatos().to_parquet(config.DIR_PROCESSADOS / f"candidatos_{config.ANO_ALVO}.parquet", index=False)
    except FileNotFoundError:
        print("  ausente: candidatos 2026 — a previsão 2026 não poderá ser feita")
    el = preparar_eleitorado()
    if el is not None:
        el.to_parquet(config.DIR_PROCESSADOS / f"eleitorado_{config.ANO_ALVO}.parquet", index=False)
    else:
        print("  aviso: eleitorado 2026 indisponível — pesos usarão só a eleição anterior")

    from . import pesquisas
    pesquisas.preparar_registro()
    print("Preparação concluída.")
