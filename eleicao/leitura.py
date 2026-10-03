"""Leitura robusta dos arquivos CSV (dentro de .zip) do TSE.

Particularidades tratadas:
* separador ';', codificação latin-1, valores entre aspas;
* o .zip traz um arquivo por UF e, em geral, um arquivo "_BRASIL" com tudo —
  quando ele existe, lemos só ele para não contar votos em dobro;
* arquivos antigos podem não ter linha de cabeçalho (layout do LEIAME);
* o nome da coluna de ano muda entre conjuntos (ANO_ELEICAO na votação,
  AA_ELEICAO no perfil do eleitorado e nas pesquisas), por isso o cabeçalho
  é reconhecido pela forma dos campos, não por um nome fixo;
* nomes de colunas diferentes entre anos (aliases normalizados abaixo);
* marcadores de nulo do TSE (#NULO#, #NE#, -1, -3).
"""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

NULOS_TSE = ["#NULO#", "#NULO", "#NE#", "#NE", "-1", "-3", ""]

# Layouts sem cabeçalho (LEIAME antigo do TSE). Usados só se o arquivo não
# tiver cabeçalho e o número de colunas bater.
LAYOUT_ANTIGO = {
    "votacao": [
        "DATA_GERACAO", "HORA_GERACAO", "ANO_ELEICAO", "NUM_TURNO", "DESCRICAO_ELEICAO",
        "SIGLA_UF", "SIGLA_UE", "CODIGO_MUNICIPIO", "NOME_MUNICIPIO", "NUMERO_ZONA",
        "CODIGO_CARGO", "NUMERO_CAND", "SQ_CANDIDATO", "NOME_CANDIDATO", "NOME_URNA_CANDIDATO",
        "DESCRICAO_CARGO", "COD_SIT_CAND_SUPERIOR", "DESC_SIT_CAND_SUPERIOR",
        "CODIGO_SIT_CANDIDATO", "DESC_SIT_CANDIDATO", "CODIGO_SIT_CAND_TOT", "DESC_SIT_CAND_TOT",
        "NUMERO_PARTIDO", "SIGLA_PARTIDO", "NOME_PARTIDO", "SEQUENCIAL_LEGENDA",
        "NOME_COLIGACAO", "COMPOSICAO_LEGENDA", "TOTAL_VOTOS",
    ],
    "detalhe": [
        "DATA_GERACAO", "HORA_GERACAO", "ANO_ELEICAO", "NUM_TURNO", "DESCRICAO_ELEICAO",
        "SIGLA_UF", "SIGLA_UE", "CODIGO_MUNICIPIO", "NOME_MUNICIPIO", "NUMERO_ZONA",
        "CODIGO_CARGO", "DESCRICAO_CARGO", "QTD_APTOS", "QTD_SECOES", "QTD_SECOES_AGREGADAS",
        "QTD_APTOS_TOT", "QTD_SECOES_TOT", "QTD_COMPARECIMENTO", "QTD_ABSTENCOES",
        "QTD_VOTOS_NOMINAIS", "QTD_VOTOS_BRANCOS", "QTD_VOTOS_NULOS", "QTD_VOTOS_LEGENDA",
        "QTD_VOTOS_ANULADOS_APU_SEP", "DATA_ULT_TOTALIZACAO", "HORA_ULT_TOTALIZACAO",
    ],
}

# Nome antigo -> nome padronizado atual
ALIASES = {
    "NUM_TURNO": "NR_TURNO", "SIGLA_UF": "SG_UF", "CODIGO_MUNICIPIO": "CD_MUNICIPIO",
    "NOME_MUNICIPIO": "NM_MUNICIPIO", "NUMERO_ZONA": "NR_ZONA", "CODIGO_CARGO": "CD_CARGO",
    "DESCRICAO_CARGO": "DS_CARGO", "NUMERO_CAND": "NR_CANDIDATO", "NOME_CANDIDATO": "NM_CANDIDATO",
    "NOME_URNA_CANDIDATO": "NM_URNA_CANDIDATO", "DESC_SIT_CAND_TOT": "DS_SIT_TOT_TURNO",
    "SIGLA_PARTIDO": "SG_PARTIDO", "NUMERO_PARTIDO": "NR_PARTIDO", "NOME_PARTIDO": "NM_PARTIDO",
    "TOTAL_VOTOS": "QT_VOTOS_NOMINAIS", "QTD_APTOS": "QT_APTOS",
    "QTD_COMPARECIMENTO": "QT_COMPARECIMENTO", "QTD_ABSTENCOES": "QT_ABSTENCOES",
    "DESC_SIT_CANDIDATO": "DS_SITUACAO_CANDIDATURA", "DESCRICAO_ELEICAO": "DS_ELEICAO",
    # Perfil do eleitorado e pesquisas usam AA_ELEICAO; a partir de 2026 a
    # contagem de eleitores do perfil passou a se chamar QT_ELEITORES.
    "AA_ELEICAO": "ANO_ELEICAO", "QT_ELEITORES": "QT_ELEITORES_PERFIL",
    "QT_ELEITORES_PERFIL": "QT_ELEITORES_PERFIL",
}

# Cabeçalho do TSE: todo campo é um identificador em maiúsculas.
_RE_NOME_COLUNA = re.compile(r"[A-Z][A-Z0-9_]*")


def membros_csv(zf: zipfile.ZipFile) -> list[str]:
    nomes = [n for n in zf.namelist() if n.lower().endswith((".csv", ".txt"))
             and "leiame" not in n.lower()]
    brasil = [n for n in nomes if n.upper().rsplit(".", 1)[0].endswith("_BRASIL")]
    return brasil[:1] if brasil else sorted(nomes)


def _primeira_linha(zf: zipfile.ZipFile, membro: str) -> str:
    with zf.open(membro) as f:
        return f.readline().decode("latin-1")


def _campos(linha: str) -> list[str]:
    """Campos da primeira linha, sem aspas nem espacos, em maiusculas."""
    # O strip de cada campo ja remove o fim de linha do ultimo deles.
    return [c.strip().strip(chr(34)).strip().upper() for c in linha.split(";")]


def tem_cabecalho(linha: str) -> bool:
    """A primeira linha do arquivo é o cabeçalho?

    Não serve procurar um nome fixo: os arquivos de votação e de detalhe usam
    ANO_ELEICAO, mas o perfil do eleitorado e as pesquisas usam AA_ELEICAO. O
    que distingue um cabeçalho do TSE é que TODOS os campos são identificadores
    em maiúsculas (A-Z, 0-9, _). Uma linha de dados começa por data
    ("20/09/2023"), traz acentos ou texto em minúsculas e não passa no teste.
    """
    campos = _campos(linha)
    return len(campos) > 1 and all(_RE_NOME_COLUNA.fullmatch(c) for c in campos)


def ler_zip_tse(caminho: Path, colunas: Iterable[str], tipo_layout: str | None = None,
                filtro: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
                tamanho_bloco: int = 400_000) -> pd.DataFrame:
    """Lê as colunas pedidas (nomes padronizados) de todos os CSVs relevantes do zip.

    Colunas pedidas que não existirem no arquivo são simplesmente omitidas;
    quem chama decide se a ausência é aceitável.
    """
    desejadas = set(colunas)
    partes: list[pd.DataFrame] = []
    with zipfile.ZipFile(caminho) as zf:
        membros = membros_csv(zf)
        for membro in membros:
            linha = _primeira_linha(zf, membro)
            if tem_cabecalho(linha):
                cab = _campos(linha)
                nomes = None
                header = 0
            else:
                n = len(linha.rstrip("\r\n").split(";"))
                layout = LAYOUT_ANTIGO.get(tipo_layout or "")
                if not layout or n < len(layout):
                    raise ValueError(
                        f"{caminho.name}/{membro}: arquivo sem cabeçalho e layout desconhecido "
                        f"({n} colunas). Consulte o LEIAME do TSE e ajuste LAYOUT_ANTIGO.")
                nomes = layout + [f"EXTRA_{i}" for i in range(n - len(layout))]
                cab = nomes
                header = None
            padronizado = [ALIASES.get(c, c) for c in cab]
            usar_idx = [i for i, c in enumerate(padronizado) if c in desejadas]
            if not usar_idx:
                continue
            with zf.open(membro) as f:
                texto = io.TextIOWrapper(f, encoding="latin-1", newline="")
                leitor = pd.read_csv(texto, sep=";", header=header, names=nomes, dtype=str,
                                     usecols=usar_idx, keep_default_na=False,
                                     chunksize=tamanho_bloco, quotechar='"', on_bad_lines="warn")
                for bloco in leitor:
                    bloco.columns = [padronizado[i] for i in sorted(usar_idx)]
                    bloco = bloco.apply(lambda s: s.str.strip())
                    if filtro is not None:
                        bloco = filtro(bloco)
                    if len(bloco):
                        partes.append(bloco)
    if not partes:
        return pd.DataFrame(columns=sorted(desejadas))
    df = pd.concat(partes, ignore_index=True)
    if len(membros) > 1:
        # Só quando lemos vários membros do mesmo zip (quando não existe o
        # arquivo _BRASIL) a mesma linha pode aparecer repetida em dois deles.
        # Dentro de UM arquivo, duas linhas iguais nas colunas lidas são
        # registros distintos — é o caso do perfil do eleitorado, em que cada
        # linha é um estrato (zona, gênero, faixa etária...). Descartá-las ali
        # subestimaria o eleitorado do município.
        df = df.drop_duplicates(ignore_index=True)
    return df


def para_inteiro(s: pd.Series) -> pd.Series:
    s = s.replace(NULOS_TSE, pd.NA)
    return pd.to_numeric(s, errors="coerce").fillna(0).astype("int64")


def filtro_presidente(bloco: pd.DataFrame) -> pd.DataFrame:
    if "CD_CARGO" in bloco.columns:
        return bloco[bloco["CD_CARGO"] == "1"]
    if "DS_CARGO" in bloco.columns:
        return bloco[bloco["DS_CARGO"].str.upper().str.contains("PRESIDENTE")
                     & ~bloco["DS_CARGO"].str.upper().str.contains("VICE")]
    return bloco
