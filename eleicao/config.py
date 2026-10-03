"""Configurações centrais do projeto.

Tudo que um analista pode querer ajustar (anos, URLs, hiperparâmetros,
datas das eleições) fica aqui, para que as escolhas fiquem documentadas
em um único lugar.
"""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
RAIZ = Path(os.environ.get("ELEICAO_RAIZ", Path(__file__).resolve().parent.parent))
DIR_DADOS = RAIZ / "dados"
DIR_BRUTOS = DIR_DADOS / "brutos"
DIR_PROCESSADOS = DIR_DADOS / "processados"
DIR_MANUAL = DIR_DADOS / "manual"
DIR_MODELOS = RAIZ / "modelos"
DIR_RESULTADOS = RAIZ / "resultados"
ARQ_FONTES = DIR_DADOS / "fontes.json"
# Amostra versionada no repositório: permite abrir o app logo após clonar, sem
# baixar os 2,3 GB do TSE. Gerada por `python -m eleicao.pipeline amostra`.
DIR_AMOSTRA = RAIZ / "amostra"
ARQ_LINHAGEM = Path(__file__).resolve().parent.parent / "config" / "linhagem_partidos.csv"
ARQ_PESQUISAS_MANUAIS = DIR_MANUAL / "pesquisas_resultados.csv"


def garantir_pastas() -> None:
    for p in (DIR_BRUTOS, DIR_PROCESSADOS, DIR_MANUAL, DIR_MODELOS, DIR_RESULTADOS):
        p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Eleições
# ---------------------------------------------------------------------------
# Eleições presidenciais com resultados oficiais usados no histórico.
# 1998 só serve como "eleição anterior" (defasagem) para 2002.
ANOS_HISTORICOS = [1998, 2002, 2006, 2010, 2014, 2018, 2022]
ANO_ALVO = 2026
INTERVALO = 4  # anos entre eleições presidenciais

# Primeira eleição que pode ser alvo de treino (precisa de uma anterior).
PRIMEIRO_ANO_ALVO = 2002

# Eleições usadas como teste na validação temporal. Cada uma é prevista por
# um modelo treinado apenas com eleições anteriores a ela.
ANOS_TESTE_BACKTEST = [2010, 2014, 2018, 2022]

CD_CARGO_PRESIDENTE = 1

# Datas oficiais (domingos) das eleições — usadas para filtrar pesquisas
# divulgadas antes do pleito. Fonte: calendários eleitorais do TSE.
DATAS_ELEICAO = {
    (2002, 1): date(2002, 10, 6), (2002, 2): date(2002, 10, 27),
    (2006, 1): date(2006, 10, 1), (2006, 2): date(2006, 10, 29),
    (2010, 1): date(2010, 10, 3), (2010, 2): date(2010, 10, 31),
    (2014, 1): date(2014, 10, 5), (2014, 2): date(2014, 10, 26),
    (2018, 1): date(2018, 10, 7), (2018, 2): date(2018, 10, 28),
    (2022, 1): date(2022, 10, 2), (2022, 2): date(2022, 10, 30),
    (2026, 1): date(2026, 10, 4), (2026, 2): date(2026, 10, 25),
}

# ---------------------------------------------------------------------------
# Fontes (Portal de Dados Abertos do TSE — CDN oficial)
# ---------------------------------------------------------------------------
CDN = "https://cdn.tse.jus.br/estatistica/sead/odsele"


def url_votacao(ano: int) -> str:
    return f"{CDN}/votacao_candidato_munzona/votacao_candidato_munzona_{ano}.zip"


def url_detalhe(ano: int) -> str:
    return f"{CDN}/detalhe_votacao_munzona/detalhe_votacao_munzona_{ano}.zip"


def url_candidatos(ano: int) -> str:
    return f"{CDN}/consulta_cand/consulta_cand_{ano}.zip"


def url_eleitorado(ano: int) -> str:
    return f"{CDN}/perfil_eleitorado/perfil_eleitorado_{ano}.zip"


def url_pesquisas(ano: int) -> str:
    return f"{CDN}/pesquisa_eleitoral/pesquisa_eleitoral_{ano}.zip"


PAGINAS_DATASET = {
    "resultados": "https://dadosabertos.tse.jus.br/dataset/resultados-{ano}",
    "candidatos": "https://dadosabertos.tse.jus.br/dataset/candidatos-{ano}",
    "eleitorado": "https://dadosabertos.tse.jus.br/dataset/eleitorado-{ano}",
    "pesquisas": "https://dadosabertos.tse.jus.br/dataset/pesquisas-eleitorais-{ano}",
}

# ---------------------------------------------------------------------------
# Rede neural (MLP) — hiperparâmetros documentados
# ---------------------------------------------------------------------------
HIPERPARAMETROS = {
    "camadas_ocultas": [64, 32],
    "dropout": 0.10,
    "taxa_aprendizado": 1e-3,
    "decaimento_pesos": 1e-3,
    "tamanho_lote": 256,          # nº de disputas (municípios-turno) por lote
    "max_epocas": 200,
    "paciencia": 15,              # parada antecipada: épocas sem melhora na validação
    "n_validacoes_internas": 3,   # últimas eleições de treino usadas, uma a uma, como validação
    "sementes_validacao": 2,      # redes por eleição de validação (para escolher as épocas)
    "epocas_sem_validacao": 50,   # usado só se houver uma única eleição de treino
    "sementes": [11, 23, 37, 51, 73],  # conjunto (ensemble) de 5 redes
    "peso_disputa": "raiz_votos_validos",
}

# Janela de pesquisas (opcional): só pesquisas com campo encerrado entre
# 30 dias e 1 dia antes da eleição.
JANELA_PESQUISAS_DIAS = 30

REGIOES = {
    "AC": "N", "AM": "N", "AP": "N", "PA": "N", "RO": "N", "RR": "N", "TO": "N",
    "AL": "NE", "BA": "NE", "CE": "NE", "MA": "NE", "PB": "NE", "PE": "NE",
    "PI": "NE", "RN": "NE", "SE": "NE",
    "DF": "CO", "GO": "CO", "MS": "CO", "MT": "CO",
    "ES": "SE", "MG": "SE", "RJ": "SE", "SP": "SE",
    "PR": "S", "RS": "S", "SC": "S",
}
LISTA_REGIOES = ["N", "NE", "CO", "SE", "S", "EXT"]
NOMES_UF = {
    "AC": "Acre", "AL": "Alagoas", "AM": "Amazonas", "AP": "Amapá", "BA": "Bahia",
    "CE": "Ceará", "DF": "Distrito Federal", "ES": "Espírito Santo", "GO": "Goiás",
    "MA": "Maranhão", "MG": "Minas Gerais", "MS": "Mato Grosso do Sul",
    "MT": "Mato Grosso", "PA": "Pará", "PB": "Paraíba", "PE": "Pernambuco",
    "PI": "Piauí", "PR": "Paraná", "RJ": "Rio de Janeiro", "RN": "Rio Grande do Norte",
    "RO": "Rondônia", "RR": "Roraima", "RS": "Rio Grande do Sul", "SC": "Santa Catarina",
    "SE": "Sergipe", "SP": "São Paulo", "TO": "Tocantins", "ZZ": "Exterior",
}


def regiao(uf: str) -> str:
    return REGIOES.get(uf, "EXT")
