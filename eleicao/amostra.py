"""Amostra versionada no repositório, para o app abrir logo após o clone.

O pipeline completo baixa 2,3 GB do TSE e leva perto de 40 minutos. Quem clona o
repositório para só *ver* o resultado não deve precisar disso — sem nada em
`dados/processados/`, o app abriria na tela de "nenhum dado processado".

Esta etapa copia para `amostra/` um recorte pequeno do que o pipeline já produziu:

* **completos**, porque são pequenos e são a prova que interessa — a tabela de
  métricas dos quatro testes retrospectivos, o agregado nacional estimado contra o
  oficial, o relatório das verificações, os metadados do modelo e o catálogo de fontes;
* **completos** também o histórico oficial por município e a estimativa de 2026;
* **recortado** apenas o arquivo grande de previsões municipais do backtest, que fica
  só com a eleição de `ANO_AMOSTRA` (os dois turnos, todos os municípios). Um teste
  retrospectivo inteiro e honesto, em vez de quatro pela metade.

Nada aqui é recalculado nem arredondado: são os mesmos números do pipeline.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime

import pandas as pd

from . import config

# Eleição mantida por inteiro no backtest da amostra.
ANO_AMOSTRA = 2022

# Copiados sem alteração (origem relativa a DIR_RESULTADOS, salvo indicação).
ARQUIVOS_INTEIROS = [
    "avaliacao_metricas.csv",
    "backtest_nacional.csv",
    "relatorio_verificacoes.json",
    f"previsao_{config.ANO_ALVO}.parquet",
    f"previsao_{config.ANO_ALVO}_meta.json",
]


def gerar() -> dict:
    """Monta `amostra/` a partir dos resultados do pipeline. Devolve o que foi gravado."""
    destino = config.DIR_AMOSTRA
    destino.mkdir(parents=True, exist_ok=True)
    gravados: dict[str, int] = {}

    def registrar(nome: str) -> None:
        gravados[nome] = (destino / nome).stat().st_size

    # --- histórico oficial e cadastros (o app não abre sem eles) --------------
    for nome in ["resultados_municipio.parquet",
                 f"candidatos_{config.ANO_ALVO}.parquet",
                 f"pesquisas_registro_{config.ANO_ALVO}.parquet"]:
        origem = config.DIR_PROCESSADOS / nome
        if origem.exists():
            shutil.copyfile(origem, destino / nome)
            registrar(nome)

    # --- resultados do modelo -------------------------------------------------
    for nome in ARQUIVOS_INTEIROS:
        origem = config.DIR_RESULTADOS / nome
        if origem.exists():
            shutil.copyfile(origem, destino / nome)
            registrar(nome)

    # --- backtest recortado a uma eleição ------------------------------------
    origem = config.DIR_RESULTADOS / "previsoes_backtest.parquet"
    if origem.exists():
        b = pd.read_parquet(origem)
        recorte = b[b.ano == ANO_AMOSTRA]
        if recorte.empty:  # eleição ausente: guarda a mais recente que existir
            recorte = b[b.ano == b.ano.max()]
        recorte.to_parquet(destino / "previsoes_backtest.parquet", index=False)
        registrar("previsoes_backtest.parquet")

    # --- metadados do modelo e catálogo de fontes ----------------------------
    meta = config.DIR_MODELOS / f"mlp_{config.ANO_ALVO}" / "modelo.json"
    if meta.exists():
        shutil.copyfile(meta, destino / "modelo.json")
        registrar("modelo.json")
    if config.ARQ_FONTES.exists():
        shutil.copyfile(config.ARQ_FONTES, destino / "fontes.json")
        registrar("fontes.json")

    # --- procedência ----------------------------------------------------------
    anos = sorted(int(a) for a in pd.read_parquet(destino / "previsoes_backtest.parquet").ano.unique()) \
        if (destino / "previsoes_backtest.parquet").exists() else []
    (destino / "AMOSTRA.json").write_text(json.dumps({
        "gerado_em": datetime.now().astimezone().isoformat(timespec="seconds"),
        "o_que_e": "Recorte dos resultados do pipeline, versionado para o app abrir sem baixar os dados do TSE.",
        "backtest_contem_eleicoes": anos,
        "metricas_contem": "os quatro testes retrospectivos (2010, 2014, 2018, 2022), completos",
        "observacao": "Números idênticos aos do pipeline; nada foi recalculado. "
                      "Para o conjunto completo rode: python -m eleicao.pipeline tudo",
        "arquivos": gravados,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    total = sum(gravados.values()) / 1e6
    print(f"  amostra gravada em {destino.name}/ — {len(gravados)} arquivos, {total:.1f} MB")
    for nome, tam in sorted(gravados.items()):
        print(f"    {nome:42s} {tam/1e6:6.2f} MB")
    return gravados


def disponivel() -> bool:
    return (config.DIR_AMOSTRA / "resultados_municipio.parquet").exists()
