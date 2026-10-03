"""Linha de comando do projeto.

    python -m eleicao.pipeline baixar      # baixa os arquivos oficiais do TSE
    python -m eleicao.pipeline preparar    # gera tabelas limpas em dados/processados
    python -m eleicao.pipeline avaliar     # validação temporal (backtests)
    python -m eleicao.pipeline prever      # treina a rede final e estima 2026
    python -m eleicao.pipeline verificar   # checa escala, consistência e vazamento
    python -m eleicao.pipeline amostra     # recorte pequeno versionado no repositório
    python -m eleicao.pipeline tudo        # todas as etapas acima, em ordem
"""
from __future__ import annotations

import argparse
import sys
import time

from . import config


def _saida_utf8() -> None:
    """Imprime em UTF-8 mesmo no console do Windows (padrão cp1252).

    Sem isso, qualquer caractere fora do cp1252 derruba o pipeline inteiro no
    meio de um print — foi o que acontecia ao listar as verificações.
    """
    for fluxo in (sys.stdout, sys.stderr):
        try:
            fluxo.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass  # saída já capturada (pytest) ou sem suporte a reconfigure


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Estimativa experimental de votos para Presidente (rede neural)")
    ap.add_argument("etapa",
                    choices=["baixar", "preparar", "avaliar", "prever", "verificar", "amostra", "tudo"])
    ap.add_argument("--forcar-download", action="store_true", help="baixa de novo mesmo se o arquivo existir")
    args = ap.parse_args(argv)
    _saida_utf8()
    config.garantir_pastas()
    t0 = time.time()
    etapas = (["baixar", "preparar", "avaliar", "prever", "verificar", "amostra"]
              if args.etapa == "tudo" else [args.etapa])
    for e in etapas:
        print(f"\n=== {e.upper()} ===")
        if e == "baixar":
            from .download import baixar_tudo
            baixar_tudo(forcar=args.forcar_download)
        elif e == "preparar":
            from .preparacao import preparar_tudo
            preparar_tudo()
        elif e == "avaliar":
            from .avaliacao import backtest
            met = backtest()
            print(met[met.modelo.str.startswith("Rede")][["eleicao_teste", "turno", "mae_pp", "rmse_pp",
                                                           "erro_nacional_mae_pp"]].to_string(index=False))
        elif e == "prever":
            from .avaliacao import prever_alvo
            prever_alvo()
        elif e == "verificar":
            from .verificacao import verificar
            verificar()
        elif e == "amostra":
            from .amostra import gerar
            gerar()
    print(f"\nConcluído em {time.time() - t0:.0f} s.")


if __name__ == "__main__":
    main()
