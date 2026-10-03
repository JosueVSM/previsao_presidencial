"""Configuração dos testes: cria uma pasta temporária com dados SINTÉTICOS no
formato do TSE e aponta o projeto para ela (variável ELEICAO_RAIZ), antes de
qualquer import do pacote. Nenhum dado real é usado nem gravado."""
import os
import sys
import tempfile
from pathlib import Path

RAIZ_TESTE = Path(tempfile.mkdtemp(prefix="eleicao_teste_"))
os.environ["ELEICAO_RAIZ"] = str(RAIZ_TESTE)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest  # noqa: E402

from eleicao import config  # noqa: E402

# Treino mais curto nos testes (2 redes, até 40 épocas).
config.HIPERPARAMETROS.update({"sementes": [11, 23], "max_epocas": 40, "paciencia": 8,
                               "sementes_validacao": 1})


@pytest.fixture(scope="session")
def dados_sinteticos():
    import sintetico
    return sintetico.escrever(RAIZ_TESTE, n_por_uf=20)


@pytest.fixture(scope="session")
def pipeline_completo(dados_sinteticos):
    from eleicao.avaliacao import backtest, prever_alvo
    from eleicao.preparacao import preparar_tudo
    from eleicao.verificacao import verificar
    preparar_tudo()
    met = backtest(verbose=False)
    prev = prever_alvo(verbose=False)
    itens = verificar()
    return {"metricas": met, "previsao": prev, "verificacoes": itens}
