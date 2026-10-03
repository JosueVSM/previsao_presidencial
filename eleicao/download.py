"""Download dos arquivos oficiais do TSE (Portal de Dados Abertos / CDN).

Uso:  python -m eleicao.pipeline baixar
Os arquivos ficam em dados/brutos/ e cada download é registrado em
dados/fontes.json (endereço, data de acesso, hash).
"""
from __future__ import annotations

import time
import urllib.request
from pathlib import Path

from . import config, fontes


def _baixar(url: str, destino: Path, tentativas: int = 3, forcar: bool = False) -> str:
    """Retorna 'novo', 'existente' ou '' (falha)."""
    if destino.exists() and destino.stat().st_size > 0 and not forcar:
        print(f"  já existe: {destino.name}")
        return "existente"
    destino.parent.mkdir(parents=True, exist_ok=True)
    tmp = destino.with_suffix(destino.suffix + ".parcial")
    for t in range(1, tentativas + 1):
        try:
            print(f"  baixando {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "previsao-presidencial/1.0"})
            with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as f:
                total = int(resp.headers.get("Content-Length") or 0)
                lido = 0
                while True:
                    bloco = resp.read(1 << 20)
                    if not bloco:
                        break
                    f.write(bloco)
                    lido += len(bloco)
                    if total:
                        print(f"\r    {lido/1e6:7.1f} / {total/1e6:7.1f} MB", end="")
            print()
            tmp.replace(destino)
            return "novo"
        except Exception as e:  # noqa: BLE001 — queremos relatar qualquer falha
            print(f"\n    falha ({t}/{tentativas}): {e}")
            time.sleep(3 * t)
    if tmp.exists():
        tmp.unlink()
    return ""


def tarefas() -> list[dict]:
    """Lista de todos os arquivos necessários, com metadados para o catálogo."""
    pag = config.PAGINAS_DATASET
    lst: list[dict] = []
    for ano in config.ANOS_HISTORICOS:
        lst.append(dict(chave=f"votacao_{ano}", url=config.url_votacao(ano),
                        pagina=pag["resultados"].format(ano=ano),
                        arquivo=config.DIR_BRUTOS / f"votacao_candidato_munzona_{ano}.zip",
                        periodo=f"Eleições gerais {ano} (1º e 2º turnos)",
                        descricao="Votação nominal por candidato, município e zona",
                        obrigatorio=True))
        lst.append(dict(chave=f"detalhe_{ano}", url=config.url_detalhe(ano),
                        pagina=pag["resultados"].format(ano=ano),
                        arquivo=config.DIR_BRUTOS / f"detalhe_votacao_munzona_{ano}.zip",
                        periodo=f"Eleições gerais {ano} (1º e 2º turnos)",
                        descricao="Detalhe da apuração por município e zona (aptos, comparecimento)",
                        obrigatorio=False))
    a = config.ANO_ALVO
    lst.append(dict(chave=f"candidatos_{a}", url=config.url_candidatos(a),
                    pagina=pag["candidatos"].format(ano=a),
                    arquivo=config.DIR_BRUTOS / f"consulta_cand_{a}.zip",
                    periodo=f"Candidaturas registradas para as eleições {a}",
                    descricao="Candidatos e partidos (situação da candidatura)",
                    obrigatorio=True))
    lst.append(dict(chave=f"eleitorado_{a}", url=config.url_eleitorado(a),
                    pagina=pag["eleitorado"].format(ano=a),
                    arquivo=config.DIR_BRUTOS / f"perfil_eleitorado_{a}.zip",
                    periodo=f"Cadastro eleitoral para as eleições {a}",
                    descricao="Perfil do eleitorado por município",
                    obrigatorio=False))
    lst.append(dict(chave=f"pesquisas_{a}", url=config.url_pesquisas(a),
                    pagina=pag["pesquisas"].format(ano=a),
                    arquivo=config.DIR_BRUTOS / f"pesquisa_eleitoral_{a}.zip",
                    periodo=f"Pesquisas registradas para as eleições {a}",
                    descricao="Registro de pesquisas eleitorais (metadados; sem percentuais)",
                    obrigatorio=False))
    return lst


def baixar_tudo(forcar: bool = False) -> None:
    config.garantir_pastas()
    faltando = []
    for t in tarefas():
        status = _baixar(t["url"], t["arquivo"], forcar=forcar)
        if not status:  # só entra aqui quando o download realmente falhou
            faltando.append((t["chave"], t["obrigatorio"]))
            continue
        ja_registrado = t["chave"] in fontes.catalogo()
        # A data de acesso só é (re)gravada quando o arquivo foi de fato baixado
        # agora, ou quando um arquivo colocado manualmente ainda não tem registro.
        if status == "novo" or not ja_registrado:
            fontes.registrar_download(t["chave"], fonte="Tribunal Superior Eleitoral (TSE) — Portal de Dados Abertos",
                                      url=t["url"], pagina=t["pagina"], arquivo=t["arquivo"],
                                      periodo=t["periodo"], descricao=t["descricao"])
    if faltando:
        print("\nArquivos não baixados:")
        for chave, obrig in faltando:
            print(f"  - {chave} ({'OBRIGATÓRIO' if obrig else 'opcional'})")
        print("Você pode baixá-los manualmente e colocá-los em dados/brutos/ com o mesmo nome.")
