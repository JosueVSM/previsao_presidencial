"""Registro das fontes de dados (catálogo em dados/fontes.json).

Para cada conjunto baixado guardamos: fonte, endereço, data de acesso,
período coberto, colunas utilizadas, tamanho e hash do arquivo. O app lê
esse catálogo para exibir as fontes ao usuário.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

from . import config


def _carregar() -> dict:
    if config.ARQ_FONTES.exists():
        return json.loads(config.ARQ_FONTES.read_text(encoding="utf-8"))
    return {}


def _salvar(cat: dict) -> None:
    config.ARQ_FONTES.parent.mkdir(parents=True, exist_ok=True)
    config.ARQ_FONTES.write_text(json.dumps(cat, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(caminho: Path) -> str:
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def registrar_download(chave: str, *, fonte: str, url: str, pagina: str, arquivo: Path,
                       periodo: str, descricao: str) -> None:
    cat = _carregar()
    item = cat.get(chave, {})
    item.update({
        "fonte": fonte,
        "descricao": descricao,
        "url": url,
        "pagina_dataset": pagina,
        "arquivo_local": str(arquivo.relative_to(config.RAIZ)) if arquivo.is_relative_to(config.RAIZ) else str(arquivo),
        "data_acesso": datetime.now().astimezone().isoformat(timespec="seconds"),
        "periodo_coberto": periodo,
        "tamanho_bytes": arquivo.stat().st_size,
        "sha256": sha256(arquivo),
    })
    cat[chave] = item
    _salvar(cat)


def registrar_uso(chave: str, colunas: list[str], linhas: int, observacao: str = "") -> None:
    """Anota quais colunas do conjunto foram efetivamente usadas."""
    cat = _carregar()
    item = cat.setdefault(chave, {})
    item["colunas_utilizadas"] = sorted(set(colunas))
    item["linhas_utilizadas"] = int(linhas)
    if observacao:
        item["observacao"] = observacao
    _salvar(cat)


def catalogo() -> dict:
    return _carregar()
