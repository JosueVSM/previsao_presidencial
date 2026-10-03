"""Rede neural (perceptron multicamadas) com softmax por disputa.

Arquitetura
-----------
Para cada linha (disputa d, candidato c) com vetor de entrada x_dc, a MLP
calcula uma pontuação escalar

    s_dc = MLP(x_dc)     (Linear → ReLU → Dropout → Linear → ReLU → Dropout → Linear)

com os MESMOS pesos para todos os candidatos. As porcentagens previstas são
obtidas com um softmax restrito aos candidatos daquela disputa:

    p_dc = exp(s_dc) / Σ_k∈d exp(s_dk)

Assim toda previsão fica entre 0 e 1 e soma exatamente 1 (100% dos votos
válidos) em cada município, para qualquer número de candidatos.
Disputas com menos candidatos recebem preenchimento mascarado (−∞ antes
do softmax).

Função de perda
---------------
Entropia cruzada entre a distribuição observada y_d e a prevista p_d,
ponderada por disputa:

    perda = − Σ_d w_d Σ_c y_dc · log p_dc / Σ_d w_d,   w_d = √(votos válidos_d)

Com w_d = votos válidos a perda seria a verossimilhança multinomial exata dos
votos, mas as capitais dominariam o treino; √ é um meio-termo documentado.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from . import config

torch.set_num_threads(max(1, torch.get_num_threads()))


# ---------------------------------------------------------------------------
# Pré-processamento
# ---------------------------------------------------------------------------
@dataclass
class Padronizador:
    """Padronização z-score ajustada SOMENTE nos dados de treino."""
    media: np.ndarray | None = None
    desvio: np.ndarray | None = None

    def ajustar(self, X: np.ndarray) -> "Padronizador":
        self.media = X.mean(axis=0)
        d = X.std(axis=0)
        self.desvio = np.where(d < 1e-8, 1.0, d)
        return self

    def transformar(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.media) / self.desvio).astype(np.float32)


@dataclass
class Lote:
    X: torch.Tensor      # [D, Cmax, F]
    mascara: torch.Tensor  # [D, Cmax] bool
    Y: torch.Tensor      # [D, Cmax]
    w: torch.Tensor      # [D]


def tensores(df: pd.DataFrame, colunas: list[str], pad: Padronizador) -> tuple[Lote, np.ndarray, np.ndarray]:
    """Converte a tabela longa em tensores com preenchimento por disputa.
    Retorna o lote e os índices (disputa, posição) de cada linha original."""
    cod, _ = pd.factorize(df["id_disputa"], sort=False)
    pos = df.groupby(cod).cumcount().values
    D, C = cod.max() + 1, pos.max() + 1
    F = len(colunas)
    Xr = pad.transformar(df[colunas].to_numpy(dtype=np.float64))
    X = np.zeros((D, C, F), dtype=np.float32)
    M = np.zeros((D, C), dtype=bool)
    Y = np.zeros((D, C), dtype=np.float32)
    X[cod, pos] = Xr
    M[cod, pos] = True
    if "y" in df and df["y"].notna().all():
        Y[cod, pos] = df["y"].to_numpy(dtype=np.float32)
    vv = df.groupby(cod)["votos_validos_disputa"].first().fillna(1.0).to_numpy()
    w = np.sqrt(np.maximum(vv, 1.0))
    w = w / w.mean()
    lote = Lote(torch.from_numpy(X), torch.from_numpy(M), torch.from_numpy(Y),
                torch.from_numpy(w.astype(np.float32)))
    return lote, cod, pos


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------
class MLPDisputa(nn.Module):
    def __init__(self, n_entradas: int, ocultas: list[int], dropout: float):
        super().__init__()
        camadas: list[nn.Module] = []
        ant = n_entradas
        for h in ocultas:
            camadas += [nn.Linear(ant, h), nn.ReLU(), nn.Dropout(dropout)]
            ant = h
        camadas.append(nn.Linear(ant, 1))
        self.rede = nn.Sequential(*camadas)

    def forward(self, X: torch.Tensor, mascara: torch.Tensor) -> torch.Tensor:
        s = self.rede(X).squeeze(-1)                       # [D, C]
        s = s.masked_fill(~mascara, float("-inf"))
        return torch.log_softmax(s, dim=1)                 # log p, soma 1 por disputa


def perda(logp: torch.Tensor, Y: torch.Tensor, mascara: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    lp = logp.masked_fill(~mascara, 0.0)
    ce = -(Y * lp).sum(dim=1)
    return (w * ce).sum() / w.sum()


def _treinar_uma(lote: Lote, semente: int, hp: dict, val: Lote | None, epocas_fixas: int | None):
    torch.manual_seed(semente)
    np.random.seed(semente)
    modelo = MLPDisputa(lote.X.shape[-1], hp["camadas_ocultas"], hp["dropout"])
    opt = torch.optim.AdamW(modelo.parameters(), lr=hp["taxa_aprendizado"], weight_decay=hp["decaimento_pesos"])
    D = lote.X.shape[0]
    gerador = torch.Generator().manual_seed(semente)
    max_ep = epocas_fixas if epocas_fixas is not None else hp["max_epocas"]
    melhor, melhor_ep, sem_melhora = float("inf"), 0, 0
    melhor_estado = None
    historico = []
    for ep in range(1, max_ep + 1):
        modelo.train()
        ordem = torch.randperm(D, generator=gerador)
        soma, n = 0.0, 0
        for i in range(0, D, hp["tamanho_lote"]):
            idx = ordem[i:i + hp["tamanho_lote"]]
            opt.zero_grad()
            l = perda(modelo(lote.X[idx], lote.mascara[idx]), lote.Y[idx], lote.mascara[idx], lote.w[idx])
            l.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), 5.0)
            opt.step()
            soma += l.item() * len(idx)
            n += len(idx)
        reg = {"epoca": ep, "perda_treino": soma / n}
        if val is not None:
            modelo.eval()
            with torch.no_grad():
                lv = perda(modelo(val.X, val.mascara), val.Y, val.mascara, val.w).item()
            reg["perda_validacao"] = lv
            if lv < melhor - 1e-5:
                melhor, melhor_ep, sem_melhora = lv, ep, 0
                melhor_estado = copy.deepcopy(modelo.state_dict())
            else:
                sem_melhora += 1
                if sem_melhora >= hp["paciencia"]:
                    historico.append(reg)
                    break
        historico.append(reg)
    if val is not None and melhor_estado is not None:
        modelo.load_state_dict(melhor_estado)
    else:
        melhor_ep = max_ep
    modelo.eval()
    return modelo, melhor_ep, historico


@dataclass
class ConjuntoMLP:
    """Conjunto (ensemble) de MLPs treinadas com sementes diferentes.

    Escolha do número de épocas (parada antecipada) com validação temporal
    interna e expansiva: para cada uma das últimas eleições de treino v,
    treina-se com as eleições anteriores a v e mede-se a perda em v (eleição
    inteira fora do treino). As curvas de validação são promediadas e o
    número de épocas com menor perda média é escolhido. Depois, cada rede do
    conjunto é re-treinada do zero com TODAS as eleições de treino por esse
    número de épocas. Usar várias eleições de validação em vez de uma só
    reduz a dependência de uma única eleição atípica.

    A média das probabilidades das redes é a estimativa; o desvio-padrão
    entre redes mede só a instabilidade do treino (NÃO é intervalo de confiança).
    """
    colunas: list[str]
    hp: dict = field(default_factory=lambda: dict(config.HIPERPARAMETROS))
    padronizador: Padronizador | None = None
    modelos: list[MLPDisputa] = field(default_factory=list)
    epocas: list[int] = field(default_factory=list)
    curvas_validacao: dict = field(default_factory=dict)
    eleicoes_treino: list[str] = field(default_factory=list)
    eleicoes_validacao: list[str] = field(default_factory=list)

    def _escolher_epocas(self, df: pd.DataFrame, anos: list[int], verbose: bool) -> int:
        vals = anos[1:][-self.hp.get("n_validacoes_internas", 3):]
        if not vals:
            return self.hp.get("epocas_sem_validacao", 50)
        self.eleicoes_validacao = [str(v) for v in vals]
        max_ep = self.hp["max_epocas"]
        curvas = []
        for v in vals:
            dtr, dva = df[df.ano < v], df[df.ano == v]
            # Padronização ajustada só no treino interno.
            pad = Padronizador().ajustar(dtr[self.colunas].to_numpy(dtype=np.float64))
            ti, _, _ = tensores(dtr, self.colunas, pad)
            tv, _, _ = tensores(dva, self.colunas, pad)
            for sem in self.hp["sementes"][: self.hp.get("sementes_validacao", 2)]:
                _, _, hist = _treinar_uma(ti, sem, self.hp, tv, None)
                c = [h["perda_validacao"] for h in hist]
                c = c + [c[-1]] * (max_ep - len(c))  # após a parada, repete o último valor
                curvas.append(c)
                self.curvas_validacao.setdefault(str(v), []).append(c[: len(hist)])
        media = np.mean(np.array(curvas), axis=0)
        ep = int(np.argmin(media)) + 1
        self.curvas_validacao["media"] = media.tolist()
        if verbose:
            print(f"    épocas escolhidas: {ep} (validação interna em {', '.join(self.eleicoes_validacao)})")
        return ep

    def treinar(self, df_treino: pd.DataFrame, verbose: bool = True) -> "ConjuntoMLP":
        if df_treino["y"].isna().any():
            raise ValueError("Dados de treino contêm alvo ausente.")
        anos = sorted(int(a) for a in df_treino["ano"].unique())
        self.eleicoes_treino = [str(a) for a in anos]
        ep = self._escolher_epocas(df_treino, anos, verbose)
        self.padronizador = Padronizador().ajustar(df_treino[self.colunas].to_numpy(dtype=np.float64))
        tudo, _, _ = tensores(df_treino, self.colunas, self.padronizador)
        for s in self.hp["sementes"]:
            modelo, _, _ = _treinar_uma(tudo, s, self.hp, None, ep)
            self.modelos.append(modelo)
            self.epocas.append(ep)
        return self

    def prever(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        lote, cod, pos = tensores(df, self.colunas, self.padronizador)
        probs = []
        with torch.no_grad():
            for m in self.modelos:
                probs.append(m(lote.X, lote.mascara).exp().numpy()[cod, pos])
        P = np.stack(probs)
        return P.mean(axis=0), P.std(axis=0)

    # ------------------------------------------------------------------ disco
    def salvar(self, pasta: Path) -> None:
        pasta.mkdir(parents=True, exist_ok=True)
        torch.save([m.state_dict() for m in self.modelos], pasta / "pesos_mlp.pt")
        meta = {
            "colunas_entrada": self.colunas, "hiperparametros": self.hp, "epocas_por_semente": self.epocas,
            "media_padronizacao": self.padronizador.media.tolist(),
            "desvio_padronizacao": self.padronizador.desvio.tolist(),
            "eleicoes_treino": self.eleicoes_treino, "eleicoes_validacao_interna": self.eleicoes_validacao,
            "curvas_validacao": self.curvas_validacao,
            "arquitetura": f"MLP {len(self.colunas)} → " + " → ".join(map(str, self.hp["camadas_ocultas"]))
                           + " → 1 (pontuação) + softmax por disputa",
        }
        (pasta / "modelo.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def carregar(cls, pasta: Path) -> "ConjuntoMLP":
        meta = json.loads((pasta / "modelo.json").read_text(encoding="utf-8"))
        obj = cls(colunas=meta["colunas_entrada"], hp=meta["hiperparametros"])
        obj.padronizador = Padronizador(np.array(meta["media_padronizacao"]), np.array(meta["desvio_padronizacao"]))
        for sd in torch.load(pasta / "pesos_mlp.pt", weights_only=True):
            m = MLPDisputa(len(obj.colunas), obj.hp["camadas_ocultas"], obj.hp["dropout"])
            m.load_state_dict(sd)
            m.eval()
            obj.modelos.append(m)
        obj.epocas = meta["epocas_por_semente"]
        obj.eleicoes_treino = meta["eleicoes_treino"]
        obj.eleicoes_validacao = meta["eleicoes_validacao_interna"]
        obj.curvas_validacao = meta.get("curvas_validacao", {})
        return obj
