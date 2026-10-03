import numpy as np
import pandas as pd
import torch

from eleicao.modelo import ConjuntoMLP, MLPDisputa, Padronizador, tensores


def _dados(n_disp=400, semente=0):
    rng = np.random.default_rng(semente)
    linhas = []
    for d in range(n_disp):
        k = rng.integers(2, 7)  # número variável de candidatos
        x = rng.normal(size=k)
        p = np.exp(1.5 * x) / np.exp(1.5 * x).sum()
        for c in range(k):
            linhas.append({"id_disputa": f"{2000 + 4 * (d % 3)}-{d}", "ano": 2000 + 4 * (d % 3),
                           "candidato": f"C{c}", "x": x[c], "ruido": rng.normal(), "y": p[c],
                           "votos_validos_disputa": int(rng.integers(100, 10000))})
    return pd.DataFrame(linhas).sort_values("id_disputa").reset_index(drop=True)


def test_softmax_por_disputa_soma_1():
    df = _dados(50)
    pad = Padronizador().ajustar(df[["x", "ruido"]].to_numpy())
    lote, cod, pos = tensores(df, ["x", "ruido"], pad)
    m = MLPDisputa(2, [8], 0.0)
    with torch.no_grad():
        p = m(lote.X, lote.mascara).exp().numpy()
    assert np.allclose(p.sum(axis=1), 1, atol=1e-6)
    assert (p[~lote.mascara.numpy()] == 0).all()        # posições de preenchimento
    pr = p[cod, pos]
    assert ((pr >= 0) & (pr <= 1)).all()


def test_rede_aprende_sinal_e_salva(tmp_path):
    df = _dados(600)
    conj = ConjuntoMLP(["x", "ruido"]).treinar(df, verbose=False)
    teste = _dados(200, semente=1)
    p, dp = conj.prever(teste)
    soma = pd.Series(p).groupby(teste.id_disputa.values).sum()
    assert np.allclose(soma, 1, atol=1e-5)
    unif = 1 / teste.groupby("id_disputa").candidato.transform("count")
    assert np.abs(p - teste.y).mean() < 0.5 * np.abs(unif - teste.y).mean()
    conj.salvar(tmp_path)
    p2, _ = ConjuntoMLP.carregar(tmp_path).prever(teste)
    assert np.allclose(p, p2, atol=1e-6)
