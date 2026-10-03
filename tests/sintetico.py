"""Gerador de dados SINTÉTICOS no formato dos arquivos do TSE — uso exclusivo em testes.

Nenhum número aqui é real: candidatos, partidos e votos são fictícios
("SINTETICO"), gerados para exercitar o pipeline de ponta a ponta sem
acesso à internet. Os testes gravam esses arquivos numa pasta temporária,
nunca em dados/.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

UFS = ["SP", "CE", "RS", "BA", "MG", "PA"]
ANOS = [1998, 2002, 2006, 2010, 2014, 2018, 2022]

# (nome civil, nome de urna, sigla por ano) — partidos/linhagens fictícios, mas com
# mudanças de sigla reais da tabela de linhagem para testar a padronização.
CANDIDATOS = {
    1998: [("ALFA SINTETICO UM", "ALFA", "PT"), ("BETA SINTETICO DOIS", "BETA", "PSDB"),
           ("GAMA SINTETICO TRES", "GAMA", "PMDB"), ("DELTA SINTETICO", "DELTA", "PRONA")],
    2002: [("ALFA SINTETICO UM", "ALFA", "PT"), ("EPSILON SINTETICO", "EPSILON", "PSDB"),
           ("ZETA SINTETICA", "ZETA", "PSB"), ("ETA SINTETICO", "ETA", "PPS")],
    2006: [("ALFA SINTETICO UM", "ALFA", "PT"), ("TETA SINTETICO", "TETA", "PSDB"),
           ("IOTA SINTETICA", "IOTA", "PSOL"), ("KAPA SINTETICO", "KAPA", "PDT")],
    2010: [("LAMBDA SINTETICA", "LAMBDA", "PT"), ("MI SINTETICO", "MI", "PSDB"),
           ("NI SINTETICA", "NI", "PV"), ("CSI SINTETICO", "CSI", "PSOL")],
    2014: [("LAMBDA SINTETICA", "LAMBDA", "PT"), ("OMICRON SINTETICO", "OMICRON", "PSDB"),
           ("NI SINTETICA", "NI", "PSB"), ("PI SINTETICO", "PI", "PSOL"), ("RO SINTETICO", "RO", "PSC")],
    2018: [("SIGMA SINTETICO", "SIGMA", "PSL"), ("TAU SINTETICO", "TAU", "PT"),
           ("UPSILON SINTETICO", "UPSILON", "PDT"), ("FI SINTETICO", "FI", "PSDB"),
           ("NI SINTETICA", "NI", "REDE")],
    2022: [("ALFA SINTETICO UM", "ALFA", "PT"), ("SIGMA SINTETICO", "SIGMA", "PL"),
           ("CHI SINTETICA", "CHI", "MDB"), ("UPSILON SINTETICO", "UPSILON", "PDT"),
           ("PSI SINTETICA", "PSI", "UNIÃO")],
}
CANDIDATOS_2026 = [
    ("ALFA SINTETICO UM", "ALFA", "PT", "APTO"), ("SIGMA SINTETICO", "SIGMA", "PL", "APTO"),
    ("OMEGA SINTETICO", "OMEGA", "NOVO", "APTO"), ("PSI SINTETICA", "PSI", "UNIÃO", "APTO"),
    ("FORA SINTETICO", "FORA", "PCO", "INAPTO"),
]
# Força latente (nacional) e inclinação ideológica por linhagem.
FORCA = {"PT": 1.6, "PSDB": 1.3, "MDB": 0.6, "PL": 1.5, "UNIAO": 1.4, "PSB": 0.4, "CIDADANIA": 0.2,
         "PSOL": -0.3, "PDT": 0.3, "PV": 0.5, "PODE": -0.5, "REDE": -0.2}
INCLINACAO = {"PT": 1.0, "PSDB": -0.6, "MDB": -0.1, "PL": -1.1, "UNIAO": -0.9, "PSB": 0.4, "CIDADANIA": 0.0,
              "PSOL": 0.7, "PDT": 0.5, "PV": 0.1, "PODE": -0.3, "REDE": 0.3}

COLS_VOT = ["DT_GERACAO", "HH_GERACAO", "ANO_ELEICAO", "CD_TIPO_ELEICAO", "NM_TIPO_ELEICAO", "NR_TURNO",
            "CD_ELEICAO", "DS_ELEICAO", "DT_ELEICAO", "TP_ABRANGENCIA", "SG_UF", "SG_UE", "NM_UE",
            "CD_MUNICIPIO", "NM_MUNICIPIO", "NR_ZONA", "CD_CARGO", "DS_CARGO", "SQ_CANDIDATO", "NR_CANDIDATO",
            "NM_CANDIDATO", "NM_URNA_CANDIDATO", "SG_PARTIDO", "NM_PARTIDO", "ST_VOTO_EM_TRANSITO",
            "QT_VOTOS_NOMINAIS", "NM_TIPO_DESTINACAO_VOTOS", "QT_VOTOS_NOMINAIS_VALIDOS",
            "CD_SIT_TOT_TURNO", "DS_SIT_TOT_TURNO"]


def _linhagem(sigla):
    from eleicao.partidos import linhagem
    return linhagem(sigla)


def municipios(n_por_uf: int, ano: int, rng_base: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(rng_base)
    linhas = []
    cod = 10000
    for uf in UFS:
        desl = {"SP": -0.4, "CE": 0.8, "RS": -0.6, "BA": 0.7, "MG": 0.0, "PA": 0.3}[uf]
        for i in range(n_por_uf):
            cod += 7
            linhas.append(dict(uf=uf, cd=cod, nome=f"MUNICIPIO SINTETICO {uf} {i:03d}",
                               ideologia=desl + rng.normal(0, 0.6), eleitores=int(rng.lognormal(9.3, 1.1)) + 500))
    df = pd.DataFrame(linhas)
    # Município "criado" em 2014: não existe antes.
    if ano < 2014:
        df = df[df.nome != f"MUNICIPIO SINTETICO CE {n_por_uf - 1:03d}"]
    # Exterior
    df = pd.concat([df, pd.DataFrame([dict(uf="ZZ", cd=99999, nome="EXTERIOR SINTETICO", ideologia=-0.2,
                                           eleitores=20000)])], ignore_index=True)
    return df


def _shares(mun, cands, rng, ano):
    lin = [_linhagem(c[2]) for c in cands]
    f = np.array([FORCA.get(l, -0.8) for l in lin])
    b = np.array([INCLINACAO.get(l, 0.0) for l in lin])
    # choque nacional por candidato e eleição + ruído local
    choque = rng.normal(0, 0.35, len(cands))
    s = f + choque + np.outer(mun.ideologia.values, b) + rng.normal(0, 0.25, (len(mun), len(cands)))
    e = np.exp(s - s.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def gerar_resultados(n_por_uf: int, semente: int = 1) -> dict[int, pd.DataFrame]:
    rng = np.random.default_rng(semente)
    out = {}
    for ano in ANOS:
        mun = municipios(n_por_uf, ano).reset_index(drop=True)
        mun["eleitores"] = (mun.eleitores * (1 + 0.02 * (ano - 1998) / 4)).astype(int)
        cands = CANDIDATOS[ano]
        sh = _shares(mun, cands, rng, ano)
        validos = (mun.eleitores * rng.uniform(0.70, 0.82, len(mun)) * 0.92).astype(int).values
        votos1 = np.floor(sh * validos[:, None]).astype(int)
        nac = votos1.sum(axis=0)
        ordem = np.argsort(-nac)
        linhas = []
        turnos = [1]
        if nac.max() / nac.sum() <= 0.5:
            turnos.append(2)
        for turno in turnos:
            if turno == 1:
                idx, V = list(range(len(cands))), votos1
            else:
                idx = list(ordem[:2])
                s2 = _shares(mun, [cands[i] for i in idx], rng, ano)
                V = np.floor(s2 * (validos[:, None] * 0.98)).astype(int)
            tot_nac = V.sum(axis=0)
            vencedor = int(np.argmax(tot_nac))
            for j, ci in enumerate(idx):
                nome, urna, sig = cands[ci]
                if turno == 1:
                    sit = "2º TURNO" if (2 in turnos and ci in ordem[:2]) else ("ELEITO" if j == vencedor and len(turnos) == 1 else "NÃO ELEITO")
                else:
                    sit = "ELEITO" if j == vencedor else "NÃO ELEITO"
                for m in range(len(mun)):
                    v = int(V[m, j])
                    zonas = [v] if mun.eleitores[m] < 30000 else [v // 2, v - v // 2]  # cidade com 2 zonas
                    for z, vz in enumerate(zonas, start=1):
                        linhas.append({
                            "ANO_ELEICAO": ano, "CD_TIPO_ELEICAO": 2, "NR_TURNO": turno, "SG_UF": mun.uf[m],
                            "CD_MUNICIPIO": mun.cd[m], "NM_MUNICIPIO": mun.nome[m], "NR_ZONA": z,
                            "CD_CARGO": 1, "DS_CARGO": "Presidente", "NR_CANDIDATO": 10 + ci,
                            "NM_CANDIDATO": nome, "NM_URNA_CANDIDATO": urna, "SG_PARTIDO": sig,
                            "QT_VOTOS_NOMINAIS": vz, "DS_SIT_TOT_TURNO": sit,
                            "aptos": int(mun.eleitores[m] / len(zonas)), "comparec": int(mun.eleitores[m] * 0.8 / len(zonas)),
                        })
        df = pd.DataFrame(linhas)
        # Linhas de outro cargo (devem ser descartadas pelo filtro de Presidente)
        extra = df.head(5).assign(CD_CARGO=6, DS_CARGO="Deputado Federal", QT_VOTOS_NOMINAIS=999999)
        out[ano] = pd.concat([df, extra], ignore_index=True)
    return out


def _csv_bytes(df: pd.DataFrame, cabecalho: bool = True) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, sep=";", index=False, header=cabecalho, quoting=1)
    return buf.getvalue().encode("latin-1")


def escrever(raiz: Path, n_por_uf: int = 25, semente: int = 1) -> dict[int, pd.DataFrame]:
    brutos = raiz / "dados" / "brutos"
    brutos.mkdir(parents=True, exist_ok=True)
    res = gerar_resultados(n_por_uf, semente)
    for ano, df in res.items():
        full = df.copy()
        for c in COLS_VOT:
            if c not in full.columns:
                full[c] = "#NULO#"
        full["QT_VOTOS_NOMINAIS_VALIDOS"] = full["QT_VOTOS_NOMINAIS"]
        full = full[COLS_VOT]
        with zipfile.ZipFile(brutos / f"votacao_candidato_munzona_{ano}.zip", "w", zipfile.ZIP_DEFLATED) as zf:
            if ano == 1998:
                # Arquivo antigo SEM cabeçalho, layout do LEIAME, sem arquivo _BRASIL.
                from eleicao.leitura import LAYOUT_ANTIGO
                lay = LAYOUT_ANTIGO["votacao"]
                leg = pd.DataFrame({c: "#NULO#" for c in lay}, index=df.index)
                mapa = {"ANO_ELEICAO": "ANO_ELEICAO", "NUM_TURNO": "NR_TURNO", "SIGLA_UF": "SG_UF",
                        "CODIGO_MUNICIPIO": "CD_MUNICIPIO", "NOME_MUNICIPIO": "NM_MUNICIPIO", "NUMERO_ZONA": "NR_ZONA",
                        "CODIGO_CARGO": "CD_CARGO", "NUMERO_CAND": "NR_CANDIDATO", "NOME_CANDIDATO": "NM_CANDIDATO",
                        "NOME_URNA_CANDIDATO": "NM_URNA_CANDIDATO", "DESCRICAO_CARGO": "DS_CARGO",
                        "DESC_SIT_CAND_TOT": "DS_SIT_TOT_TURNO", "SIGLA_PARTIDO": "SG_PARTIDO",
                        "TOTAL_VOTOS": "QT_VOTOS_NOMINAIS"}
                for a, b in mapa.items():
                    leg[a] = df[b].values
                for uf, g in leg.groupby("SIGLA_UF"):
                    zf.writestr(f"votacao_candidato_munzona_{ano}_{uf}.txt", _csv_bytes(g, cabecalho=False))
            else:
                for uf, g in full.groupby("SG_UF"):
                    zf.writestr(f"votacao_candidato_munzona_{ano}_{uf}.csv", _csv_bytes(g))
                zf.writestr(f"votacao_candidato_munzona_{ano}_BRASIL.csv", _csv_bytes(full))
            zf.writestr("leiame.pdf", b"%PDF sintetico")
        det = (df[df.CD_CARGO == 1].groupby(["ANO_ELEICAO", "NR_TURNO", "SG_UF", "CD_MUNICIPIO", "NR_ZONA"],
                                           as_index=False)[["aptos", "comparec"]].first()
               .rename(columns={"aptos": "QT_APTOS", "comparec": "QT_COMPARECIMENTO"}))
        det["CD_CARGO"], det["DS_CARGO"], det["CD_TIPO_ELEICAO"] = 1, "Presidente", 2
        if ano != 2006:  # 2006 sem detalhe: testa o caminho de dados ausentes
            with zipfile.ZipFile(brutos / f"detalhe_votacao_munzona_{ano}.zip", "w") as zf:
                zf.writestr(f"detalhe_votacao_munzona_{ano}_BRASIL.csv", _csv_bytes(det))

    # Candidatos 2026
    cand = pd.DataFrame([{"ANO_ELEICAO": 2026, "CD_TIPO_ELEICAO": 2, "NR_TURNO": 1, "CD_CARGO": 1,
                          "DS_CARGO": "PRESIDENTE", "NR_CANDIDATO": 50 + i, "NM_CANDIDATO": n,
                          "NM_URNA_CANDIDATO": u, "SG_PARTIDO": p, "NM_PARTIDO": f"PARTIDO {p}",
                          "DS_SITUACAO_CANDIDATURA": s, "DS_DETALHE_SITUACAO_CAND": "DEFERIDO" if s == "APTO" else "INDEFERIDO",
                          "DS_SIT_TOT_TURNO": "#NULO#", "ST_REELEICAO": "S" if u == "ALFA" else "N"}
                         for i, (n, u, p, s) in enumerate(CANDIDATOS_2026)])
    cand = pd.concat([cand, cand.head(1).assign(CD_CARGO=3, DS_CARGO="GOVERNADOR", NM_CANDIDATO="GOV SINTETICO")])
    with zipfile.ZipFile(brutos / "consulta_cand_2026.zip", "w") as zf:
        zf.writestr("consulta_cand_2026_BRASIL.csv", _csv_bytes(cand))

    # Eleitorado 2026 (inclui um município novo, sem histórico)
    mun = municipios(n_por_uf, 2026)
    mun = mun[mun.uf != "ZZ"]
    novo = pd.DataFrame([dict(uf="MG", cd=88888, nome="MUNICIPIO SINTETICO NOVO", eleitores=9000)])
    mun = pd.concat([mun, novo], ignore_index=True)
    perfil = pd.concat([mun.assign(QT_ELEITORES_PERFIL=(mun.eleitores * 1.1 * f).astype(int), DS_GENERO=g)
                        for f, g in [(0.52, "FEMININO"), (0.48, "MASCULINO")]])
    perfil = perfil.rename(columns={"uf": "SG_UF", "cd": "CD_MUNICIPIO", "nome": "NM_MUNICIPIO"})
    perfil["ANO_ELEICAO"] = 2026
    with zipfile.ZipFile(brutos / "perfil_eleitorado_2026.zip", "w") as zf:
        zf.writestr("perfil_eleitorado_2026.csv",
                    _csv_bytes(perfil[["ANO_ELEICAO", "SG_UF", "CD_MUNICIPIO", "NM_MUNICIPIO", "DS_GENERO",
                                       "QT_ELEITORES_PERFIL"]]))

    # Registro de pesquisas 2026 (metadados, sem percentuais — como no TSE)
    pesq = pd.DataFrame([
        {"ANO_ELEICAO": 2026, "NR_IDENTIFICACAO_PESQUISA": "BR-00001/2026", "NM_EMPRESA": "INSTITUTO SINTETICO A",
         "DT_REGISTRO": "01/09/2026", "DT_INICIO_PESQUISA": "05/09/2026", "DT_FIM_PESQUISA": "08/09/2026",
         "DS_CARGOS": "Presidente", "SG_UF": "BR", "NM_UE": "BRASIL", "QT_ENTREVISTADOS": "2000",
         "DS_METODOLOGIA_PESQUISA": "Entrevistas presenciais (sintético)"},
        {"ANO_ELEICAO": 2026, "NR_IDENTIFICACAO_PESQUISA": "CE-00002/2026", "NM_EMPRESA": "INSTITUTO SINTETICO B",
         "DT_REGISTRO": "02/09/2026", "DT_INICIO_PESQUISA": "06/09/2026", "DT_FIM_PESQUISA": "07/09/2026",
         "DS_CARGOS": "Governador", "SG_UF": "CE", "NM_UE": "CEARÁ", "QT_ENTREVISTADOS": "800",
         "DS_METODOLOGIA_PESQUISA": "Telefone (sintético)"},
    ])
    with zipfile.ZipFile(brutos / "pesquisa_eleitoral_2026.zip", "w") as zf:
        zf.writestr("pesquisa_eleitoral_2026.csv", _csv_bytes(pesq))
    return res
