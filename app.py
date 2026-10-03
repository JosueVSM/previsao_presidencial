"""Interface Streamlit — estimativa EXPERIMENTAL de votos para Presidente por rede neural.

Executar:  streamlit run app.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from eleicao import agregacao, config
from eleicao import amostra as amostra_mod
from eleicao.features import colunas_entrada

st.set_page_config(page_title="Estimativa experimental — Presidente", page_icon="🗳️", layout="wide")

COR_REDE = "#3b6ea8"      # estimativa da rede neural
COR_OFICIAL = "#5f6368"   # resultado oficial do TSE
COR_NACIONAL = "#c9a227"  # estimativa nacional agregada
COR_REF = "#9aa5b1"       # referência simples

ROTULO_REDE = "Estimativa da rede neural (experimental)"
ROTULO_OFICIAL = "Resultado oficial (TSE)"
ROTULO_PESQ = "Resultado de pesquisa"


# ---------------------------------------------------------------------------
# Carregamento
# ---------------------------------------------------------------------------
def _pq(caminho):
    return pd.read_parquet(caminho) if caminho.exists() else None


def _js(caminho):
    return json.loads(caminho.read_text(encoding="utf-8")) if caminho.exists() else None


def _csv(caminho):
    return pd.read_csv(caminho) if caminho.exists() else None


@st.cache_data(show_spinner="Carregando dados…")
def carregar():
    A = config.ANO_ALVO
    P, R = config.DIR_PROCESSADOS, config.DIR_RESULTADOS
    M, F = config.DIR_MODELOS / f"mlp_{A}", config.ARQ_FONTES
    # Sem o pipeline rodado, cai na amostra versionada no repositório em vez de
    # abrir vazio. Ver eleicao/amostra.py.
    amostra = not (P / "resultados_municipio.parquet").exists() and amostra_mod.disponivel()
    if amostra:
        P = R = M = config.DIR_AMOSTRA
        F = config.DIR_AMOSTRA / "fontes.json"
    d = {
        "amostra": amostra,
        "amostra_meta": _js(config.DIR_AMOSTRA / "AMOSTRA.json") if amostra else None,
        "oficial": _pq(P / "resultados_municipio.parquet"),
        "candidatos": _pq(P / f"candidatos_{A}.parquet"),
        "pesq_registro": _pq(P / f"pesquisas_registro_{A}.parquet"),
        "previsao": _pq(R / f"previsao_{A}.parquet"),
        "previsao_meta": _js(R / f"previsao_{A}_meta.json"),
        "backtest": _pq(R / "previsoes_backtest.parquet"),
        "metricas": _csv(R / "avaliacao_metricas.csv"),
        "bt_nacional": _csv(R / "backtest_nacional.csv"),
        "verificacoes": _js(R / "relatorio_verificacoes.json"),
        "modelo_meta": _js(M / "modelo.json"),
        "fontes": _js(F) or {},
    }
    try:
        from eleicao.pesquisas import carregar_manuais
        d["pesq_manuais"] = carregar_manuais()
    except Exception as e:  # noqa: BLE001
        d["pesq_manuais"] = None
        d["pesq_erro"] = str(e)
    if d["oficial"] is not None:
        d["oficial"] = d["oficial"][d["oficial"].ano < A]  # nunca exibir/usar resultado do ano-alvo como entrada
    return d


D = carregar()

st.title("Estimativa experimental de votos para Presidente")
st.warning(
    "**As estimativas desta página são experimentais**, produzidas por uma rede neural treinada "
    "com resultados de eleições anteriores. Elas **não são resultado oficial, não são pesquisa eleitoral "
    "e não garantem quem vencerá.** Campanha, candidaturas, alianças, comparecimento e muitos fatores "
    "que não estão nos dados podem alterar o resultado. Esta página não faz recomendação de voto.",
    icon="⚠️",
)

if D["oficial"] is None:
    st.error("Nenhum dado processado foi encontrado. No terminal, na pasta do projeto, rode:")
    st.code("python -m eleicao.pipeline tudo", language="bash")
    st.stop()

if D["amostra"]:
    anos_bt = (D["amostra_meta"] or {}).get("backtest_contem_eleicoes") or []
    st.info(
        "**Você está vendo a amostra que acompanha o repositório.** Os números são os mesmos "
        "produzidos pelo pipeline — nada foi recalculado —, mas o teste retrospectivo navegável "
        f"traz só {', '.join(map(str, anos_bt)) or 'uma eleição'}; a tabela da aba *Avaliação da rede* "
        "cobre as quatro eleições de teste. Para gerar o conjunto completo, rode "
        "`python -m eleicao.pipeline tudo` (baixa cerca de 2,3 GB do TSE).",
        icon="📦",
    )

# ---------------------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------------------
opcoes_eleicao = []
if D["previsao"] is not None:
    opcoes_eleicao.append(f"{config.ANO_ALVO} — estimativa")
if D["backtest"] is not None:
    opcoes_eleicao += [f"{a} — teste retrospectivo" for a in sorted(D["backtest"].ano.unique(), reverse=True)]
if not opcoes_eleicao:
    st.info("A rede ainda não foi treinada. Rode `python -m eleicao.pipeline avaliar` e "
            "`python -m eleicao.pipeline prever`. Enquanto isso, a aba de histórico oficial está disponível.")

with st.sidebar:
    st.markdown("**Alunos**")
    st.markdown("Josué de Vasconcelos Silveira — matrícula 2517534")
    st.markdown("Plinio Rodrigues — matrícula 2526504")
    st.divider()
    st.header("Seleção")
    escolha = st.selectbox("Eleição", opcoes_eleicao or ["(sem estimativas)"])
    ano_sel = int(escolha[:4]) if opcoes_eleicao else None
    modo_alvo = ano_sel == config.ANO_ALVO
    base = D["previsao"] if modo_alvo else (D["backtest"][D["backtest"].ano == ano_sel] if ano_sel else None)

    turnos_disp = sorted(base.turno.unique()) if base is not None else []
    turno = st.radio("Turno", [1, 2], format_func=lambda t: f"{t}º turno", horizontal=True)
    if base is not None and turno not in turnos_disp:
        if modo_alvo and turno == 2:
            st.info("Os candidatos do 2º turno de 2026 só são conhecidos após a totalização do 1º turno. "
                    "Quando o TSE atualizar o cadastro de candidaturas, rode o pipeline de novo.")
        else:
            st.info(f"Não houve {turno}º turno nesta eleição.")
        base = None
    elif base is not None:
        base = base[base.turno == turno]

    nivel = st.radio("Nível", ["Brasil", "Estado", "Município"], horizontal=True)
    ufs = sorted((base if base is not None else D["oficial"]).uf.unique())
    uf = mun = None
    if nivel in ("Estado", "Município"):
        uf = st.selectbox("Estado", ufs, format_func=lambda u: f"{u} — {config.NOMES_UF.get(u, u)}",
                          index=ufs.index("CE") if "CE" in ufs else 0)
    if nivel == "Município":
        fonte_m = base if base is not None else D["oficial"]
        muns = (fonte_m[fonte_m.uf == uf][["cd_municipio", "nm_municipio"]].drop_duplicates()
                .sort_values("nm_municipio"))
        rot_mun = dict(zip(muns.cd_municipio.astype(int), muns.nm_municipio))
        mun = st.selectbox("Município", list(rot_mun), format_func=lambda c: rot_mun.get(c, str(c)))
    st.caption("O 'teste retrospectivo' mostra o que a rede teria estimado para uma eleição passada "
               "sem ter visto o resultado dela — útil para julgar a confiabilidade.")


def nome_local() -> str:
    if nivel == "Brasil":
        return "Brasil"
    if nivel == "Estado":
        return config.NOMES_UF.get(uf, uf)
    nm = base if base is not None else D["oficial"]
    return f"{nm[(nm.uf == uf) & (nm.cd_municipio == mun)].nm_municipio.iloc[0].title()} ({uf})"


def estimativa_local(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if nivel == "Brasil":
        return agregacao.agregar(df, col, [])
    if nivel == "Estado":
        return agregacao.agregar(df[df.uf == uf], col, ["uf"])
    return df[(df.uf == uf) & (df.cd_municipio == mun)]


def oficial_local(ano: int, t: int) -> pd.DataFrame:
    o = D["oficial"]
    o = o[(o.ano == ano) & (o.turno == t)]
    if nivel == "Estado":
        o = o[o.uf == uf]
    elif nivel == "Município":
        o = o[(o.uf == uf) & (o.cd_municipio == mun)]
    # Agrupa pelo nome civil normalizado; nome de urna e sigla são só rótulos e
    # não entram na chave (ver agregacao._rotulos).
    g = o.groupby("candidato", as_index=False)["votos"].sum()
    g = g.merge(agregacao._rotulos(o, [], "votos"), on="candidato", how="left")
    g["pct_oficial"] = g.votos / g.votos.sum() if g.votos.sum() > 0 else np.nan
    return g


def rotulo_cand(df):
    return df["nome_urna"].str.title() + " (" + df["partido"] + ")"


def barras(series: list[tuple[str, pd.Series, str]], categorias: list[str], titulo: str):
    fig = go.Figure()
    for nome, vals, cor in series:
        fig.add_bar(y=categorias, x=vals * 100, name=nome, orientation="h", marker_color=cor,
                    text=[f"{v * 100:.1f}%" for v in vals], textposition="outside")
    fig.update_layout(barmode="group", title=titulo, xaxis_title="% dos votos válidos",
                      yaxis=dict(autorange="reversed"), height=120 + 60 * len(categorias) * max(1, len(series)) // 2,
                      legend=dict(orientation="h", y=-0.15), margin=dict(l=10, r=40, t=50, b=10))
    fig.update_xaxes(range=[0, 105])
    return fig


abas = st.tabs(["📊 Estimativa", "🗺️ Comparar localidades", "📜 Histórico oficial",
                "🧪 Avaliação da rede", "📋 Pesquisas", "ℹ️ Fontes, método e limitações"])

# ---------------------------------------------------------------------------
# 1. Estimativa
# ---------------------------------------------------------------------------
with abas[0]:
    if base is None or base.empty:
        st.info("Não há estimativa para a combinação selecionada.")
    else:
        local = nome_local()
        st.subheader(f"{local} — {ano_sel}, {turno}º turno")
        est = estimativa_local(base, "p_rede")
        nac = agregacao.agregar(base, "p_rede", [])
        tab = est[["candidato", "nome_urna", "partido", "p_rede"]].merge(
            nac[["candidato", "p_rede"]].rename(columns={"p_rede": "p_nacional"}), on="candidato")
        if nivel == "Município":
            tab = tab.merge(est[["candidato", "p_rede_dp"]], on="candidato")
        if not modo_alvo:
            tab = tab.merge(oficial_local(ano_sel, turno)[["candidato", "pct_oficial"]], on="candidato", how="left")
        tab = tab.sort_values("p_rede", ascending=False)
        cats = rotulo_cand(tab).tolist()
        series = [(f"{ROTULO_REDE} — {local}", tab.p_rede, COR_REDE)]
        if nivel != "Brasil":
            series.append((f"{ROTULO_REDE} — Brasil (agregado)", tab.p_nacional, COR_NACIONAL))
        if not modo_alvo:
            series.append((f"{ROTULO_OFICIAL} — {local}", tab.pct_oficial.fillna(0), COR_OFICIAL))
        st.plotly_chart(barras(series, cats, "Porcentagem de votos válidos por candidato"), width="stretch")

        exib = pd.DataFrame({"Candidato": tab.nome_urna.str.title(), "Partido": tab.partido,
                             "Estimativa da rede (%)": (tab.p_rede * 100).round(1)})
        if nivel != "Brasil":
            exib["Estimativa nacional agregada (%)"] = (tab.p_nacional * 100).round(1).values
        if nivel == "Município":
            exib["Variação entre redes do conjunto (p.p.)"] = (tab.p_rede_dp * 100).round(1).values
        if not modo_alvo:
            exib["Resultado oficial TSE (%)"] = (tab.pct_oficial * 100).round(1).values
            exib["Erro da rede (p.p.)"] = ((tab.p_rede - tab.pct_oficial) * 100).round(1).values
        st.dataframe(exib, hide_index=True, width="stretch")
        st.caption(
            "Porcentagem de votos válidos = votos do candidato ÷ soma dos votos de todos os candidatos "
            "(brancos e nulos excluídos). A estimativa nacional e as estaduais são médias das estimativas "
            "municipais ponderadas pelos votos válidos esperados de cada município (ver aba de método). "
            "A 'variação entre redes' indica só a instabilidade do treino — não é margem de erro.")
        if modo_alvo:
            meta = D["previsao_meta"] or {}
            st.caption(f"Estimativa gerada em {meta.get('gerado_em', '—')}. "
                       f"Rede treinada com as eleições {', '.join(meta.get('eleicoes_treino', []))}. "
                       f"Pesquisas como entrada da rede: {'sim' if meta.get('pesquisas_como_entrada') else 'não'}.")
            if nivel == "Município" and est["sem_historico_mun"].iloc[0] == 1:
                st.info("Este município não existia (ou não tinha votação) na eleição anterior; "
                        "a estimativa usa o comportamento do estado.")
            cands = D["candidatos"]
            if cands is not None and (~cands.na_urna).any():
                fora = cands[~cands.na_urna]
                st.caption("Candidaturas registradas mas não aptas no arquivo do TSE (não incluídas): "
                           + ", ".join(f"{n.title()} ({s})" for n, s in zip(fora.nome_urna, fora.situacao_candidatura)))
            st.caption(f"Resultado oficial de {config.ANO_ALVO}: **ainda não disponível / não utilizado** pela rede.")
        with st.expander("Comparar com a referência simples (persistência partidária)"):
            ref = estimativa_local(base, "p_persistencia")
            r2 = tab[["candidato", "nome_urna", "partido", "p_rede"]].merge(ref[["candidato", "p_persistencia"]], on="candidato")
            st.dataframe(pd.DataFrame({"Candidato": r2.nome_urna.str.title(), "Partido": r2.partido,
                                       "Rede neural (%)": (r2.p_rede * 100).round(1),
                                       "Persistência partidária (%)": (r2.p_persistencia * 100).round(1)}),
                         hide_index=True, width="stretch")
            st.caption("A referência repete a votação do partido (ou do próprio candidato) na eleição anterior "
                       "no mesmo município, renormalizada entre os candidatos atuais. Serve apenas de comparação.")

# ---------------------------------------------------------------------------
# 2. Comparar localidades
# ---------------------------------------------------------------------------
with abas[1]:
    if base is None or base.empty:
        st.info("Não há estimativa para a combinação selecionada.")
    else:
        cands = base[["candidato", "nome_urna", "partido"]].drop_duplicates()
        rot_cand = dict(zip(cands.candidato, rotulo_cand(cands)))
        c = st.selectbox("Candidato", list(rot_cand), format_func=lambda k: rot_cand.get(k, str(k)))
        nac = agregacao.agregar(base, "p_rede", [])
        v_nac = float(nac.loc[nac.candidato == c, "p_rede"].iloc[0])
        if nivel == "Brasil":
            g = agregacao.agregar(base, "p_rede", ["uf"])
            g = g[g.candidato == c].assign(local=lambda x: x.uf.map(lambda u: config.NOMES_UF.get(u, u)))
            titulo = "Estimativa por estado"
        else:
            g = base[(base.uf == uf) & (base.candidato == c)].assign(local=lambda x: x.nm_municipio.str.title())
            titulo = f"Estimativa por município — {config.NOMES_UF.get(uf, uf)}"
        g = g.sort_values("p_rede", ascending=False)
        n_max = st.slider("Quantidade de localidades no gráfico", 5, max(5, min(60, len(g))), min(27, len(g))) if len(g) > 5 else len(g)
        gg = pd.concat([g.head(n_max // 2 + n_max % 2), g.tail(n_max // 2)]).drop_duplicates("local") if len(g) > n_max else g
        fig = go.Figure(go.Bar(x=gg.local, y=gg.p_rede * 100, marker_color=COR_REDE, name=ROTULO_REDE))
        fig.add_hline(y=v_nac * 100, line_dash="dash", line_color=COR_NACIONAL,
                      annotation_text=f"Brasil (agregado): {v_nac * 100:.1f}%")
        fig.update_layout(title=f"{titulo} — {rot_cand[c]}", yaxis_title="% dos votos válidos (estimativa)", height=480)
        st.plotly_chart(fig, width="stretch")
        if len(g) > n_max:
            st.caption(f"Mostrando as {n_max // 2 + n_max % 2} maiores e as {n_max // 2} menores estimativas de {len(g)}.")
        st.dataframe(pd.DataFrame({"Localidade": g.local, "Estimativa da rede (%)": (g.p_rede * 100).round(1),
                                   "Diferença para o Brasil (p.p.)": ((g.p_rede - v_nac) * 100).round(1)}),
                     hide_index=True, width="stretch", height=320)

# ---------------------------------------------------------------------------
# 3. Histórico oficial
# ---------------------------------------------------------------------------
with abas[2]:
    st.markdown(f"**{ROTULO_OFICIAL}** — eleições usadas como contexto e treino. Local: **{nome_local()}**.")
    anos_h = sorted(D["oficial"].ano.unique(), reverse=True)
    c1, c2 = st.columns(2)
    ah = c1.selectbox("Eleição", anos_h, key="hist_ano")
    th = c2.radio("Turno", sorted(D["oficial"][D["oficial"].ano == ah].turno.unique()), key="hist_turno",
                  format_func=lambda t: f"{t}º turno", horizontal=True)
    of = oficial_local(ah, th).sort_values("pct_oficial", ascending=False)
    if of.empty:
        st.info("Sem dados oficiais para esta localidade nesta eleição (o município pode não existir na época).")
    else:
        st.plotly_chart(barras([(ROTULO_OFICIAL, of.pct_oficial, COR_OFICIAL)], rotulo_cand(of).tolist(),
                               f"{ah}, {th}º turno"), width="stretch")
        st.dataframe(pd.DataFrame({"Candidato": of.nome_urna.str.title(), "Partido": of.partido,
                                   "Votos": of.votos, "% válidos": (of.pct_oficial * 100).round(2)}),
                     hide_index=True, width="stretch")
    with st.expander("Evolução por partido (1º turno, pela linhagem partidária)"):
        o = D["oficial"][D["oficial"].turno == 1]
        if nivel == "Estado":
            o = o[o.uf == uf]
        elif nivel == "Município":
            o = o[(o.uf == uf) & (o.cd_municipio == mun)]
        ev = o.groupby(["ano", "linhagem"])["votos"].sum().reset_index()
        ev["pct"] = 100 * ev.votos / ev.groupby("ano").votos.transform("sum")
        principais = ev.groupby("linhagem").pct.max().nlargest(8).index
        fig = go.Figure()
        for l in principais:
            e = ev[ev.linhagem == l]
            fig.add_scatter(x=e.ano, y=e.pct, mode="lines+markers", name=l)
        fig.update_layout(yaxis_title="% dos votos válidos (oficial)", xaxis=dict(dtick=4), height=420)
        st.plotly_chart(fig, width="stretch")
        st.caption("Partidos agrupados pela linhagem (sigla atual que sucedeu o partido por mudança de nome, "
                   "incorporação ou fusão). Pontos ausentes = partido sem candidato naquela eleição.")

# ---------------------------------------------------------------------------
# 4. Avaliação
# ---------------------------------------------------------------------------
with abas[3]:
    met = D["metricas"]
    if met is None:
        st.info("A avaliação ainda não foi executada (`python -m eleicao.pipeline avaliar`).")
    else:
        st.markdown(
            "**Validação temporal.** Para cada eleição de teste, a rede foi treinada apenas com eleições "
            "anteriores e avaliada na eleição de teste, que ficou **inteira** fora do treino (todos os "
            "municípios, os dois turnos). Erros em **pontos percentuais (p.p.)** de votos válidos.")
        tabm = met.rename(columns={
            "eleicao_teste": "Eleição de teste", "turno": "Turno", "modelo": "Modelo",
            "eleicoes_treino": "Eleições de treino", "mae_pp": "EAM municipal (p.p.)",
            "rmse_pp": "REQM municipal (p.p.)", "mae_pp_ponderado_votos": "EAM ponderado por votos (p.p.)",
            "acerto_mais_votado_municipio_pct": "Acerto do mais votado no município (%)",
            "erro_nacional_mae_pp": "EAM do agregado nacional (p.p.)",
            "erro_nacional_max_pp": "Maior erro nacional (p.p.)", "n_municipios": "Municípios",
            "n_candidatos": "Candidatos", "epocas_medias": "Épocas"})
        st.dataframe(tabm.round(2), hide_index=True, width="stretch")
        for t in sorted(met.turno.unique()):
            mt = met[met.turno == t]
            fig = go.Figure()
            for modelo, cor in [("Rede neural (MLP)", COR_REDE), ("Referência: persistência partidária", COR_REF),
                                ("Referência: divisão igual", "#d0d4d9")]:
                m = mt[mt.modelo == modelo]
                fig.add_bar(x=m.eleicao_teste.astype(str), y=m.mae_pp, name=modelo, marker_color=cor,
                            text=m.mae_pp.round(1), textposition="outside")
            fig.update_layout(barmode="group", title=f"Erro absoluto médio por município — {t}º turno",
                              yaxis_title="p.p.", height=380, legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig, width="stretch")
        rede = met[met.modelo == "Rede neural (MLP)"]
        n_el = rede.eleicao_teste.nunique()
        st.markdown(
            f"**Como ler.** Há muitos municípios, mas apenas **{n_el} eleições de teste** e cerca de seis "
            "eleições de treino. Tudo o que é comum a uma eleição inteira — o desempenho nacional de um "
            "candidato novo, uma onda de última hora, um partido que muda de candidato — aparece só uma vez "
            "por eleição. Por isso a rede tende a acertar melhor a **distribuição geográfica** (onde um "
            "candidato vai melhor ou pior) do que o **nível nacional**, e o erro de uma eleição para outra "
            "varia muito. Com tão poucas eleições independentes, essas métricas não permitem afirmar que a "
            "rede aprendeu padrões confiáveis para 2026; elas mostram o tamanho dos erros que já ocorreram.")
        if D["bt_nacional"] is not None:
            with st.expander("Agregado nacional estimado × resultado oficial (testes retrospectivos)"):
                bn = D["bt_nacional"]
                bn = bn[bn.modelo == "Rede neural (MLP)"]
                st.dataframe(pd.DataFrame({
                    "Eleição": bn.ano, "Turno": bn.turno, "Candidato": bn.nome_urna.str.title(), "Partido": bn.partido,
                    "Estimativa da rede (%)": (bn.pct_estimado * 100).round(1),
                    "Oficial TSE (%)": (bn.pct_oficial * 100).round(1), "Erro (p.p.)": bn.erro_pp.round(1)}),
                    hide_index=True, width="stretch")
        mm = D["modelo_meta"]
        if mm and mm.get("curvas_validacao", {}).get("media"):
            with st.expander("Curva de validação usada para escolher o número de épocas (rede final)"):
                y = mm["curvas_validacao"]["media"]
                fig = go.Figure(go.Scatter(x=list(range(1, len(y) + 1)), y=y, mode="lines", line_color=COR_REDE))
                fig.add_vline(x=mm["epocas_por_semente"][0], line_dash="dash",
                              annotation_text=f"escolhida: {mm['epocas_por_semente'][0]}")
                fig.update_layout(xaxis_title="Época", yaxis_title="Perda média nas eleições de validação", height=320)
                st.plotly_chart(fig, width="stretch")
                st.caption(f"Validação interna nas eleições {', '.join(mm.get('eleicoes_validacao_interna', []))}.")

# ---------------------------------------------------------------------------
# 5. Pesquisas
# ---------------------------------------------------------------------------
with abas[4]:
    st.markdown(
        "O Portal de Dados Abertos do TSE publica o **registro** das pesquisas (instituto, datas, tamanho da "
        "amostra, metodologia), mas **não publica os percentuais** de intenção de voto. Por isso esta aba "
        "mostra o registro oficial e, separadamente, percentuais que o próprio usuário tenha informado.")
    reg = D["pesq_registro"]
    st.subheader(f"Pesquisas para Presidente registradas no TSE ({config.ANO_ALVO})")
    if reg is None or reg.empty:
        st.info("Registro de pesquisas não disponível nos dados baixados.")
    else:
        mostrar = reg.rename(columns={"registro": "Registro", "instituto": "Instituto", "data_inicio": "Início do campo",
                                      "data_fim": "Fim do campo", "data_divulgacao": "Divulgação",
                                      "abrangencia": "Abrangência", "n_entrevistados": "Entrevistados",
                                      "metodologia": "Metodologia", "margem_erro": "Margem de erro"})
        mostrar = mostrar.loc[:, (mostrar != "").any()]
        st.dataframe(mostrar, hide_index=True, width="stretch", height=300)
        st.caption(f"{len(reg)} registros. Fonte: TSE — Pesquisas eleitorais {config.ANO_ALVO}.")
    st.subheader(f"{ROTULO_PESQ} (informado pelo usuário)")
    pm = D.get("pesq_manuais")
    if D.get("pesq_erro"):
        st.error(f"Arquivo de pesquisas inválido: {D['pesq_erro']}")
    if pm is None or pm[pm.ano == config.ANO_ALVO].empty:
        st.info("Nenhum percentual de pesquisa foi fornecido. Para incluir, preencha "
                "`dados/manual/pesquisas_resultados.csv` seguindo o modelo `pesquisas_resultados_MODELO.csv`.")
    else:
        p = pm[pm.ano == config.ANO_ALVO].sort_values("data_fim", ascending=False)
        st.dataframe(p[["instituto", "nr_registro_tse", "data_inicio", "data_fim", "abrangencia", "n_entrevistados",
                        "margem_erro_pp", "turno", "candidato", "pct", "base"]],
                     hide_index=True, width="stretch")
        st.caption("Pesquisas medem intenção de voto de uma amostra numa data; não são resultado observado, "
                   "e pesquisas nacionais ou estaduais não dizem nada diretamente sobre um município específico.")
    if D["previsao_meta"]:
        st.caption("Pesquisas como entrada da rede: " +
                   ("**sim** (média de pesquisas nacionais nos 30 dias anteriores a cada eleição)."
                    if D["previsao_meta"].get("pesquisas_como_entrada") else
                    "**não** — só seriam usadas se houvesse percentuais para todas as eleições de treino e para 2026."))

# ---------------------------------------------------------------------------
# 6. Fontes, método e limitações
# ---------------------------------------------------------------------------
with abas[5]:
    st.subheader("Fontes dos dados")
    fontes = D["fontes"]
    if fontes:
        linhas = [{"Conjunto": k, "Descrição": v.get("descricao", ""), "Fonte": v.get("fonte", ""),
                   "Endereço": v.get("url", ""), "Data de acesso": v.get("data_acesso", "—"),
                   "Período coberto": v.get("periodo_coberto", ""),
                   "Colunas utilizadas": ", ".join(v.get("colunas_utilizadas", []))} for k, v in fontes.items()]
        st.dataframe(pd.DataFrame(linhas), hide_index=True, width="stretch",
                     column_config={"Endereço": st.column_config.LinkColumn("Endereço")})
    else:
        st.info("Catálogo de fontes vazio (dados/fontes.json).")
    meta = D["previsao_meta"] or {}
    st.markdown(f"**Última atualização da estimativa:** {meta.get('gerado_em', '—')}")

    st.subheader("Método")
    mm = D["modelo_meta"] or {}
    st.markdown(f"""
- **Unidade:** cada *disputa* é um município num turno; dentro dela há um candidato por linha.
- **Rede neural:** perceptron multicamadas (MLP, PyTorch) — {mm.get('arquitetura', 'ver README')}. A mesma rede
  dá uma pontuação a cada candidato e um **softmax entre os candidatos da disputa** transforma as pontuações em
  porcentagens entre 0 e 100% que somam exatamente 100% em cada município.
- **Perda:** entropia cruzada entre as porcentagens oficiais e as previstas, com peso √(votos válidos) por município.
- **Entradas** ({len(mm.get('colunas_entrada', colunas_entrada(False)))}): desempenho do partido (pela linhagem) e do
  próprio candidato na eleição anterior (município, estado, Brasil), indicadores de quem concorreu antes, de reeleição
  e do partido do eleito anterior, e características do município (região, eleitorado, comparecimento e fragmentação
  anteriores). Nomes de candidatos não entram na rede.
- **Treino:** conjunto de {len(mm.get('epocas_por_semente', [])) or 5} redes (sementes diferentes), AdamW, padronização
  das entradas ajustada só no treino, número de épocas escolhido por validação temporal interna (eleições inteiras).
- **Agregação:** estado e Brasil = média das estimativas municipais ponderada pelos votos válidos esperados
  (eleitorado do ano × votos válidos por eleitor apto na eleição anterior).
""")
    with st.expander("Hiperparâmetros e entradas"):
        st.json({"hiperparametros": mm.get("hiperparametros", config.HIPERPARAMETROS),
                 "colunas_entrada": mm.get("colunas_entrada", []), "epocas": mm.get("epocas_por_semente", [])})

    st.subheader("Limitações")
    st.markdown("""
- **Poucas eleições independentes:** só há meia dúzia de eleições presidenciais com dados por município; muitos
  municípios não substituem muitas eleições.
- **Candidatos e partidos novos:** a rede só conhece o passado. Um candidato forte de um partido que não disputou a
  eleição anterior tende a ser **subestimado**; mudanças de partido e fusões são tratadas por uma tabela de linhagem
  editável, de forma aproximada.
- **Sem dados de campanha:** alianças, debates, economia, eventos de última hora e comparecimento não estão no modelo.
- **Pesquisas:** o TSE não publica percentuais; sem um arquivo fornecido pelo usuário, a rede não usa pesquisas.
- **2º turno de 2026:** só pode ser estimado depois que o TSE registrar quem disputa o 2º turno.
- **Incerteza:** a variação entre redes do conjunto não é intervalo de confiança; o erro real visto nos testes
  retrospectivos (aba de avaliação) é a melhor indicação do tamanho possível do erro.
""")
    ver = D["verificacoes"]
    if ver:
        with st.expander(f"Verificações automáticas — {'todas OK' if ver['todas_ok'] else 'HÁ FALHAS'} ({ver['gerado_em']})"):
            st.dataframe(pd.DataFrame(ver["itens"]), hide_index=True, width="stretch")
