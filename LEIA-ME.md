# Leia-me — como abrir o aplicativo

**Alunos:**

- Josué de Vasconcelos Silveira — matrícula 2517534
- Plinio Rodrigues — matrícula 2526504

Este projeto estima, com uma rede neural, a porcentagem de votos válidos de cada candidato à
Presidência **por município**, e mostra tudo numa interface web (Streamlit).

A explicação de como a rede foi construída está em [`documentacao_rede_neural.html`](documentacao_rede_neural.html)
(abra com dois cliques, em qualquer navegador). Os detalhes técnicos completos estão no
[`README.md`](README.md).

---

## Abrir o aplicativo

**Não é preciso baixar nada do TSE para ver o projeto funcionando.** O repositório já traz a pasta
`amostra/`, com os resultados que o pipeline produziu: a tabela de métricas das quatro eleições de
teste, o teste retrospectivo navegável de 2022 por município, a estimativa de 2026 e o relatório das
verificações. O app detecta sozinho e avisa na tela que está usando a amostra.

Depois de instalar as bibliotecas (passo 1 da seção seguinte), abra o terminal na pasta do projeto
e rode:

**Windows (PowerShell)**

```powershell
.venv\Scripts\activate
streamlit run app.py
```

**Linux ou macOS**

```bash
source .venv/bin/activate
streamlit run app.py
```

O navegador abre sozinho em **http://localhost:8501**. Se não abrir, copie esse endereço e cole no
navegador.

Para **encerrar**, volte ao terminal e pressione `Ctrl + C`.

---

## Instalar e, se quiser, reproduzir tudo

O passo 1 é obrigatório uma única vez. Os passos 2 e 3 só fazem sentido se você quiser **refazer**
os resultados a partir dos arquivos oficiais, em vez de olhar a amostra.

### 1. Criar o ambiente virtual e instalar as bibliotecas

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**Linux ou macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Instala pandas, numpy, pyarrow, PyTorch, Streamlit, Plotly e pytest. Demora alguns minutos,
principalmente por causa do PyTorch.

### 2. Rodar o pipeline completo (opcional)

```bash
python -m eleicao.pipeline tudo
```

Isso executa, em ordem: **baixar → preparar → avaliar → prever → verificar → amostra**. A partir
daí o app passa a usar os dados completos, e não mais a amostra — o aviso some da tela.

> **Reserve tempo e espaço em disco.** São cerca de **2,3 GB** baixados do site do TSE e perto de
> **40 minutos** no total, numa máquina comum. É normal o terminal parecer parado por vários
> minutos em `Preparando 2018...` e `Preparando 2022...` — esses arquivos têm mais de 4 GB quando
> descompactados.

Se preferir acompanhar etapa por etapa, dá para rodar separadamente:

```bash
python -m eleicao.pipeline baixar      # baixa os arquivos oficiais do TSE
python -m eleicao.pipeline preparar    # limpa os dados e gera as tabelas
python -m eleicao.pipeline avaliar     # testes retrospectivos (a parte mais demorada)
python -m eleicao.pipeline prever      # treina a rede final e estima 2026
python -m eleicao.pipeline verificar   # confere escala, consistência e vazamento
python -m eleicao.pipeline amostra     # regenera o recorte versionado
```

Ao final do `verificar`, as treze linhas devem aparecer com `[OK]`.

### 3. Conferir que está tudo certo (opcional)

```bash
pytest -q
```

Devem passar **19 testes** em cerca de 30 segundos. Eles usam dados **sintéticos** gerados numa
pasta temporária — nada em `dados/` é tocado.

### 4. Abrir o aplicativo

```bash
streamlit run app.py
```

---

## O que você vê no aplicativo

Na **barra lateral esquerda** ficam os controles:

| Controle | Para que serve |
|---|---|
| **Eleição** | `2026 — estimativa` ou uma eleição passada como `2022 — teste retrospectivo` |
| **Turno** | 1º ou 2º turno |
| **Nível** | Brasil, Estado ou Município |
| **Estado / Município** | aparecem conforme o nível escolhido |

As seis abas na parte de cima:

1. **Estimativa** — porcentagem por candidato na localidade escolhida
2. **Comparar localidades** — como um candidato varia entre estados ou municípios
3. **Histórico oficial** — resultados reais do TSE, de 1998 a 2022
4. **Avaliação da rede** — a tabela de erros dos testes retrospectivos
5. **Pesquisas** — registro oficial de pesquisas do TSE
6. **Fontes, método e limitações** — catálogo das fontes e as verificações automáticas

O **teste retrospectivo** é a aba mais útil para julgar a confiabilidade: mostra o que a rede teria
estimado para uma eleição passada **sem ter visto o resultado dela**, lado a lado com o resultado
oficial.

---

## Se algo der errado

| Problema | O que fazer |
|---|---|
| `streamlit: command not found` ou `não é reconhecido` | O ambiente virtual não está ativo. Rode o `activate` da sua plataforma e tente de novo. |
| A página diz **"Nenhum dado processado foi encontrado"** | A pasta `amostra/` não veio no clone. Baixe o repositório de novo, ou execute `python -m eleicao.pipeline tudo`. |
| Aparece a faixa **"Você está vendo a amostra"** | É o esperado em um clone novo, e os números são os mesmos do pipeline. Para o conjunto completo, rode `python -m eleicao.pipeline tudo`. |
| A página diz **"A rede ainda não foi treinada"** | Faltam as etapas de modelo. Execute `python -m eleicao.pipeline avaliar` e depois `prever`. |
| `Port 8501 is already in use` | Já há um app aberto. Use `streamlit run app.py --server.port 8502`. |
| Um download do TSE falhou | Baixe o arquivo pelo endereço que apareceu no terminal e salve em `dados/brutos/` **com o mesmo nome**. Depois rode `python -m eleicao.pipeline preparar`. |
| No PowerShell, `activate` é bloqueado por política de execução | Rode `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` nessa mesma janela e tente de novo. |

---

## Aviso importante

As estimativas são **experimentais**. Não são resultado oficial, não são pesquisa eleitoral e não
garantem quem vencerá. O aplicativo sempre separa, com rótulos distintos, o **Resultado oficial
(TSE)**, o **Resultado de pesquisa** e a **Estimativa da rede neural**, e não faz recomendação de
voto.
