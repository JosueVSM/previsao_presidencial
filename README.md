# Estimativa experimental de votos para Presidente com rede neural

Nome: Josué Vasconcelos Silveira Moreira
Matricula: 2517534

Nome: Plinio Rodrigues
Matricula: 2526504

Projeto de ciência de dados em Python que estima a **porcentagem de votos válidos** de cada candidato à
Presidência do Brasil **por município**, agrega as estimativas por **estado** e **Brasil** e apresenta
tudo numa interface **Streamlit** em português. A solução principal é uma **rede neural do tipo perceptron
multicamadas (MLP)** em PyTorch; métodos simples aparecem apenas como comparação.

> **Natureza das estimativas.** Os números produzidos são **estimativas experimentais**. Não são resultado
> oficial, não são pesquisa eleitoral e não garantem quem vencerá. A interface separa sempre
> **Resultado oficial (TSE)**, **Resultado de pesquisa** e **Estimativa da rede neural**, e não faz
> recomendação de voto.

---

## 1. Como executar

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py                 # já abre, usando a amostra do repositório
```

Recém-clonado, o app abre sobre `amostra/` — um recorte dos resultados versionado junto com o
código — e avisa disso na tela. Para reproduzir tudo a partir dos arquivos oficiais do TSE:

```bash
python -m eleicao.pipeline tudo      # baixar → preparar → avaliar → prever → verificar → amostra
streamlit run app.py
```

Etapas separadas: `baixar`, `preparar`, `avaliar` (validação temporal), `prever` (rede final + 2026),
`verificar` (escala, consistência, vazamento), `amostra` (regenera o recorte versionado).
O download pode passar de 1 GB (arquivos de votação por
zona trazem todos os cargos); a preparação filtra só Presidente. Em um computador comum, avaliação +
previsão levam alguns minutos em CPU.

Se algum download falhar, baixe o arquivo pelo navegador no endereço indicado e salve-o em `dados/brutos/`
com o mesmo nome.

Testes automatizados (usam **dados sintéticos** gerados numa pasta temporária, nunca em `dados/`):

```bash
pytest -q
```

## 2. Estrutura

```
app.py                       interface Streamlit
eleicao/config.py            anos, URLs, datas das eleições, hiperparâmetros
eleicao/download.py          download dos arquivos oficiais + registro das fontes
eleicao/leitura.py           leitura robusta dos CSV do TSE (latin-1, ';', layouts antigos)
eleicao/partidos.py          padronização de siglas, linhagem partidária, nomes
eleicao/preparacao.py        tabelas limpas (parquet) e fórmula dos votos válidos
eleicao/features.py          entradas da rede, sem vazamento temporal; referência simples
eleicao/modelo.py            MLP com softmax por disputa, treino, conjunto de redes
eleicao/avaliacao.py         validação temporal e previsão do ano-alvo
eleicao/agregacao.py         agregação ponderada para UF e Brasil
eleicao/pesquisas.py         registro de pesquisas do TSE e percentuais opcionais do usuário
eleicao/verificacao.py       verificações automáticas
eleicao/amostra.py           recorte versionado que faz o app abrir sem baixar nada
amostra/                     esse recorte (vai no repositório, ao contrário de dados/ e resultados/)
config/linhagem_partidos.csv tabela editável de sucessão de partidos
dados/fontes.json            catálogo das fontes (gerado no download/preparação)
dados/manual/                pesquisas_resultados.csv (opcional) e modelo do arquivo
tests/                       testes com dados sintéticos
```

## 3. Dados

Somente fontes oficiais do **Tribunal Superior Eleitoral (TSE)**, pelo Portal de Dados Abertos
(<https://dadosabertos.tse.jus.br/>), arquivos servidos pelo CDN `https://cdn.tse.jus.br/estatistica/sead/odsele/`.

| Conjunto | Endereço (padrão) | Período | Colunas utilizadas |
|---|---|---|---|
| Votação nominal por município e zona | `votacao_candidato_munzona/votacao_candidato_munzona_{ano}.zip` | 1998–2022, 1º e 2º turnos | `ANO_ELEICAO, NR_TURNO, CD_TIPO_ELEICAO, CD_CARGO, DS_CARGO, SG_UF, CD_MUNICIPIO, NM_MUNICIPIO, NR_ZONA, NR_CANDIDATO, NM_CANDIDATO, NM_URNA_CANDIDATO, SG_PARTIDO, QT_VOTOS_NOMINAIS_VALIDOS` (ou `QT_VOTOS_NOMINAIS`), `NM_TIPO_DESTINACAO_VOTOS, DS_SIT_TOT_TURNO` |
| Detalhe da apuração por município e zona | `detalhe_votacao_munzona/detalhe_votacao_munzona_{ano}.zip` | 1998–2022 | `ANO_ELEICAO, NR_TURNO, CD_CARGO, SG_UF, CD_MUNICIPIO, NR_ZONA, QT_APTOS, QT_COMPARECIMENTO` |
| Candidaturas | `consulta_cand/consulta_cand_2026.zip` | registro 2026 | `CD_CARGO, NR_CANDIDATO, NM_CANDIDATO, NM_URNA_CANDIDATO, SG_PARTIDO, NM_PARTIDO, SG_FEDERACAO, DS_COMPOSICAO_COLIGACAO, DS_SITUACAO_CANDIDATURA, DS_DETALHE_SITUACAO_CAND, DS_SIT_TOT_TURNO, ST_REELEICAO` |
| Perfil do eleitorado | `perfil_eleitorado/perfil_eleitorado_2026.zip` | cadastro 2026 | `SG_UF, CD_MUNICIPIO, NM_MUNICIPIO, QT_ELEITORES` (chamada `QT_ELEITORES_PERFIL` até 2022) |
| Pesquisas eleitorais (registro) | `pesquisa_eleitoral/pesquisa_eleitoral_2026.zip` | registros 2026 | instituto, nº de registro, datas de campo/divulgação, abrangência, entrevistados, metodologia (nomes detectados automaticamente) |

Páginas dos conjuntos: `https://dadosabertos.tse.jus.br/dataset/resultados-{ano}`,
`.../candidatos-2026`, `.../eleitorado-2026`, `.../pesquisas-eleitorais-2026`. Resultados consolidados também
podem ser conferidos em <https://www.tse.jus.br/eleicoes/resultados-eleicoes>.

**Registro de cada fonte.** O arquivo `dados/fontes.json` guarda, para cada conjunto: fonte, endereço,
página do conjunto, **data de acesso** (momento do download), período coberto, **colunas efetivamente
utilizadas**, número de linhas usadas, tamanho e hash SHA-256 do arquivo. O app exibe esse catálogo.

**O que não existe nos dados oficiais.** O conjunto de pesquisas do TSE traz só o **registro** das pesquisas
(sem percentuais). Nada é inventado para preencher essa lacuna: ver seção 7.

## 4. Preparação dos dados

- **Identificação:** eleição (`ANO_ELEICAO`, só eleições ordinárias), turno (`NR_TURNO`), cargo
  (`CD_CARGO = 1`, Presidente), candidato, partido, município (`CD_MUNICIPIO`, código TSE) e estado
  (`SG_UF`; `ZZ` = exterior). Votos por zona são somados por município.
- **Arquivos do TSE:** o `.zip` traz um CSV por UF e, em geral, um `_BRASIL` com tudo; quando ele existe,
  só ele é lido (evita contagem em dobro). Arquivos antigos sem cabeçalho são lidos pelo layout do LEIAME.
  O cabeçalho é reconhecido pela forma dos campos (todos identificadores em maiúsculas), e não por um nome
  fixo: a votação usa `ANO_ELEICAO`, mas o perfil do eleitorado e as pesquisas usam `AA_ELEICAO`.
- **Eleitorado:** o arquivo de perfil tem uma linha por estrato (zona, gênero, faixa etária, escolaridade,
  raça/cor...) — 11 milhões de linhas em 2026. Cada bloco lido já é somado por município, de modo que a
  memória não cresce com o tamanho do arquivo; a soma das somas é idêntica à soma de tudo.
- **Votos válidos:** para cada disputa (eleição × turno × município)

  `pct_validos(c) = 100 × votos_válidos(c) / Σ_k votos_válidos(k)`

  somando todos os candidatos da disputa; brancos e nulos ficam fora (definição do TSE). Usa-se
  `QT_VOTOS_NOMINAIS_VALIDOS` quando existe (já exclui votos em candidaturas anuladas); senão,
  `QT_VOTOS_NOMINAIS`, descartando destinações "anulado/nulo".
- **Municípios:** o código do TSE é estável entre eleições; nomes vêm do arquivo mais recente. Municípios
  sem votação na eleição anterior (criados depois) recebem os valores do estado e um indicador próprio.
- **Candidatos entre eleições:** identificados pelo **nome civil normalizado** (sem acentos, maiúsculas),
  que é estável; número e sequencial mudam a cada eleição.
- **Partidos:** siglas normalizadas e mapeadas para uma **linhagem** (`config/linhagem_partidos.csv`):
  PMDB→MDB, PFL→DEM→UNIÃO, PSL→UNIÃO, PR→PL, PTN→PODE, PRB→REPUBLICANOS, PPS→CIDADANIA, PTB/Patriota→PRD
  etc. Assim, o desempenho anterior de um partido é encontrado mesmo após mudança de nome ou fusão. A tabela
  é editável e deve ser revisada a cada nova fusão.
- **Mudanças de partido e ausências:** um candidato que trocou de partido carrega o próprio histórico pelas
  variáveis do candidato; o novo partido carrega o histórico do partido. Quem não concorreu antes recebe
  zero nas variáveis de desempenho e um indicador "não concorreu", para a rede distinguir "zero votos" de
  "sem histórico".
- **Candidatos de 2026:** entram os registros com situação **APTO** (inclui candidaturas sub judice que
  constam da urna). Inaptos aparecem no app como não incluídos.

### Sem vazamento temporal

Para estimar a eleição do ano *T*, as entradas usam **apenas**: resultados da eleição anterior
(*T − 4*) e de anteriores, a lista de candidatos de *T*, o eleitorado apto de *T* (cadastro fechado
meses antes) e, opcionalmente, pesquisas divulgadas **antes** de *T*. Os votos de *T* são usados somente
como alvo de treino/avaliação. O resultado oficial de 2026 **nunca** é usado como entrada: a carga de
dados descarta qualquer linha de 2026 antes de montar as entradas. Testes automáticos alteram os votos da
eleição-alvo e confirmam que nenhuma entrada muda.

## 5. Rede neural

**Representação numérica.** A rede não recebe nomes nem códigos de partido (isso não generalizaria para
candidatos novos). Cada linha (município, candidato) é descrita por:

| Grupo | Variáveis |
|---|---|
| Partido (linhagem) na eleição anterior, 1º turno | % no município, na UF e no Brasil; log da % municipal; concorreu (0/1) |
| Partido na eleição anterior, 2º turno | % no município; esteve no 2º turno (0/1) |
| Próprio candidato | % na eleição anterior no município e no Brasil; concorreu na anterior; já concorreu alguma vez; foi eleito na anterior (reeleição) |
| Contexto | partido do eleito na eleição anterior (0/1) |
| Pesquisas (só se habilitadas) | média nacional na janela pré-eleitoral; log; candidato fora das pesquisas (0/1) |
| Município (iguais para todos os candidatos da disputa) | 2º turno (0/1); nº de candidatos; log do eleitorado; comparecimento e fragmentação (nº efetivo de candidatos) na anterior; sem histórico (0/1); região (one-hot N, NE, CO, SE, S, exterior) |

Todas as entradas são padronizadas (z-score) com média e desvio calculados **só no treino**.

**Arquitetura.** MLP compartilhada entre candidatos:
`entradas → 64 (ReLU, dropout 0,1) → 32 (ReLU, dropout 0,1) → 1 pontuação`.
As pontuações dos candidatos de um município passam por um **softmax restrito àquela disputa**:
`p_c = exp(s_c) / Σ_k exp(s_k)`. Logo, toda estimativa fica entre 0 e 100% e **soma exatamente 100%** em
cada município, para qualquer número de candidatos (posições de preenchimento são mascaradas).

**Função de perda.** Entropia cruzada entre a distribuição oficial `y` e a prevista `p`, ponderada por
município com `w = √(votos válidos)` (compromisso entre a verossimilhança multinomial exata, dominada pelas
capitais, e o peso igual por município).

**Treinamento.** AdamW (taxa 1e-3, decaimento de pesos 1e-3), lotes de 256 municípios, corte de gradiente 5,
até 200 épocas. **Critério de parada:** validação temporal interna — cada uma das três últimas eleições de
treino é usada como validação (eleição inteira, treinando só com as anteriores a ela); as curvas de perda
são promediadas e escolhe-se o número de épocas de menor perda média (paciência de 15 épocas). Depois, cada
rede é re-treinada do zero com todas as eleições de treino por esse número de épocas. Usa-se um **conjunto de
5 redes** (sementes diferentes); a estimativa é a média. O desvio entre redes mede apenas a instabilidade
do treino e **não** é intervalo de confiança.

Hiperparâmetros em `eleicao/config.py → HIPERPARAMETROS`. Eles foram fixados a priori (não otimizados no
conjunto de teste); com tão poucas eleições, uma busca extensa de hiperparâmetros tenderia a sobreajustar.

## 6. Avaliação

**Validação temporal (teste retrospectivo).** Para cada eleição de teste *T* ∈ {2010, 2014, 2018, 2022}, a
rede é treinada só com eleições-alvo anteriores (desde 2002, cada uma usando a sua anterior como entrada) e
avaliada em *T*, que fica **inteira** fora do treino — todos os municípios e os dois turnos. Municípios de uma
mesma eleição nunca se dividem entre treino e teste.

**Métricas** (em pontos percentuais de votos válidos), por eleição e turno: erro absoluto médio (EAM) e raiz
do erro quadrático médio (REQM) por município e candidato; EAM ponderado pelos votos; % de municípios em
que o mais votado foi acertado; e erro do **agregado nacional** frente ao resultado oficial.

**Comparações (complementares):** *persistência partidária* e *divisão igual*. A persistência
repete, no mesmo município, a disputa anterior **do mesmo tipo de turno**: para um 1º turno, o 1º turno
anterior (do partido pela linhagem ou do próprio candidato, o que for maior); para um 2º turno, o
**2º turno anterior** sempre que o partido esteve nele, recaindo no 1º turno só quando não há 2º turno
a repetir (caso do PSL em 2018). Medir um 2º turno contra as fatias de um 1º turno anterior
enfraqueceria a referência artificialmente e faria a rede parecer melhor do que é. Todas as colunas
vêm de T−4: a referência também não enxerga o ano previsto.

**Limitação central:** há milhares de municípios, mas **pouquíssimas eleições independentes** (seis alvos de
treino, quatro testes). Tudo o que é comum a uma eleição inteira — força nacional de um candidato novo, ondas
de opinião, troca de partido de um candidato forte — é observado poucas vezes. Por isso a rede tende a captar
melhor a **geografia relativa** do voto do que o **nível nacional**, e o desempenho pode variar muito de uma
eleição para outra. As métricas mostram o tamanho dos erros passados; não demonstram que a rede aprendeu
padrões confiáveis para 2026.

## 7. Pesquisas eleitorais

- O TSE publica o **registro** (instituto, nº de registro, datas, amostra, metodologia), mas não os
  percentuais. O app mostra o registro de 2026 como contexto.
- Percentuais só entram se você preencher `dados/manual/pesquisas_resultados.csv` (modelo em
  `pesquisas_resultados_MODELO.csv`): `ano, turno, nr_registro_tse, instituto, data_inicio, data_fim,
  abrangencia (BR ou UF), n_entrevistados, margem_erro_pp, candidato (nome de urna), pct, base`.
  Eles aparecem rotulados como **Resultado de pesquisa**.
- **Na rede**, pesquisas só são usadas se houver percentuais nacionais para **todas** as eleições de treino e
  para a eleição estimada (senão a rede teria de dar peso a uma variável que nunca viu). Entram como uma
  característica do candidato: média das pesquisas **nacionais** com campo encerrado nos 30 dias anteriores à
  eleição, convertida para base de votos válidos e ponderada por √entrevistados.
- **Inferências não possíveis:** uma pesquisa nacional ou estadual não é resultado observado de nenhum
  município; ela não é usada como alvo nem atribuída a municípios. Pesquisas estaduais não entram na rede.

## 8. Agregação nacional e estadual

`p_BR(c) = Σ_m W_m · p_m(c) / Σ_m W_m`, com `W_m = eleitorado apto do ano × (votos válidos ÷ aptos) do
município na eleição anterior` — os **votos válidos esperados**. A soma é feita **por candidato**
(nome civil normalizado, que é estável); nome de urna e sigla são apenas rótulos, escolhidos depois
pelo maior peso agregado. Se entrassem na chave do agrupamento, uma grafia diferente em alguns
municípios partiria a mesma pessoa em duas linhas e o gráfico mostraria uma barra menor que a real. Justificativa: a % nacional oficial é
`Σ votos(c) / Σ válidos`, que é exatamente a média das % municipais ponderada pelos votos válidos de cada
município (um teste automático confirma essa identidade). Como os válidos do ano não são
conhecidos antes da eleição, usa-se a estimativa acima. **A média simples entre municípios não é usada.**

## 9. Verificações

`python -m eleicao.pipeline verificar` gera `resultados/relatorio_verificacoes.json` (também exibido no app):
chave única por município/candidato, votos não negativos, % oficiais somando 100%, contagem de municípios,
totais nacionais recalculados (para conferir com o site do TSE), ausência de resultado de 2026 nas entradas,
treino sempre anterior ao teste, estimativas em [0, 1] somando 100% por município e no agregado nacional.

## 10. Limitações

- Poucas eleições independentes (seção 6).
- Candidatos e partidos sem histórico tendem a ser **subestimados**; fusões e trocas de partido são tratadas
  de forma aproximada pela tabela de linhagem.
- Campanha, alianças, debates, economia, comparecimento e eventos recentes não estão nos dados.
- Sem percentuais de pesquisa fornecidos, a rede não tem nenhuma informação sobre o cenário de 2026 além da
  lista de candidatos e do passado.
- O 2º turno de 2026 só pode ser estimado depois que o TSE registrar quem o disputa.
- Mudanças de formato nos arquivos do TSE podem exigir ajustes em `eleicao/leitura.py`. O TSE renomeia
  colunas entre conjuntos e entre anos (`AA_ELEICAO`/`ANO_ELEICAO`, `QT_ELEITORES`/`QT_ELEITORES_PERFIL`,
  `QT_ENTREVISTADO`/`QT_ENTREVISTADOS`); nomes novos entram na tabela `ALIASES`.
- Em 2026, `DS_SITUACAO_CANDIDATURA` ainda vem como `#NE` para todos os candidatos a Presidente (o registro
  não foi julgado). Nesse caso entram todos os registrados que não estejam explicitamente `INAPTO`, e o app
  informa a situação como ela consta do arquivo — nada é preenchido por suposição.

## 11. Como atualizar

- **Novos dados oficiais (candidaturas, eleitorado, pesquisas):** `python -m eleicao.pipeline baixar --forcar-download`
  e depois `python -m eleicao.pipeline preparar prever verificar` (uma etapa por comando, ou `tudo`).
- **Após a totalização do 1º turno:** quando o cadastro de candidaturas indicar quem vai ao 2º turno
  (`DS_SIT_TOT_TURNO = "2º TURNO"`), rode de novo `baixar --forcar-download`, `preparar`, `prever`; o app passa a
  mostrar o 2º turno. As entradas continuam vindo só de eleições anteriores.
- **Depois que o resultado oficial de 2026 for publicado:** ele pode ser usado para **avaliar** a estimativa
  (inclua 2026 em `ANOS_HISTORICOS` e em `ANOS_TESTE_BACKTEST`) e, numa eleição futura, como histórico
  (ajuste `ANO_ALVO`, `DATAS_ELEICAO` e as URLs). Nunca como entrada para estimar a própria eleição de 2026.
- **Novas fusões de partidos:** acrescente linhas em `config/linhagem_partidos.csv`.
