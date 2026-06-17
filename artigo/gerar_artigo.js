// Gera o artigo científico do OscaBet em .docx (normas ABNT).
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, HeadingLevel, BorderStyle, WidthType, ShadingType,
  ImageRun, Header, Footer, PageNumber, LevelFormat
} = require("docx");

const ROOT = path.resolve(__dirname, "..");
const EVAL = path.join(ROOT, "evaluation");

// ── Constantes de layout (DXA: 1 cm = 567) ───────────────────────────────────
const CM = 567;
const CONTENT_W = 11906 - Math.round(3 * CM) - Math.round(2 * CM); // A4 - 3cm esq - 2cm dir
const FONT = "Times New Roman";
const SZ = 24;       // 12pt (half-points)
const SZ_SMALL = 20; // 10pt
const LINE_15 = 360; // 1,5 entrelinhas
const INDENT = Math.round(1.25 * CM); // 709

// ── Helpers ──────────────────────────────────────────────────────────────────
// Parágrafo de corpo (justificado, recuo 1ª linha, 1,5)
function P(text_or_runs, opts = {}) {
  const runs = Array.isArray(text_or_runs)
    ? text_or_runs
    : [new TextRun({ text: text_or_runs, size: SZ, font: FONT })];
  return new Paragraph({
    alignment: opts.align || AlignmentType.JUSTIFIED,
    spacing: { line: LINE_15, after: opts.after ?? 0, before: opts.before ?? 0 },
    indent: opts.noIndent ? undefined : { firstLine: INDENT },
    children: runs,
  });
}
// Run em negrito / itálico inline
const b = (t) => new TextRun({ text: t, bold: true, size: SZ, font: FONT });
const t = (t) => new TextRun({ text: t, size: SZ, font: FONT });
const it = (t) => new TextRun({ text: t, italics: true, size: SZ, font: FONT });

// Título de seção primária (1 INTRODUÇÃO) — caixa alta, negrito
function H1(num, text) {
  return new Paragraph({
    spacing: { before: 360, after: 200, line: LINE_15 },
    outlineLevel: 0,
    children: [new TextRun({ text: num ? `${num}  ${text.toUpperCase()}` : text.toUpperCase(), bold: true, size: SZ, font: FONT, color: "000000" })],
  });
}
function H2(num, text) {
  return new Paragraph({
    spacing: { before: 260, after: 160, line: LINE_15 },
    outlineLevel: 1,
    children: [new TextRun({ text: `${num}  ${text}`, bold: true, size: SZ, font: FONT, color: "000000" })],
  });
}

// Tabela ABNT (bordas só horizontais; título acima; fonte abaixo)
function abntTable(num, title, headers, rows, source, widths) {
  const hb = { style: BorderStyle.SINGLE, size: 8, color: "000000" };
  const noB = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
  const tot = widths.reduce((a, c) => a + c, 0);
  const mkCell = (txt, isHeader, top, bottom, alignRight, w) => new TableCell({
    width: { size: w, type: WidthType.DXA },
    borders: { top: top ? hb : noB, bottom: bottom ? hb : noB, left: noB, right: noB },
    margins: { top: 40, bottom: 40, left: 80, right: 80 },
    children: [new Paragraph({
      alignment: alignRight ? AlignmentType.CENTER : AlignmentType.LEFT,
      spacing: { line: 240, after: 0 },
      children: [new TextRun({ text: txt, bold: !!isHeader, size: SZ_SMALL, font: FONT })],
    })],
  });
  const headRow = new TableRow({
    children: headers.map((h, i) => mkCell(h, true, true, true, i > 0, widths[i])),
  });
  const bodyRows = rows.map((r, ri) => new TableRow({
    children: r.map((cell, i) => mkCell(String(cell), false, false, ri === rows.length - 1, i > 0, widths[i])),
  }));
  return [
    new Paragraph({ alignment: AlignmentType.LEFT, spacing: { before: 200, after: 40, line: 240 },
      children: [new TextRun({ text: `Tabela ${num} – ${title}`, size: SZ_SMALL, font: FONT })] }),
    new Table({ width: { size: tot, type: WidthType.DXA }, columnWidths: widths, alignment: AlignmentType.CENTER, rows: [headRow, ...bodyRows] }),
    new Paragraph({ alignment: AlignmentType.LEFT, spacing: { before: 40, after: 200, line: 240 },
      children: [new TextRun({ text: `Fonte: ${source}`, size: SZ_SMALL, font: FONT })] }),
  ];
}

// Figura ABNT (legenda acima, imagem centralizada, fonte abaixo)
function abntFigure(num, caption, file, source, w, h) {
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 220, after: 60, line: 240 },
      children: [new TextRun({ text: `Figura ${num} – ${caption}`, size: SZ_SMALL, font: FONT })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40 },
      children: [new ImageRun({ type: "png", data: fs.readFileSync(path.join(EVAL, file)),
        transformation: { width: w, height: h },
        altText: { title: caption, description: caption, name: `fig${num}` } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 220, line: 240 },
      children: [new TextRun({ text: `Fonte: ${source}`, size: SZ_SMALL, font: FONT })] }),
  ];
}

// Referência (ABNT, espaço simples, alinhada à esquerda)
function ref(runs) {
  return new Paragraph({ alignment: AlignmentType.LEFT, spacing: { line: 240, after: 120 },
    children: runs.map((r) => typeof r === "string" ? t(r) : r) });
}

// ── Conteúdo ──────────────────────────────────────────────────────────────────
const children = [];

// Título
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80, line: LINE_15 },
  children: [new TextRun({ text: "OscaBet: Agente de Inteligência Artificial para Previsão de Partidas de Futebol com Aprendizado de Máquina Multi-Alvo (Rede Neural e Gradient Boosting) e Interação em Linguagem Natural", bold: true, size: 28, font: FONT })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40, line: 240 }, children: [new TextRun({ text: "Ian Marchi", size: SZ, font: FONT })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240, line: 240 }, children: [new TextRun({ text: "Trabalho da disciplina de Inteligência Artificial Agêntica — 1º Período", italics: true, size: SZ_SMALL, font: FONT })] }));

// Resumo
children.push(new Paragraph({ spacing: { after: 80, line: 240 }, children: [b("RESUMO")] }));
children.push(P([t("A previsão de resultados de partidas de futebol é um problema reconhecidamente difícil, em razão da elevada aleatoriedade do esporte. Este trabalho apresenta o OscaBet, um agente de inteligência artificial capaz de responder, em linguagem natural, a perguntas sobre futebol e de gerar previsões probabilísticas para três mercados: resultado (vitória do mandante, empate ou vitória do visitante), total de cartões e total de escanteios. O sistema combina uma rede neural multi-alvo, treinada por aprendizado supervisionado sobre aproximadamente 17,8 mil partidas reais de dez competições coletadas do portal Sofascore, a um modelo de linguagem de grande porte que decide autonomamente quais ferramentas acionar. A metodologia adotou divisão temporal estrita para evitar vazamento de informação, atributos de janela móvel e avaliação por meio de um backtest de 6.856 partidas. Os resultados mostram que o modelo atinge o teto realista de previsibilidade do problema — acurácia de 0,52 no resultado, 0,62 em cartões e 0,57 em escanteios —, igualando ou superando modelos tabulares de referência, com probabilidades bem calibradas. Desenvolveu-se ainda um segundo modelo, baseado em árvores de decisão impulsionadas por gradiente (CatBoost), que explora nativamente atributos categóricos — liga e equipes — e superou a rede neural nos três mercados, com destaque para os escanteios, sendo adotado como modelo padrão do sistema. Conclui-se que, embora o sistema supere consistentemente os classificadores triviais e demonstre conteúdo informativo monetizável, os mercados de over/under apresentam baixo teto de previsibilidade, o que evidencia a importância de comunicar a incerteza de forma honesta.")], { noIndent: true }));
children.push(new Paragraph({ spacing: { before: 120, after: 240, line: 240 }, children: [b("Palavras-chave: "), t("Inteligência artificial. Redes neurais artificiais. Previsão esportiva. Agentes inteligentes. Aprendizado de máquina.")] }));

// Abstract
children.push(new Paragraph({ spacing: { after: 80, line: 240 }, children: [b("ABSTRACT")] }));
children.push(P([t("Predicting football match outcomes is a notoriously hard problem due to the sport's high randomness. This work presents OscaBet, an artificial intelligence agent able to answer football questions in natural language and to generate probabilistic predictions for three markets: match result (home win, draw or away win), total cards and total corners. The system couples a multi-target neural network — trained by supervised learning on roughly 17.8 thousand real matches from ten competitions collected from the Sofascore portal — with a large language model that autonomously decides which tools to call. The methodology adopted a strict temporal split to avoid data leakage, rolling-window features and evaluation through a 6,856-match backtest. Results show the model reaches the realistic predictability ceiling of the problem (accuracy of 0.52 for result, 0.62 for cards and 0.57 for corners), matching or surpassing strong tabular baselines, with well-calibrated probabilities. A second model, based on gradient-boosted decision trees (CatBoost) that natively handles categorical features — league and teams — outperformed the neural network across all three markets, especially for corners, and was adopted as the system's default. We conclude that, although the system consistently beats trivial classifiers and shows monetizable informational content, over/under markets have a low predictability ceiling, highlighting the importance of honestly communicating uncertainty.")], { noIndent: true }));
children.push(new Paragraph({ spacing: { before: 120, after: 240, line: 240 }, children: [b("Keywords: "), t("Artificial intelligence. Artificial neural networks. Sports forecasting. Intelligent agents. Machine learning.")] }));

// 1 INTRODUÇÃO
children.push(H1("1", "Introdução"));
children.push(P("O futebol é o esporte mais popular do mundo e movimenta um mercado bilionário de apostas, no qual a antecipação de resultados é objeto de interesse tanto comercial quanto acadêmico. Prever o desfecho de uma partida, contudo, é uma tarefa intrinsecamente incerta: a baixa quantidade de gols, a influência de fatores não observáveis e a própria aleatoriedade do jogo impõem um teto à previsibilidade que mesmo modelos sofisticados não conseguem ultrapassar."));
children.push(P("Paralelamente, os avanços recentes em modelos de linguagem de grande porte (LLMs) viabilizaram a construção de agentes inteligentes capazes de raciocinar sobre uma tarefa e acionar ferramentas externas de forma autônoma. Esse paradigma — conhecido como uso de ferramentas (tool use) — permite combinar a fluência conversacional dos LLMs com a precisão de componentes especializados, como modelos preditivos quantitativos."));
children.push(P("Este trabalho descreve o desenvolvimento do OscaBet, um agente de inteligência artificial que une essas duas vertentes. O sistema responde, em português, a perguntas sobre futebol e, quando solicitado, aciona uma rede neural multi-alvo para prever três aspectos de uma partida: o resultado (vitória do mandante, empate ou vitória do visitante), o total de cartões e o total de escanteios."));
children.push(P([b("Objetivo geral. "), t("Construir e avaliar um agente de IA que integre uma rede neural de previsão esportiva a um modelo de linguagem, disponibilizando o resultado por meio de uma aplicação web.")]));
children.push(P([b("Objetivos específicos: "), t("(i) coletar e tratar uma base de dados real de partidas de futebol; (ii) projetar, treinar e otimizar uma rede neural multi-alvo; (iii) integrar o modelo a um agente conversacional capaz de decidir quando utilizá-lo; (iv) avaliar o desempenho preditivo de forma rigorosa e honesta, contextualizando-o frente aos limites teóricos do problema; e (v) entregar o sistema em uma interface utilizável.")]));
children.push(P("A principal contribuição não está apenas no artefato produzido, mas no rigor metodológico e na transparência da avaliação: demonstra-se que o modelo opera no teto estatístico do problema, discute-se por que metas de acurácia popularmente associadas a apostas são inatingíveis com dados reais e evidencia-se a importância de comunicar a incerteza em vez de simular falsa confiança."));

// 2 FUNDAMENTAÇÃO TEÓRICA
children.push(H1("2", "Fundamentação Teórica"));
children.push(H2("2.1", "Previsão de resultados esportivos"));
children.push(P("A modelagem quantitativa de partidas de futebol tem origem nos trabalhos clássicos de Dixon e Coles (1997), que propuseram um modelo baseado na distribuição de Poisson para o número de gols, incorporando forças de ataque e defesa de cada equipe. Estudos posteriores aplicaram redes bayesianas, modelos de Elo e técnicas de aprendizado de máquina. Hubáček, Šourek e Železný (2019) demonstraram que árvores de decisão impulsionadas por gradiente (gradient boosting) sobre atributos relacionais são competitivas para a previsão de resultados, atingindo acurácias da ordem de 50% a 53% para o desfecho 1-X-2 — patamar consistente com o consenso de que a previsibilidade do resultado é limitada."));
children.push(H2("2.2", "Redes neurais artificiais e aprendizado multi-tarefa"));
children.push(P("Redes neurais artificiais são aproximadores universais de funções, capazes de aprender relações não lineares a partir de exemplos (GOODFELLOW; BENGIO; COURVILLE, 2016). No aprendizado multi-tarefa, um mesmo modelo aprende simultaneamente várias tarefas correlacionadas por meio de uma representação compartilhada, o que pode melhorar a generalização e reduzir o número de parâmetros. No presente trabalho, adota-se uma arquitetura com um tronco (backbone) compartilhado e três cabeças de classificação independentes, uma para cada mercado previsto."));
children.push(H2("2.3", "Árvores de decisão impulsionadas por gradiente e CatBoost"));
children.push(P("As árvores de decisão impulsionadas por gradiente (gradient boosting) constroem um conjunto (ensemble) de árvores de forma sequencial, em que cada nova árvore corrige os erros residuais das anteriores. Esse paradigma figura entre os mais eficazes para dados tabulares, em que frequentemente iguala ou supera redes neurais. A biblioteca CatBoost (PROKHORENKOVA et al., 2018) aprimora a técnica em dois aspectos relevantes para este trabalho: um tratamento nativo e estatisticamente fundamentado de atributos categóricos de alta cardinalidade — dispensando a codificação one-hot — e o uso de árvores simétricas (oblivious trees), que reduzem o sobreajuste e aceleram a inferência. Tais propriedades são particularmente adequadas a um domínio com muitas equipes e ligas distintas e com grande volume de exemplos."));
children.push(H2("2.4", "Agentes inteligentes e uso de ferramentas por LLMs"));
children.push(P("Um agente inteligente percebe seu ambiente e age para atingir objetivos. Quando construído sobre um modelo de linguagem, o agente pode decidir, a cada passo, se responde diretamente ou se invoca ferramentas externas — funções com interface bem definida — cujos resultados são reincorporados ao raciocínio. Esse laço agêntico (agentic loop) permite que o LLM delegue cálculos quantitativos a componentes especializados, mantendo o controle do diálogo em linguagem natural."));
children.push(H2("2.5", "Calibração e avaliação de modelos probabilísticos"));
children.push(P("Além da acurácia, a qualidade de um modelo probabilístico depende de sua calibração: a frequência empírica dos eventos deve corresponder às probabilidades previstas. Um modelo bem calibrado que afirma 60% de chance acerta, no longo prazo, cerca de 60% das vezes. Métricas como a perda logarítmica (log-loss) e o escore de Brier (BRIER, 1950) quantificam conjuntamente acerto e calibração, sendo mais informativas que a acurácia isolada em problemas de alta incerteza."));

// 3 METODOLOGIA
children.push(H1("3", "Materiais e Métodos"));
children.push(P("O desenvolvimento seguiu uma abordagem experimental e incremental. Esta seção descreve a base de dados, a engenharia de atributos, a definição dos alvos, a arquitetura e o treinamento da rede, o agente conversacional, a aplicação web e o protocolo de avaliação."));
children.push(H2("3.1", "Coleta de dados"));
children.push(P("Os dados foram coletados do portal Sofascore por meio de um coletor (scraper) próprio, respeitando limites de requisição. A base reúne aproximadamente 17,8 mil partidas reais de dez competições — Brasileirão Série A e B, Copa do Brasil, Libertadores, Premier League, La Liga, Serie A, Bundesliga, Ligue 1 e Champions League —, abrangendo as temporadas de 2020 a 2026. Para cada partida registram-se o placar e cerca de quarenta estatísticas detalhadas, incluindo posse de bola, finalizações, escanteios, cartões, passes, desarmes e gols esperados (xG)."));
children.push(H2("3.2", "Engenharia de atributos"));
children.push(P("A partir das estatísticas brutas, computaram-se atributos de janela móvel (rolling window) considerando os dez jogos anteriores de cada equipe, garantindo a ausência de vazamento temporal: nenhum atributo de uma partida utiliza informação posterior a ela. Após o tratamento, o conjunto final reúne 103 atributos numéricos por partida, normalizados ao intervalo [0,1] por competição. Entre eles destacam-se o aproveitamento recente, médias de gols, finalizações, posse, escanteios, cartões, duelos, o histórico de confrontos diretos (H2H), a posição na tabela e, notadamente, as features derivadas do xG."));
children.push(H2("3.3", "Definição dos alvos"));
children.push(P("Foram definidos três alvos de classificação: (i) resultado, com três classes — vitória do mandante (H), empate (D) e vitória do visitante (A); (ii) total de cartões, em formato over/under na linha de 4,5; e (iii) total de escanteios, em formato over/under na linha de 9,5. As linhas adotadas refletem as efetivamente utilizadas na construção dos rótulos, mantendo equilíbrio razoável entre as classes."));
children.push(H2("3.4", "Arquitetura e treinamento da rede"));
children.push(P("A rede neural multi-alvo, implementada em PyTorch (PASZKE et al., 2019), possui um tronco compartilhado com camadas densas de 256, 128 e 64 unidades, com normalização em lote, ativação ReLU e regularização por dropout. Sobre o tronco, três cabeças independentes produzem as distribuições de probabilidade de cada mercado. O modelo conta com cerca de 72 mil parâmetros treináveis."));
children.push(P("Adotou-se divisão temporal: partidas anteriores a março de 2024 compõem o conjunto de treino (9.319 jogos) e as posteriores formam a validação (6.856 jogos). A receita de treino foi otimizada experimentalmente; as escolhas mais relevantes foram a remoção de pesos de classe e de suavização de rótulos (label smoothing), o uso de forte regularização e a seleção do melhor estado do modelo pela maior acurácia média de validação. O modelo final é um ensemble de cinco sementes (seeds): a média de suas probabilidades aumenta a estabilidade e a calibração."));
children.push(H2("3.5", "Modelo alternativo de gradient boosting (CatBoost)"));
children.push(P("Em razão do grande volume de dados e da natureza tabular do problema, implementou-se um segundo modelo com a biblioteca CatBoost, a fim de compará-lo à rede neural. Para cada um dos três mercados treinou-se um classificador independente (CatBoostClassifier), sob a mesma divisão temporal e sem pesos de classe, garantindo comparabilidade direta. Avaliaram-se duas variantes: uma utilizando apenas os 103 atributos numéricos e outra acrescentando atributos categóricos nativos — liga, equipe mandante e equipe visitante. Esta última, que dispensa codificação one-hot, mostrou-se superior e foi a adotada. Os principais hiperparâmetros foram 1.500 iterações, taxa de aprendizado de 0,03, profundidade 6 e regularização L2, com parada antecipada (early stopping) guiada pela acurácia de validação. Toda a inferência do sistema é abstraída por uma camada comum, de modo que rede neural e CatBoost são intercambiáveis sem alterar o restante da aplicação."));
children.push(H2("3.6", "Tratamento de empates"));
children.push(P("Por ser a classe de menor frequência (cerca de 25%), o empate quase nunca é o desfecho isoladamente mais provável; um classificador que maximiza a acurácia tende, portanto, a nunca prevê-lo. Em vez de forçar a previsão de empates no treino — o que, como mostrado adiante, reduz a acurácia —, implementou-se uma camada de decisão sobre as probabilidades: quando o empate está a uma margem reduzida do favorito, o palpite passa a ser empate e a partida é sinalizada como equilibrada, comunicando a incerteza ao usuário."));
children.push(H2("3.7", "Agente conversacional"));
children.push(P("O agente é orquestrado por um modelo de linguagem (gpt-oss:120b, via Ollama Cloud) que dispõe de sete ferramentas: consulta de estatísticas de um time, histórico de confrontos, tabela da competição, agenda de jogos, acionamento da rede neural de previsão, busca de apostas de valor e previsões da Copa do Mundo. A cada pergunta, o modelo decide autonomamente quais ferramentas chamar, executa-as e integra os resultados em uma resposta fluida em português."));
children.push(H2("3.8", "Aplicação web e atualização automática"));
children.push(P("A interface foi construída com Flask no backend e React no frontend, oferecendo uma aba de conversa e um painel com simulador de confrontos e previsões da rodada. Um pipeline de atualização automática coleta incrementalmente as partidas novas, reprocessa os atributos e retreina o modelo, podendo ser agendado semanalmente de forma multiplataforma."));
children.push(H2("3.9", "Protocolo de avaliação"));
children.push(P("A avaliação principal consistiu em um backtest temporal sobre as 6.856 partidas de validação: cada jogo foi previsto utilizando apenas informações anteriores a ele, simulando a previsão da próxima rodada. O desempenho foi comparado a três baselines — palpite aleatório, classe majoritária e modelos tabulares fortes (regressão logística e gradient boosting) — e complementado por análise de matrizes de confusão, curvas de calibração, estabilidade temporal e uma simulação de retorno de apostas baseada em valor esperado."));

// 4 RESULTADOS
children.push(H1("4", "Resultados e Discussão"));
children.push(H2("4.1", "Teto de previsibilidade do problema"));
children.push(P("Antes de otimizar a rede, mediu-se o teto de previsibilidade dos dados com modelos tabulares de referência, sob a mesma divisão temporal. A Tabela 1 resume os resultados e revela um achado central: existe sinal aprendível (todos superam a classe majoritária), mas o teto realista situa-se em torno de 0,51 no resultado, 0,62 em cartões e 0,56 em escanteios. Metas de 0,70 para mercados over/under, frequentemente associadas a apostas, são inatingíveis com dados reais, pois tais mercados se aproximam de um lançamento de moeda."));
children.push(...abntTable("1", "Teto de acurácia por mercado (modelos de referência, conjunto de validação)",
  ["Mercado", "Classe maj.", "Reg. logística", "Gradient boosting"],
  [["Resultado", "0,457", "0,505", "0,512"],
   ["Cartões (O/U 4,5)", "0,587", "0,617", "0,603"],
   ["Escanteios (O/U 9,5)", "0,521", "0,557", "0,557"]],
  "elaborado pelo autor (2026).", [3000, 2000, 2035, 2036]));
children.push(H2("4.2", "Otimização do treinamento"));
children.push(P("Partindo de um baseline em que a acurácia de cartões ficava abaixo do classificador trivial (efeito da combinação de pesos de classe e label smoothing), três configurações foram testadas. A Tabela 2 mostra que a remoção desses mecanismos, somada a regularização mais forte e à seleção de estado pela acurácia, foi a alavanca mais relevante: a configuração B elevou a média de validação de 0,5616 para 0,5656, levando a rede a igualar ou superar o teto dos modelos tabulares."));
children.push(...abntTable("2", "Configurações de treino e acurácia de validação por mercado",
  ["Configuração", "Resultado", "Cartões", "Escanteios", "Média"],
  [["A — sem pesos/label smoothing", "0,5174", "0,6107", "0,5567", "0,5616"],
   ["B — A + regularização forte", "0,5191", "0,6176", "0,5602", "0,5656"],
   ["C — reponderação de perdas", "0,5191", "0,6174", "0,5588", "0,5651"]],
  "elaborado pelo autor (2026).", [3035, 1509, 1509, 1509, 1509]));
children.push(H2("4.3", "Contribuição do xG e ensemble"));
children.push(P("Uma auditoria das estatísticas brutas revelou que o xG — uma das variáveis mais preditivas do futebol — vinha sendo descartado por um limiar de valores ausentes. Sua reinclusão, validada em múltiplas sementes, elevou a média de validação em cerca de 0,25 ponto percentual, sobretudo em escanteios e cartões. Importante destacar o contraste metodológico: a simples recombinação de atributos já existentes (diferenças e somas) não beneficiou a rede neural — que já aprende tais interações internamente —, ao passo que a adição de uma estatística bruta e independente, como o xG, produziu ganho real. O modelo final consolida essas escolhas em um ensemble de cinco sementes."));
children.push(H2("4.4", "Desempenho no backtest"));
children.push(P("A Tabela 3 apresenta as métricas do ensemble no backtest de 6.856 partidas reais. Em todos os mercados o modelo supera consistentemente os baselines triviais, confirmando que extrai sinal genuíno das estatísticas das equipes."));
children.push(...abntTable("3", "Métricas do ensemble no backtest temporal (6.856 partidas)",
  ["Mercado", "Acurácia", "Baseline", "Log-loss"],
  [["Resultado (H/D/A)", "0,518", "0,457", "0,993"],
   ["Cartões (O/U 4,5)", "0,622", "0,587", "0,628"],
   ["Escanteios (O/U 9,5)", "0,565", "0,521", "0,653"]],
  "elaborado pelo autor (2026).", [3000, 2024, 2024, 2023]));
children.push(H2("4.5", "Comparação entre a rede neural e o gradient boosting"));
children.push(P("Dado o grande volume de dados e a natureza tabular do problema, avaliou-se o modelo de gradient boosting (CatBoost) frente à rede neural, no mesmo conjunto de validação (6.965 partidas, já contemplando as coletas mais recentes). A Tabela 4 sintetiza a comparação: o CatBoost superou a rede neural em acurácia nos três mercados e em log-loss em dois deles. O maior ganho ocorreu nos escanteios (+1,5 ponto percentual), atribuível justamente aos atributos categóricos — liga e equipes —, que capturam tendências de escanteios específicas de cada competição e clube, algo que a rede neural, restrita aos atributos numéricos, não modela diretamente. Cabe notar que, na variante sem atributos categóricos, a vantagem do CatBoost sobre a rede neural foi marginal, o que reforça que o ganho decorre sobretudo do tratamento nativo das categorias. Em razão desse desempenho, o CatBoost foi adotado como modelo padrão do sistema, mantendo-se a rede neural disponível como alternativa selecionável."));
children.push(...abntTable("4", "Comparação entre rede neural e CatBoost no conjunto de validação (6.965 partidas)",
  ["Mercado", "RN (acur.)", "CatBoost (acur.)", "RN (log-loss)", "CatBoost (log-loss)"],
  [["Resultado (H/D/A)", "0,522", "0,524", "0,991", "0,989"],
   ["Cartões (O/U 4,5)", "0,625", "0,630", "0,627", "0,631"],
   ["Escanteios (O/U 9,5)", "0,566", "0,582", "0,654", "0,652"]],
  "elaborado pelo autor (2026).", [3035, 1509, 1509, 1509, 1509]));
children.push(H2("4.6", "Matrizes de confusão e calibração do modelo adotado"));
children.push(P("As Figuras 1 e 2 caracterizam o modelo adotado (CatBoost) sobre o conjunto de validação. A Figura 1 traz as matrizes de confusão: a coluna do empate é quase vazia — evidência visual da dificuldade estrutural de prever empates — e o mercado de escanteios apresenta leve viés para o over."));
children.push(...abntFigure("1", "Matrizes de confusão dos três mercados (modelo CatBoost)", "confusion_matrices.png", "elaborado pelo autor (2026).", 600, 172));
children.push(P("A Figura 2 mostra as curvas de calibração, situadas muito próximas da diagonal: quando o modelo afirma 60% de chance, o evento ocorre em cerca de 60% das vezes. Em problemas de alta incerteza, essa propriedade é tão ou mais valiosa que a acurácia bruta, pois permite quantificar corretamente o risco."));
children.push(...abntFigure("2", "Curvas de calibração por mercado (modelo CatBoost)", "calibration.png", "elaborado pelo autor (2026).", 600, 172));
children.push(H2("4.7", "O problema dos empates"));
children.push(P("Experimentos com pesos de empate no treino confirmaram que forçar a previsão dessa classe reduz a acurácia sem ganho prático: elevar o peso aumenta o recall de empates, mas derruba a acurácia global de 0,518 para 0,444. Conclui-se que não há configuração que, simultaneamente, preveja empates e aumente a acurácia — os empates são quase imprevisíveis como palpite único. A resposta adotada foi metodológica e ética: comunicar a incerteza, sinalizando jogos equilibrados, em vez de cravar um favorito artificial."));
children.push(H2("4.8", "Simulação de retorno e valor esperado"));
children.push(P("Para avaliar o conteúdo informativo do modelo, simulou-se uma estratégia de apostas no período de validação, com odds derivadas das frequências históricas de cada desfecho (um mercado ingênuo). A Figura 3 mostra o lucro acumulado, que cresce de forma consistente nos três mercados, confirmando que o modelo agrega informação além das frequências base. Cabe, porém, uma ressalva de honestidade científica: o retorno é otimista, pois casas de apostas reais já incorporam força e forma das equipes — justamente o que os atributos do modelo capturam. Assim, a curva mede conteúdo informativo, e não um retorno realista de apostas."));
children.push(...abntFigure("3", "Lucro simulado acumulado (estratégia de valor esperado)", "profit_curve.png", "elaborado pelo autor (2026).", 600, 200));
children.push(P("Sobre odds reais coletadas do mercado, o sistema passa a recomendar apostas de valor quando o valor esperado (EV = probabilidade × cotação − 1) supera um piso definido pelo usuário, sugerindo o tamanho da aposta pelo critério de Kelly fracionado (KELLY, 1956)."));
children.push(H2("4.9", "Generalização para seleções: a Copa do Mundo"));
children.push(P("Como extensão, coletaram-se 523 amistosos de seleções (2025–2026) para prever a Copa do Mundo de 2026. Trata-se de uma extrapolação assumida (out-of-distribution): o modelo foi treinado em futebol de clubes e nunca observou seleções. Ainda assim, por usar atributos genéricos de desempenho, produz estimativas plausíveis para os confrontos cujas seleções possuem histórico recente na base. Esse cenário ilustra tanto a flexibilidade do sistema quanto a importância de sinalizar explicitamente os limites de validade de uma previsão."));
children.push(H2("4.10", "Discussão geral"));
children.push(P("O conjunto de resultados sustenta uma tese coerente: o OscaBet opera no teto estatístico do problema, com probabilidades bem calibradas, mas a previsibilidade do futebol — sobretudo em mercados over/under e no empate — é intrinsecamente baixa. A comparação entre os dois modelos reforça um princípio bem estabelecido na literatura: para dados tabulares, o gradient boosting tende a igualar ou superar redes neurais — o que se confirmou aqui, ainda que ambos esbarrem no mesmo teto de previsibilidade. A estabilidade ao longo de dois anos confirma que o desempenho não é fruto de sorte amostral, embora janelas curtas (como uma única rodada) apresentem grande variância, o que pode levar a conclusões equivocadas se a avaliação não for feita em larga escala."));

// 5 CONSIDERAÇÕES FINAIS
children.push(H1("5", "Considerações Finais"));
children.push(P("Este trabalho desenvolveu e avaliou o OscaBet, um agente de inteligência artificial que integra uma rede neural multi-alvo de previsão de futebol a um modelo de linguagem, entregue por meio de uma aplicação web. Demonstrou-se que o modelo supera consistentemente os baselines triviais e iguala modelos tabulares de referência, com probabilidades bem calibradas, atingindo o teto realista de previsibilidade do problema. A comparação entre uma rede neural multi-alvo e um modelo de gradient boosting (CatBoost) mostrou a superioridade deste último para os dados tabulares do problema — sobretudo nos escanteios, graças ao tratamento nativo de atributos categóricos —, levando à sua adoção como modelo padrão do sistema."));
children.push(P("A principal contribuição é metodológica e de postura: em vez de prometer acurácias irreais, o sistema mede com rigor seus próprios limites e comunica a incerteza de forma transparente — por meio de probabilidades calibradas, sinalização de jogos equilibrados e ressalvas explícitas, como na extrapolação para seleções."));
children.push(P([b("Limitações. "), t("A imputação de xG ausente é simplista; o tratamento do empate é uma heurística de decisão, não um avanço de modelagem; e a previsão de seleções permanece experimental. ")]));
children.push(P([b("Trabalhos futuros. "), t("Incluem a exploração de arquiteturas com mais informação temporal, melhor imputação de dados faltantes, métodos de explicabilidade para identificar os atributos mais influentes e a incorporação de odds de mercado de múltiplas casas para uma avaliação de valor mais realista.")]));

// REFERÊNCIAS
children.push(H1("", "Referências"));
children.push(ref([b("BRIER, G. W. "), "Verification of forecasts expressed in terms of probability. ", it("Monthly Weather Review"), ", v. 78, n. 1, p. 1-3, 1950."]));
children.push(ref([b("DIXON, M. J.; COLES, S. G. "), "Modelling association football scores and inefficiencies in the football betting market. ", it("Journal of the Royal Statistical Society: Series C (Applied Statistics)"), ", v. 46, n. 2, p. 265-280, 1997."]));
children.push(ref([b("GOODFELLOW, I.; BENGIO, Y.; COURVILLE, A. "), it("Deep Learning"), ". Cambridge: MIT Press, 2016."]));
children.push(ref([b("HUBÁČEK, O.; ŠOUREK, G.; ŽELEZNÝ, F. "), "Learning to predict soccer results from relational data with gradient boosted trees. ", it("Machine Learning"), ", v. 108, n. 1, p. 29-47, 2019."]));
children.push(ref([b("KELLY JR., J. L. "), "A new interpretation of information rate. ", it("The Bell System Technical Journal"), ", v. 35, n. 4, p. 917-926, 1956."]));
children.push(ref([b("PASZKE, A. "), it("et al. "), "PyTorch: an imperative style, high-performance deep learning library. ", it("In"), ": ADVANCES IN NEURAL INFORMATION PROCESSING SYSTEMS (NeurIPS), 32., 2019. ", it("Proceedings"), " [...]. 2019. p. 8024-8035."]));
children.push(ref([b("PEDREGOSA, F. "), it("et al. "), "Scikit-learn: machine learning in Python. ", it("Journal of Machine Learning Research"), ", v. 12, p. 2825-2830, 2011."]));
children.push(ref([b("PROKHORENKOVA, L. "), it("et al. "), "CatBoost: unbiased boosting with categorical features. ", it("In"), ": ADVANCES IN NEURAL INFORMATION PROCESSING SYSTEMS (NeurIPS), 31., 2018. ", it("Proceedings"), " [...]. 2018. p. 6638-6648."]));
children.push(ref([b("SOFASCORE. "), it("Sofascore — Live Scores, Fixtures & Results"), ". 2026. Disponível em: https://www.sofascore.com. Acesso em: 11 jun. 2026."]));
children.push(ref([b("VASWANI, A. "), it("et al. "), "Attention is all you need. ", it("In"), ": ADVANCES IN NEURAL INFORMATION PROCESSING SYSTEMS (NeurIPS), 30., 2017. ", it("Proceedings"), " [...]. 2017. p. 5998-6008."]));

// ── Documento ─────────────────────────────────────────────────────────────────
const doc = new Document({
  creator: "Ian Marchi",
  title: "OscaBet — Artigo Científico",
  styles: {
    default: { document: { run: { font: FONT, size: SZ } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: SZ, bold: true, font: FONT, color: "000000" }, paragraph: { outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: SZ, bold: true, font: FONT, color: "000000" }, paragraph: { outlineLevel: 1 } },
    ],
  },
  sections: [{
    properties: {
      page: {
        size: { width: 11906, height: 16838 },
        margin: { top: Math.round(3 * CM), left: Math.round(3 * CM), bottom: Math.round(2 * CM), right: Math.round(2 * CM) },
      },
    },
    headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, spacing: { line: 240 },
      children: [new TextRun({ children: [PageNumber.CURRENT], size: SZ_SMALL, font: FONT })] })] }) },
    children,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(path.join(__dirname, "Artigo_OscaBet.docx"), buf);
  console.log("OK: Artigo_OscaBet.docx gerado (" + buf.length + " bytes)");
});
