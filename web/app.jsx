const { useState, useRef, useEffect, useCallback } = React;

/* ───────────────────────── Ícones (SVG, sem emoji) ───────────────────── */
const Icon = {
  Ball: (p) => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" {...p}>
      <circle cx="12" cy="12" r="9" />
      <path d="M12 7.5l3.2 2.3-1.2 3.7h-4l-1.2-3.7L12 7.5z" fill="currentColor" stroke="none" />
      <path d="M12 3v2.2M4.8 8.5l2 1.2M4.8 15.5l2-1.2M19.2 8.5l-2 1.2M19.2 15.5l-2-1.2M9 19.5l1-2.2M15 19.5l-1-2.2" />
    </svg>
  ),
  Send: (p) => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...p}>
      <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z" />
    </svg>
  ),
  Spark: (p) => (
    <svg viewBox="0 0 24 24" fill="currentColor" {...p}>
      <path d="M12 2l1.8 5.6L19.5 9l-4.6 3.4L16.3 18 12 14.7 7.7 18l1.4-5.6L4.5 9l5.7-1.4L12 2z" />
    </svg>
  ),
  Chart: (p) => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" {...p}>
      <path d="M4 20V10M10 20V4M16 20v-7M22 20H2" />
    </svg>
  ),
  User: (p) => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" {...p}>
      <circle cx="12" cy="8" r="3.4" /><path d="M5 20a7 7 0 0114 0" strokeLinecap="round" />
    </svg>
  ),
};

/* ───────────────────────── Barra de probabilidade ───────────────────── */
function ProbBar({ segments }) {
  const [on, setOn] = useState(false);
  useEffect(() => {
    const id = requestAnimationFrame(() => setOn(true));
    return () => cancelAnimationFrame(id);
  }, []);
  return (
    <div className="bar" role="img" aria-label={segments.map(s => `${s.label} ${s.pct}%`).join(", ")}>
      {segments.map((s) => (
        <div key={s.key} className={`bar__seg seg--${s.cls}`} style={{ width: on ? `${s.pct}%` : "0%" }}>
          {s.pct >= 12 && <span>{s.pct}%</span>}
        </div>
      ))}
    </div>
  );
}

function Market({ label, pick, confidence, segments, badge }) {
  return (
    <div className="market">
      <div className="market__top">
        <span className="market__label">{label}{badge && <span className="market__badge">{badge}</span>}</span>
        <span className="market__pick">{pick} · <b>{Math.round(confidence * 100)}%</b></span>
      </div>
      <ProbBar segments={segments} />
      <div className="legend">
        {segments.map((s) => (
          <span className="legend__item" key={s.key}>
            <span className="legend__sw" style={{ background: s.color }} />{s.label} {s.pct}%
          </span>
        ))}
      </div>
    </div>
  );
}

const C = { pitch: "#15E27F", amber: "#FBBF24", rose: "#F43F5E", cyan: "#22D3EE", violet: "#7C3AED" };
const RES_NAME = { H: "Casa", D: "Empate", A: "Visitante" };
const pct = (x) => Math.round((x || 0) * 100);

/* ───────────────────────── Cartão de previsão ───────────────────────── */
function PredictionCard({ pred }) {
  const r = pred.resultado, y = pred.cartoes, c = pred.escanteios;
  const resSeg = [
    { key: "h", cls: "h", color: C.pitch,  label: "Casa",     pct: pct(r.probs["H"]) },
    { key: "d", cls: "d", color: C.amber,  label: "Empate",   pct: pct(r.probs["D"]) },
    { key: "a", cls: "a", color: C.rose,   label: "Visitante",pct: pct(r.probs["A"]) },
  ];
  // Cor por SIGNIFICADO (Under=ciano, Over=violeta), não pela ordem das chaves.
  const ouSeg = (probs) => {
    const keys = Object.keys(probs);
    const underKey = keys.find((k) => /under/i.test(k)) || keys[0];
    const overKey  = keys.find((k) => /over/i.test(k))  || keys[1];
    return [
      { key: "u", cls: "under", color: C.cyan,   label: underKey, pct: pct(probs[underKey]) },
      { key: "o", cls: "over",  color: C.violet, label: overKey,  pct: pct(probs[overKey]) },
    ];
  };
  return (
    <div className="pcard">
      <div className="pcard__head">
        <Icon.Chart />
        <span className="t">PREVISÃO DA REDE NEURAL</span>
        {pred.ficticio && <span className="pcard__fic">Fictício</span>}
        {pred.match && <span className="m">{pred.match}{pred.competition ? ` · ${pred.competition}` : ""}</span>}
      </div>
      <div className="pcard__body">
        <Market label="Resultado (1×2)" pick={RES_NAME[r.label] || r.label} confidence={r.confidence}
                segments={resSeg} badge={r.equilibrado ? "Equilibrado" : null} />
        <Market label="Cartões" pick={y.label} confidence={y.confidence} segments={ouSeg(y.probs)} />
        <Market label="Escanteios" pick={c.label} confidence={c.confidence} segments={ouSeg(c.probs)} />
      </div>
      <div className="pcard__foot">
        <Icon.Spark style={{ width: 12, height: 12, color: C.pitch }} />
        Probabilidades estimadas — não são garantia de resultado.
      </div>
    </div>
  );
}

/* ───────────────────────── Bolha de mensagem ────────────────────────── */
function Message({ role, content, tools, prediction, typing }) {
  const isAI = role !== "user";
  return (
    <div className={`msg msg--${isAI ? "ai" : "user"}`}>
      <div className="msg__avatar">{isAI ? <Icon.Ball /> : <Icon.User />}</div>
      <div style={{ maxWidth: "84%" }}>
        <div className="msg__bubble">
          {typing ? (
            <div className="typing"><span></span><span></span><span></span></div>
          ) : content}
        </div>
        {!typing && tools && tools.length > 0 && (
          <div className="tools">
            {tools.map((t) => <span className="tools__pill" key={t}>{t.replace(/_/g, " ")}</span>)}
          </div>
        )}
        {!typing && prediction && <PredictionCard pred={prediction} />}
      </div>
    </div>
  );
}

/* ───────────────────────── Tela de boas-vindas ──────────────────────── */
const SUGGESTIONS = [
  { t: "Faça a previsão de Flamengo x Grêmio no Brasileirão", lead: "Previsão" },
  { t: "Como está a forma recente do Manchester City?", lead: "Forma" },
  { t: "Tabela atual do Brasileirão Série A", lead: "Tabela" },
  { t: "Histórico de confrontos Real Madrid x Getafe", lead: "Confrontos" },
];
function Welcome({ onPick }) {
  return (
    <div className="welcome">
      <div className="welcome__badge"><Icon.Spark style={{ width: 13, height: 13 }} /> Especialista de futebol com IA</div>
      <h2>Pergunte. O OscaBet responde.</h2>
      <p>Converse em linguagem natural sobre times, estatísticas e confrontos. Peça uma previsão e a rede neural entra em campo.</p>
      <div className="chips">
        {SUGGESTIONS.map((s) => (
          <button key={s.t} className="chip" onClick={() => onPick(s.t)}>
            <b>{s.lead} ·</b> {s.t}
          </button>
        ))}
      </div>
    </div>
  );
}

/* ───────────────────────── Dashboard (Painel) ───────────────────────── */
const fmtDate = (ts) => ts ? new Date(ts * 1000).toLocaleDateString("pt-BR", { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" }) : "";

function Select({ value, onChange, options, placeholder }) {
  return (
    <select className="sel" value={value} onChange={(e) => onChange(e.target.value)}>
      {placeholder && <option value="">{placeholder}</option>}
      {options.map((o) => {
        const v = o.value ?? o, l = o.label ?? o;
        return <option key={v} value={v}>{l}</option>;
      })}
    </select>
  );
}

function Spinner({ text }) {
  return <div className="spin"><span className="spin__ball" />{text || "Carregando…"}</div>;
}

function Simulador({ leagues }) {
  const [league, setLeague] = useState("brasileirao_a");
  const [teams, setTeams] = useState([]);
  const [home, setHome] = useState("");
  const [away, setAway] = useState("");
  const [pred, setPred] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!league) return;
    fetch(`/api/teams?league=${league}`).then((r) => r.json())
      .then((d) => { setTeams(d.teams || []); setHome(""); setAway(""); setPred(null); });
  }, [league]);

  const run = async () => {
    if (!home || !away || loading) return;
    setLoading(true); setPred(null);
    try {
      const r = await fetch(`/api/predict?home=${encodeURIComponent(home)}&away=${encodeURIComponent(away)}&league=${league}`);
      setPred(await r.json());
    } catch (e) { setPred({ error: "Falha na previsão." }); }
    setLoading(false);
  };

  return (
    <div className="panel">
      <h3 className="panel__h"><Icon.Spark style={{ width: 15, height: 15, color: C.pitch }} /> Simulador de confronto</h3>
      <p className="panel__sub">Escolha dois times e veja a previsão da rede neural na hora.</p>
      <div className="panel__row">
        <Select value={league} onChange={setLeague} options={leagues} />
        <Select value={home} onChange={setHome} options={teams} placeholder="Mandante…" />
        <span className="vs">×</span>
        <Select value={away} onChange={setAway} options={teams} placeholder="Visitante…" />
        <button className="btn" onClick={run} disabled={!home || !away || loading}>Prever</button>
      </div>
      {loading && <Spinner text="Rodando a rede neural…" />}
      {pred && pred.error && <div className="err">⚠ {pred.error}</div>}
      {pred && !pred.error && <PredictionCard pred={pred} />}
    </div>
  );
}

function MatchCard({ m }) {
  const r = m.prediction.resultado;
  return (
    <div className="mcard">
      <div className="mcard__top">
        <span className="mcard__teams">{m.home} <span className="mcard__vs">×</span> {m.away}</span>
        <span className="mcard__date">{fmtDate(m.ts)}</span>
      </div>
      <div className="mcard__pick">
        Palpite: <b>{RES_NAME[r.label] || r.label}</b> · {Math.round(r.confidence * 100)}%
        {r.equilibrado && <span className="market__badge">Equilibrado</span>}
      </div>
      {m.value_bets.length > 0 ? (
        <div className="mcard__bets">
          {m.value_bets.map((b, i) => (
            <div className="vbet" key={i}>
              <span className="vbet__mkt">{b.mercado}</span>
              <span>{b.aposta} @ <b>{b.odd}</b></span>
              <span className="vbet__ev">EV +{b.ev_pct}%</span>
              <span className="vbet__stake">stake {b.stake_pct}%</span>
            </div>
          ))}
        </div>
      ) : (
        <div className="mcard__noodds">{m.has_odds ? "Sem aposta de valor neste jogo." : "Odds ainda não disponíveis."}</div>
      )}
    </div>
  );
}

function Rodada({ leagues }) {
  const [league, setLeague] = useState("brasileirao_a");
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);

  const buscar = async () => {
    setLoading(true); setData(null);
    try {
      const r = await fetch(`/api/upcoming?league=${league}&max=8`);
      setData(await r.json());
    } catch (e) { setData({ error: "Falha ao buscar a rodada." }); }
    setLoading(false);
  };

  return (
    <div className="panel">
      <h3 className="panel__h"><Icon.Chart style={{ width: 15, height: 15, color: C.pitch }} /> Próxima rodada · apostas de valor</h3>
      <p className="panel__sub">Jogos futuros com previsão e apostas onde o EV passa de 5% (odds reais do Sofascore).</p>
      <div className="panel__row">
        <Select value={league} onChange={setLeague} options={leagues} />
        <button className="btn" onClick={buscar} disabled={loading}>Buscar rodada</button>
      </div>
      {loading && <Spinner text="Buscando jogos e odds (pode levar alguns segundos)…" />}
      {data && data.error && <div className="err">⚠ {data.error}</div>}
      {data && data.matches && data.matches.length === 0 && (
        <div className="empty">Nenhum jogo próximo encontrado. Pode ser período de pausa do campeonato.</div>
      )}
      {data && data.matches && data.matches.map((m) => <MatchCard key={m.id} m={m} />)}
    </div>
  );
}

function Painel({ leagues }) {
  return (
    <main className="painel">
      <Simulador leagues={leagues} />
      <Rodada leagues={leagues} />
    </main>
  );
}

/* ───────────────────────── App ──────────────────────────────────────── */
function App() {
  const [messages, setMessages] = useState([]);   // {role, content, tools?, prediction?}
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [online, setOnline] = useState(null);
  const [tab, setTab] = useState("chat");
  const [leagues, setLeagues] = useState([]);
  const scroller = useRef(null);
  const taRef = useRef(null);

  useEffect(() => {
    fetch("/api/health").then(r => r.json())
      .then(d => setOnline(!!d.agent_ready))
      .catch(() => setOnline(false));
    fetch("/api/leagues").then(r => r.json())
      .then(d => setLeagues((d || []).map(l => ({ value: l.key, label: l.label }))))
      .catch(() => {});
  }, []);

  useEffect(() => {
    const el = scroller.current;
    if (el) el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }, [messages, loading]);

  const autosize = (el) => { el.style.height = "auto"; el.style.height = Math.min(el.scrollHeight, 130) + "px"; };

  const send = useCallback(async (text) => {
    const msg = (text ?? input).trim();
    if (!msg || loading) return;
    setInput("");
    if (taRef.current) taRef.current.style.height = "auto";

    const history = messages.map(({ role, content }) => ({ role, content }));
    const next = [...messages, { role: "user", content: msg }];
    setMessages(next);
    setLoading(true);

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: msg, history }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Erro");
      setMessages([...next, {
        role: "assistant",
        content: data.text || "(sem resposta)",
        tools: data.tools_used || [],
        prediction: data.has_prediction ? data.prediction : null,
      }]);
    } catch (e) {
      setMessages([...next, { role: "assistant", content: "⚠ " + (e.message || "Falha na conexão com o agente.") }]);
    } finally {
      setLoading(false);
    }
  }, [input, loading, messages]);

  const onKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
  };

  return (
    <React.Fragment>
      <div className="bg-fx"><div className="bg-fx__orb bg-fx__orb--1" /><div className="bg-fx__orb bg-fx__orb--2" /></div>
      <div className="app">
        <header className="hdr">
          <div className="hdr__logo"><Icon.Ball /></div>
          <div className="hdr__title">
            <h1>Osca<b>Bet</b></h1>
            <span>IA de futebol · previsões por rede neural</span>
          </div>
          <div className="hdr__status">
            <span className={`dot ${online === null ? "" : online ? "dot--on" : "dot--off"}`} />
            {online === null ? "conectando…" : online ? "online" : "offline"}
          </div>
        </header>

        <nav className="tabs">
          <button className={`tabs__btn ${tab === "chat" ? "on" : ""}`} onClick={() => setTab("chat")}>Chat</button>
          <button className={`tabs__btn ${tab === "painel" ? "on" : ""}`} onClick={() => setTab("painel")}>Painel</button>
        </nav>

        {tab === "painel" && <Painel leagues={leagues} />}

        {tab === "chat" && <React.Fragment>
        <main className="chat" ref={scroller}>
          {messages.length === 0 && !loading
            ? <Welcome onPick={(t) => send(t)} />
            : messages.map((m, i) => <Message key={i} {...m} />)}
          {loading && <Message role="assistant" typing />}
        </main>

        <footer className="composer">
          <div className="composer__box">
            <textarea
              ref={taRef}
              rows="1"
              placeholder="Pergunte sobre um time, confronto ou peça uma previsão…"
              value={input}
              onChange={(e) => { setInput(e.target.value); autosize(e.target); }}
              onKeyDown={onKey}
              aria-label="Mensagem"
            />
            <button className="send" onClick={() => send()} disabled={loading || !input.trim()} aria-label="Enviar">
              <Icon.Send />
            </button>
          </div>
          <div className="composer__hint">Enter envia · Shift+Enter quebra linha</div>
        </footer>
        </React.Fragment>}
      </div>
    </React.Fragment>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
