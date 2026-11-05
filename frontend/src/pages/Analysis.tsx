import { useEffect, useState } from "react";
import { ErrorNote, Icon, Loading } from "../components/ui";
import { NoteRoll, RollLegend } from "../components/NoteRoll";
import { criterionLabel, displayScore, fmtDate, splitFeedback, type Kind } from "../model";
import { Redirect, useApp } from "../state";
import { useResult } from "./useResult";

const FILTERS: { k: Kind | "all"; label: string }[] = [
  { k: "all", label: "All" }, { k: "wrong", label: "Wrong" }, { k: "missed", label: "Missed" },
  { k: "timing", label: "Timing" }, { k: "extra", label: "Extra" },
];

export function Analysis({ id, bar }: { id: string; bar?: number }) {
  const { go, songTitle, me, criteria } = useApp();
  const r = useResult(id);
  const [sel, setSel] = useState<number | null>(null);
  const [filter, setFilter] = useState<Kind | "all">("all");

  useEffect(() => {
    if (bar && r.sections.length) setSel(Math.max(0, r.sections.findIndex((s) => bar >= s.m0 && bar <= s.m1)));
  }, [bar, r.sections]);

  if (r.error) return <ErrorNote error={r.error} onRetry={r.reload} />;
  if (!r.data || !r.roll) return <Loading what="Loading the analysis" />;
  const { a } = r.data;
  const teacher = me.role === "teacher";
  if (!teacher) return <Redirect to={["results", id]} />; // students get the song path on their results page

  const roll = r.roll;
  const sec = sel != null ? r.sections[sel] : null;
  const pad = roll.step * 0.8;
  const from = sec ? Math.max(roll.t0, sec.t0 - pad) : roll.t0;
  const to = sec ? sec.t1 + pad * 1.5 : roll.t1;
  const inView = roll.notes.filter((n) => sec ? (n.measure != null ? n.measure >= sec.m0 && n.measure <= sec.m1 : n.t >= sec.t0 && n.t <= sec.t1) : true);
  const count = (k: Kind) => inView.filter((n) => n.kind === k).length;
  const devs = inView.map((n) => n.note.onset_dev_ms).filter((d): d is number => d != null).map(Math.abs).sort((x, y) => x - y);
  const medDev = devs.length ? Math.round(devs[Math.floor(devs.length / 2)]) : null;

  const { tips } = splitFeedback(a.feedback);
  const secTips = tips.filter((t) => t.bar != null && (!sec || (t.bar >= sec.m0 && t.bar <= sec.m1)));
  const s = a.summary ?? {};
  const crit = Object.entries(a.scoring?.criteria ?? {});
  const overall = displayScore(a, r.review);

  return (
    <>
      <div className="head">
        <div className="col">
          <a className="back" onClick={() => go("results", id)}>← Results</a>
          <h1>Detailed Performance Analysis</h1>
          <span className="sub"><strong>{songTitle(a.song_id)}</strong>{a.student_id ? ` · ${a.student_id}` : ""} · {fmtDate(a.created_at)}{overall != null ? ` · Overall ${overall} / 100` : ""}</span>
        </div>
        {!teacher && <button className="btn" onClick={() => go("new")}>Try Again</button>}
      </div>

      <div className="card tight col" style={{ gap: 10 }}>
        <div className="between muted" style={{ fontSize: 12.5, fontWeight: 600 }}>
          <span>Song timeline · {r.sections.length ? r.sections[r.sections.length - 1].m1 : 0} bars</span>
          <span>Select a section to look closer</span>
        </div>
        <div className="secs">
          <button className={`sec ${sel == null ? "on" : ""}`} onClick={() => setSel(null)} style={{ flex: "0 0 110px", minWidth: 110 }}>
            <b style={{ fontSize: 14 }}>Whole song</b>
            <span className="muted" style={{ fontSize: 12 }}>{roll.notes.filter((n) => n.note.ref_index != null).length} notes</span>
          </button>
          {r.sections.map((x, i) => (
            <button key={x.label} className={`sec ${sel === i ? "on" : ""}`} onClick={() => setSel(i)}>
              <span className="between" style={{ width: "100%" }}><b style={{ fontSize: 14 }}>{x.label}</b><b className="tnum" style={{ fontSize: 12.5 }}>{x.score ?? "–"}</b></span>
              <span className="muted" style={{ fontSize: 12 }}>{x.issues === 0 ? "All good" : `${x.issues} to look at`}</span>
              <div className="meter" style={{ height: 4, width: "100%" }}>
                <i style={{ width: `${x.score ?? 0}%`, background: (x.score ?? 0) >= 85 ? "var(--brand)" : (x.score ?? 0) >= 60 ? "var(--warn)" : "var(--bad)" }} />
              </div>
            </button>
          ))}
        </div>
      </div>

      <div className="an-grid">
        <div className="card col" style={{ gap: 16 }}>
          <div className="between" style={{ flexWrap: "wrap" }}>
            <div className="seg">
              {FILTERS.map((f) => <button key={f.k} className={filter === f.k ? "on" : ""} onClick={() => setFilter(f.k)}>{f.label}</button>)}
            </div>
            <b style={{ fontSize: 13.5 }}>{sec ? sec.label : "Whole song"}</b>
          </div>
          <RollLegend expected />
          <NoteRoll roll={roll} from={from} to={to} rowH={sec ? 16 : 12} filter={filter} />
          <div className="stats">
            <div className="stat"><span>Correct</span><b className="tnum">{count("ok") + count("timing")}</b></div>
            <div className="stat"><span>Wrong notes</span><b className="tnum">{count("wrong")}</b></div>
            <div className="stat"><span>Missed</span><b className="tnum">{count("missed")}</b></div>
            <div className="stat"><span>Extra</span><b className="tnum">{count("extra")}</b></div>
            <div className="stat"><span>Typical timing</span><b className="tnum">{medDev != null ? `±${medDev} ms` : "–"}</b></div>
          </div>
          {!sec && (
            <div className="stats">
              <div className="stat"><span>Tempo played</span><b className="tnum">{s.mean_bpm ? `${Math.round(s.mean_bpm)} bpm` : "–"}</b></div>
              <div className="stat"><span>Score tempo</span><b className="tnum">{s.reference_bpm ? `${Math.round(s.reference_bpm)} bpm` : "–"}</b></div>
              <div className="stat"><span>Pauses</span><b className="tnum">{s.pause_count ?? "–"}</b></div>
              <div className="stat"><span>Restarts</span><b className="tnum">{s.restart_count ?? "–"}</b></div>
              <div className="stat"><span>Song match</span><b className="tnum">{s.reference_fit_pct != null ? `${Math.round(s.reference_fit_pct)}%` : "–"}</b></div>
            </div>
          )}
        </div>

        <div className="col" style={{ gap: 20 }}>
          <div className="card col" style={{ gap: 12, padding: 20 }}>
            <div className="between"><h2>Section tips</h2><span className="muted" style={{ fontSize: 12, fontWeight: 600 }}>{secTips.length}</span></div>
            {secTips.length === 0 ? <span className="muted small">No specific tips {sec ? "for these bars" : ""}. Nice!</span> :
              secTips.slice(0, 8).map((t, i) => (
                <div key={i} className="col" style={{ gap: 6, padding: 12, borderRadius: 12, border: "1.5px solid var(--line)" }}>
                  <span className="tag grow">Bar {t.bar}</span>
                  <span style={{ fontSize: 14, lineHeight: 1.45, fontWeight: 500 }}>{t.text}</span>
                </div>
              ))}
          </div>

          {teacher && (
            <div className="card col" style={{ gap: 8, padding: 20 }}>
              <h2>All criteria</h2>
              <span className="muted" style={{ fontSize: 12.5 }}>The client's 22 criteria as RAS scored them. "Not assessed" means it could not be measured, never guessed.</span>
              {crit.map(([cid, c]) => (
                <div key={cid} className="between" style={{ fontSize: 13, padding: "5px 0", borderTop: "1px solid var(--line-2)" }} title={c.reason ?? c.metric}>
                  <span>{criterionLabel(cid, criteria)}</span>
                  {c.status === "assessed" && c.score != null ? <b className="tnum">{Math.round(c.score)}</b> : <span className="muted" style={{ fontSize: 12 }}>not assessed</span>}
                </div>
              ))}
              {a.confidence != null && (
                <div className="note-banner" style={{ marginTop: 6, fontSize: 13 }}>
                  <Icon name="info" size={16} /><span>AI confidence {Math.round(a.confidence * 100)}%. Confidence is a first estimate until it is calibrated on teacher reviews.</span>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </>
  );
}
