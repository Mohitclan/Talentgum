import { useEffect, useMemo, useState } from "react";
import { api, type Assessment } from "../api";
import { ErrorNote, Icon, Loading, StatusPill } from "../components/ui";
import { displayScore, fmtDate, groupScores, initials, levelOfRubric, REVIEW_REASON } from "../model";
import { useApp, useLoad } from "../state";
import { useResult } from "./useResult";

const FILTERS = [
  { k: "todo", label: "To check", match: (a: Assessment) => a.status === "review_required" || a.status === "completed" },
  { k: "shared", label: "Shared", match: (a: Assessment) => a.status === "reviewed" },
  { k: "running", label: "In progress", match: (a: Assessment) => a.status === "queued" || a.status === "processing" },
  { k: "problem", label: "Problems", match: (a: Assessment) => a.status === "rejected" || a.status === "failed" },
  { k: "all", label: "All", match: () => true },
];

export function TeacherHome({ selected }: { selected?: string }) {
  const { go, songTitle, criteria } = useApp();
  const list = useLoad(() => api.assessments({ limit: 200 }), []);
  const [filter, setFilter] = useState("todo");
  const [student, setStudent] = useState("");
  const [report, setReport] = useState(false);

  const rows = useMemo(() => {
    const f = FILTERS.find((x) => x.k === filter)!;
    return (list.data ?? []).filter(f.match).filter((a) => !student || a.student_id === student);
  }, [list.data, filter, student]);
  const students = useMemo(() => [...new Set((list.data ?? []).map((a) => a.student_id).filter(Boolean) as string[])].sort(), [list.data]);
  const counts = useMemo(() => Object.fromEntries(FILTERS.map((f) => [f.k, (list.data ?? []).filter(f.match).length])), [list.data]);

  const sel = (list.data ?? []).find((a) => a.assessment_id === selected) ?? rows[0];

  // keep polling while anything is still being analysed
  const busy = (list.data ?? []).some((a) => a.status === "queued" || a.status === "processing");
  useEffect(() => {
    if (!busy) return;
    const t = setInterval(list.reload, 4000);
    return () => clearInterval(t);
  }, [busy, list.reload]);

  return (
    <>
      <div className="head">
        <div className="col">
          <span className="muted" style={{ fontSize: 13, fontWeight: 600 }}>Teacher view</span>
          <h1>Student Performance</h1>
          <span className="sub">Check AI assessments, adjust where needed, and share them with students.</span>
        </div>
        <div className="row">
          <select className="field" style={{ width: 200 }} value={student} onChange={(e) => setStudent(e.target.value)} aria-label="Filter by student">
            <option value="">All students</option>
            {students.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
          <button className="btn primary" onClick={() => go("new")}><Icon name="plus" size={16} stroke={2.2} />New assessment</button>
        </div>
      </div>

      {list.error && <ErrorNote error={list.error} onRetry={list.reload} />}

      {sel && (
        <div className="card row" style={{ gap: 28, flexWrap: "wrap" }}>
          <div className="row" style={{ gap: 14, minWidth: 220 }}>
            <div className="avatar lg">{initials(sel.student_id ?? "?")}</div>
            <div className="col" style={{ gap: 2 }}><b style={{ fontSize: 17 }}>{sel.student_id ?? "Unknown student"}</b><span className="muted small">Student</span></div>
          </div>
          <Fact k="Song" v={songTitle(sel.song_id)} />
          <Fact k="Level" v={levelOfRubric(sel.rubric.id) ?? "General"} />
          <Fact k="Date" v={fmtDate(sel.created_at)} />
          <Fact k="AI score" v={sel.total_score != null ? `${Math.round(sel.total_score)} / 100` : "–"} big />
          <div style={{ flex: 1 }} />
          <div className="row">
            <button className="btn" disabled={!isFinished(sel)} onClick={() => go("analysis", sel.assessment_id)}>Open analysis</button>
            <button className="btn primary" disabled={!isFinished(sel)} onClick={() => setReport(true)}><Icon name="report" size={16} stroke={2} />
              {sel.status === "reviewed" ? "Teacher report" : "Check & share"}</button>
          </div>
          {sel.status === "review_required" && (sel.confidence_detail?.reasons?.length ?? 0) > 0 && (
            <div className="note-banner warn" style={{ width: "100%" }}><Icon name="info" />
              <span>Flagged for a closer look: {sel.confidence_detail!.reasons.map((x) => REVIEW_REASON[x] ?? x.replace(/_/g, " ")).join(" · ")}.</span>
            </div>
          )}
        </div>
      )}

      <div className="card" style={{ padding: 0, overflow: "hidden" }}>
        <div className="between" style={{ padding: "18px 24px", flexWrap: "wrap" }}>
          <h2>Assessments</h2>
          <div className="seg">
            {FILTERS.map((f) => (
              <button key={f.k} className={filter === f.k ? "on" : ""} onClick={() => setFilter(f.k)}>
                {f.label}{f.k !== "all" && counts[f.k] ? ` · ${counts[f.k]}` : ""}
              </button>
            ))}
          </div>
        </div>
        <div className="thead"><span>Student</span><span>Song</span><span>Score</span><span>Notes</span><span>Timing</span><span>Expression</span><span>Status</span><span>Date</span><span /></div>
        {list.loading && !list.data ? <Loading /> : rows.length === 0 ? <div className="empty">Nothing here.</div> : rows.map((a) => {
          const g = groupScores(a, criteria);
          const gv = (id: string) => g.find((x) => x.id === id)?.score;
          return (
            <div key={a.assessment_id} className={`trow ${sel?.assessment_id === a.assessment_id ? "sel" : ""}`} onClick={() => go("teacher", a.assessment_id)}>
              <span className="row" style={{ gap: 10, fontWeight: 700 }}><span className="avatar grey">{initials(a.student_id ?? "?")}</span>{a.student_id ?? "–"}</span>
              <span className="col" style={{ gap: 1 }}>{songTitle(a.song_id)}{levelOfRubric(a.rubric.id) && <span className="muted" style={{ fontSize: 12 }}>{levelOfRubric(a.rubric.id)}</span>}</span>
              <b className="tnum" style={{ fontSize: 14.5 }}>{displayScore(a) ?? "–"}</b>
              <span className="tnum" style={{ color: "var(--ink-2)" }}>{gv("note") ?? "–"}</span>
              <span className="tnum" style={{ color: "var(--ink-2)" }}>{gv("timing") ?? "–"}</span>
              <span className="tnum" style={{ color: "var(--ink-2)" }}>{gv("performance") ?? "–"}</span>
              <StatusPill a={a} view="teacher" />
              <span className="muted small">{fmtDate(a.created_at)}</span>
              <span style={{ color: "#9a948a" }}><Icon name="chev" size={16} stroke={2} /></span>
            </div>
          );
        })}
        <div className="ai-note" style={{ padding: "14px 24px" }}>
          Students see a result only after you share it. "Needs review" marks results the AI was unsure about.
        </div>
      </div>

      {report && sel && <ReportDrawer id={sel.assessment_id} onClose={() => setReport(false)} onShared={() => { setReport(false); list.reload(); }} />}
    </>
  );
}

const isFinished = (a: Assessment) => ["completed", "review_required", "reviewed"].includes(a.status);

function Fact({ k, v, big }: { k: string; v: string; big?: boolean }) {
  return (
    <div className="col" style={{ gap: 3, paddingLeft: 28, borderLeft: "1px solid var(--line-2)" }}>
      <span className="muted" style={{ fontSize: 12, fontWeight: 600 }}>{k}</span>
      <b style={{ fontSize: big ? 22 : 15, letterSpacing: big ? "-.02em" : undefined }}>{v}</b>
    </div>
  );
}

const DRAFTS = "tg.drafts";
function loadDraft(id: string): { comment: string; score: string } | null {
  try { return JSON.parse(localStorage.getItem(DRAFTS) ?? "{}")[id] ?? null; } catch { return null; }
}
function saveDraft(id: string, d: { comment: string; score: string } | null) {
  try {
    const all = JSON.parse(localStorage.getItem(DRAFTS) ?? "{}");
    if (d) all[id] = d; else delete all[id];
    localStorage.setItem(DRAFTS, JSON.stringify(all));
  } catch { /* private mode */ }
}

function ReportDrawer({ id, onClose, onShared }: { id: string; onClose: () => void; onShared: () => void }) {
  const { me, setMe, criteria, songTitle, reportError } = useApp();
  const r = useResult(id);
  const [comment, setComment] = useState("");
  const [score, setScore] = useState("");
  const [checked, setChecked] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [audio, setAudio] = useState<string | null>(null);

  useEffect(() => {
    if (!r.data) return;
    const d = loadDraft(id);
    setComment(d?.comment ?? r.review?.comment ?? "");
    setScore(d?.score ?? (r.review?.overall_score != null ? String(Math.round(r.review.overall_score)) : ""));
  }, [r.data, r.review, id]);
  useEffect(() => () => { if (audio) URL.revokeObjectURL(audio); }, [audio]);

  const a = r.data?.a;
  const scoreNum = score.trim() === "" ? null : Number(score);
  const scoreOk = scoreNum == null || (Number.isFinite(scoreNum) && scoreNum >= 0 && scoreNum <= 100);

  async function share() {
    if (!a || !checked || !scoreOk) return;
    setSaving(true);
    setError(null);
    try {
      await api.review(id, { reviewer: me.teacherName || "Teacher", overall_score: scoreNum ?? a.total_score, comment: comment.trim() || undefined });
      saveDraft(id, null);
      onShared();
    } catch (e) {
      setError(reportError(e));
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="scrim" onClick={onClose}>
      <div className="drawer" onClick={(e) => e.stopPropagation()} role="dialog" aria-label="Teacher report">
        <div className="between" style={{ alignItems: "flex-start" }}>
          <div className="col" style={{ gap: 4 }}>
            <span className="muted" style={{ fontSize: 12, fontWeight: 700, letterSpacing: ".05em", textTransform: "uppercase" }}>
              Teacher report {a?.status === "reviewed" ? "· shared" : "· draft"}
            </span>
            <span style={{ fontSize: 22, fontWeight: 800 }}>{a?.student_id ?? ""}</span>
            {a && <span className="muted small">{songTitle(a.song_id)} · {fmtDate(a.created_at)}</span>}
          </div>
          <button className="linkbtn" style={{ fontSize: 24, lineHeight: 1 }} onClick={onClose} aria-label="Close">×</button>
        </div>
        {!a ? <Loading /> : (
          <>
            <div className="card tight" style={{ display: "grid", gridTemplateColumns: "repeat(3, minmax(0,1fr))", gap: 12 }}>
              <div className="col" style={{ gap: 2 }}><span className="muted" style={{ fontSize: 12, fontWeight: 600 }}>AI overall</span><b style={{ fontSize: 22 }}>{a.total_score != null ? Math.round(a.total_score) : "–"}</b></div>
              {groupScores(a, criteria).map((g) => (
                <div key={g.id} className="col" style={{ gap: 2 }}><span className="muted" style={{ fontSize: 12, fontWeight: 600 }}>{g.label}</span><b style={{ fontSize: 22 }}>{g.score != null ? `${g.score}%` : "–"}</b></div>
              ))}
            </div>

            <div className="card tight col" style={{ gap: 10 }}>
              <div className="between"><b style={{ fontSize: 14 }}>Recording</b>
                {!audio && <button className="btn sm" onClick={() => api.recording(a.recording_file_id).then(setAudio, (e) => setError(reportError(e)))}><Icon name="play" size={14} />Load</button>}
              </div>
              {audio ? <audio controls src={audio} style={{ width: "100%" }} /> : <span className="muted small">Listen before sharing. Every playback is logged.</span>}
            </div>

            <div className="card tight col" style={{ gap: 8 }}>
              <b style={{ fontSize: 14 }}>AI feedback the student will see</b>
              {(a.feedback ?? []).slice(0, 6).map((f, i) => <span key={i} style={{ fontSize: 13.5, lineHeight: 1.45, color: "var(--ink-2)" }}>• {f}</span>)}
            </div>

            <div className="col" style={{ gap: 6 }}>
              <label className="lbl" htmlFor="score">Score to share (leave empty to keep the AI score)</label>
              <input id="score" className="field" inputMode="numeric" placeholder={a.total_score != null ? String(Math.round(a.total_score)) : "0–100"}
                value={score} onChange={(e) => setScore(e.target.value)} style={{ width: 140, borderColor: scoreOk ? undefined : "var(--bad)" }} />
            </div>
            <div className="col" style={{ gap: 6 }}>
              <label className="lbl" htmlFor="comment">Personal note for the student</label>
              <textarea id="comment" className="field" placeholder="Add an encouraging note…" value={comment} onChange={(e) => setComment(e.target.value)} />
            </div>
            <div className="col" style={{ gap: 6 }}>
              <label className="lbl" htmlFor="tname">Your name (shown to the student)</label>
              <input id="tname" className="field" value={me.teacherName} onChange={(e) => setMe({ teacherName: e.target.value })} />
            </div>
            <label className="row" style={{ gap: 10, fontSize: 13.5, fontWeight: 600, cursor: "pointer" }}>
              <input type="checkbox" checked={checked} onChange={(e) => setChecked(e.target.checked)} />
              I have checked this AI assessment
            </label>
            {error && <ErrorNote error={error} />}
            <div style={{ flex: 1 }} />
            <div className="row" style={{ justifyContent: "flex-end" }}>
              <button className="btn" onClick={() => { saveDraft(id, { comment, score }); onClose(); }}>Save draft</button>
              <button className="btn primary" disabled={!checked || !scoreOk || saving} onClick={share}>
                {saving ? "Sharing…" : a.status === "reviewed" ? "Update for student" : "Share with student"}
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
