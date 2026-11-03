import { CategoryCard, ErrorNote, Icon, Loading, Ring } from "../components/ui";
import { NoteRoll, RollLegend } from "../components/NoteRoll";
import { displayScore, fmtDate, groupScores, headline, REJECT_REASON, splitFeedback, strengths, visibleToStudent } from "../model";
import { Redirect, useApp } from "../state";
import { useResult } from "./useResult";

export function Results({ id }: { id: string }) {
  const { go, songTitle, me, criteria } = useApp();
  const r = useResult(id);
  if (r.error) return <ErrorNote error={r.error} onRetry={r.reload} />;
  if (!r.data || !r.roll) return <Loading what="Loading the result" />;
  const { a } = r.data;
  const teacher = me.role === "teacher";
  const back = teacher ? () => go("teacher") : () => go("home");

  if (a.status === "queued" || a.status === "processing") {
    return <Redirect to={["processing", id]} />;
  }
  if (a.status === "rejected" || a.status === "failed") {
    return (
      <div className="col" style={{ gap: 20, maxWidth: 720 }}>
        <a className="back" onClick={back}>← Back</a>
        <h1>{songTitle(a.song_id)}</h1>
        <div className="note-banner warn"><Icon name="info" />
          <span>{REJECT_REASON[a.error_code ?? ""] ?? "This recording could not be assessed."}{teacher && a.error_detail ? ` (${a.error_detail})` : ""} Please try recording again in a quiet room, close to the keyboard.</span>
        </div>
        <button className="btn primary" style={{ alignSelf: "flex-start" }} onClick={() => go("new")}>Try again</button>
      </div>
    );
  }
  if (!teacher && !visibleToStudent(a)) {
    return (
      <div className="col" style={{ gap: 20, maxWidth: 720 }}>
        <a className="back" onClick={back}>← Dashboard</a>
        <h1>{songTitle(a.song_id)}</h1>
        <div className="card col" style={{ alignItems: "center", textAlign: "center", gap: 12, padding: 40 }}>
          <div className="ico-box"><Icon name="people" size={22} /></div>
          <h2>Your teacher is checking this one</h2>
          <span className="muted" style={{ maxWidth: 440, lineHeight: 1.5 }}>
            Your recording has been analysed. Your teacher looks at every result before it is shared, so your score and tips will appear here soon.
          </span>
        </div>
      </div>
    );
  }

  const score = displayScore(a, r.review);
  const groups = groupScores(a, criteria);
  const { opening, tips } = splitFeedback(a.feedback);
  const wins = strengths(a, criteria);
  const teacherAdjusted = r.review?.overall_score != null && a.total_score != null && Math.round(r.review.overall_score) !== Math.round(a.total_score);

  return (
    <>
      <div className="head">
        <div className="col">
          <a className="back" onClick={back}>← {teacher ? "Student performance" : "Dashboard"}</a>
          <h1>{teacher ? "Assessment result" : "Your Performance Results"}</h1>
          <span className="sub"><strong>{songTitle(a.song_id)}</strong> · {fmtDate(a.created_at)}{teacher && a.student_id ? ` · ${a.student_id}` : ""}</span>
        </div>
        <div className="row">
          {!teacher && <button className="btn" onClick={() => go("new")}>Try Again</button>}
          <button className="btn primary" onClick={() => go("analysis", id)}>View Detailed Analysis</button>
        </div>
      </div>

      {teacher && a.status !== "reviewed" && (
        <div className="note-banner warn"><Icon name="info" />
          <span>Not yet shared with the student. Check it, then share it from <a onClick={() => go("teacher", id)}>Student performance</a>.</span>
        </div>
      )}

      <div className="res-top">
        <div className="card row" style={{ gap: 22, padding: 26 }}>
          <Ring value={score}>
            <span style={{ fontSize: 40, fontWeight: 800, letterSpacing: "-.03em", lineHeight: 1 }}>{score ?? "–"}</span>
            <span className="muted" style={{ fontSize: 13, fontWeight: 600 }}>/ 100</span>
          </Ring>
          <div className="col">
            <span style={{ fontSize: 22, fontWeight: 800, letterSpacing: "-.02em" }}>{score != null ? headline(score) : "Score withheld"}</span>
            <span className="muted" style={{ fontSize: 13.5, lineHeight: 1.5 }}>
              {score == null ? "The playing did not match this song closely enough to give a score." : opening}
            </span>
            {teacherAdjusted && <span className="tag good">Adjusted by your teacher</span>}
          </div>
        </div>
        <div className="cats">{groups.map((g) => <CategoryCard key={g.id} g={g} />)}</div>
      </div>

      {r.review?.comment && (
        <div className="card row" style={{ gap: 14, alignItems: "flex-start" }}>
          <div className="avatar">{(r.review.reviewer[0] ?? "T").toUpperCase()}</div>
          <div className="col" style={{ gap: 4 }}>
            <b>Note from {r.review.reviewer}</b>
            <span style={{ fontSize: 14.5, lineHeight: 1.5 }}>{r.review.comment}</span>
          </div>
        </div>
      )}

      <div className="card col" style={{ gap: 14 }}>
        <div className="between" style={{ flexWrap: "wrap" }}>
          <div className="col" style={{ gap: 3 }}>
            <h2>{teacher ? "The performance" : "Your Performance"}</h2>
            <span className="muted small">Each bar is a note. Outlines show notes that were expected but not played right; filled bars show what was heard.</span>
          </div>
          <RollLegend />
        </div>
        <NoteRoll roll={r.roll} rowH={7} showKeys={false} />
        {r.roll.outOfRange > 0 && <span className="ai-note">{r.roll.outOfRange} played note{r.roll.outOfRange > 1 ? "s" : ""} far outside the song's range {r.roll.outOfRange > 1 ? "are" : "is"} drawn at the top or bottom edge.</span>}
      </div>

      <div className="grid2">
        <div className="card col" style={{ gap: 14 }}>
          <div className="between"><h2>What went well</h2><span className="ai-note" style={{ fontWeight: 700, textTransform: "uppercase", letterSpacing: ".04em" }}>AI feedback</span></div>
          {wins.length === 0 ? <span className="muted small">Keep practising: your strengths will show up here.</span> : wins.map((w) => (
            <div className="fb" key={w.title}>
              <span className="fb-ico win"><Icon name="check" size={13} stroke={3} /></span>
              <div><b>{w.title}</b><p>{w.detail}</p></div>
            </div>
          ))}
        </div>
        <div className="card col" style={{ gap: 14 }}>
          <div className="between"><h2>Focus areas</h2><span className="ai-note" style={{ fontWeight: 700, textTransform: "uppercase", letterSpacing: ".04em" }}>AI feedback</span></div>
          {tips.length === 0 ? <span className="muted small">Nothing specific to fix. Well done!</span> : tips.slice(0, 5).map((t, i) => (
            <div className="fb" key={i} style={{ cursor: t.bar ? "pointer" : undefined }} onClick={() => t.bar && go("analysis", id, String(t.bar))}>
              <span className="fb-ico focus"><Icon name="arrow" size={13} stroke={2.6} /></span>
              <div style={{ flex: 1 }}><p style={{ color: "var(--ink)", fontSize: 14.5, margin: 0 }}>{t.text}</p></div>
              {t.bar && <span style={{ fontSize: 12.5, fontWeight: 700, color: "var(--brand)", whiteSpace: "nowrap" }}>Bar {t.bar}</span>}
            </div>
          ))}
        </div>
      </div>
      <span className="ai-note">Scores and feedback come from an AI assessment assistant to support practice. {teacher ? "" : "Your teacher has reviewed this assessment."}</span>
    </>
  );
}
