import { api } from "../api";
import { ErrorNote, Icon, Loading, Meter, Ring, SongThumb, StatusPill } from "../components/ui";
import { displayScore, fmtDate, groupScores, visibleToStudent } from "../model";
import { useApp, useLoad } from "../state";

export function StudentHome() {
  const { me, go, songTitle, criteria } = useApp();
  const list = useLoad(() => (me.studentId ? api.assessments({ student_id: me.studentId, limit: 50 }) : Promise.resolve([])), [me.studentId]);

  if (!me.studentId) {
    return (
      <div className="card empty">
        Choose which student you are viewing as, in the panel on the left.
      </div>
    );
  }

  const items = list.data ?? [];
  const shared = items.filter(visibleToStudent);
  const last3 = shared.slice(0, 3);
  const avg = last3.length ? Math.round(last3.reduce((s, a) => s + (displayScore(a) ?? 0), 0) / last3.length) : null;
  const prev3 = shared.slice(3, 6);
  const prevAvg = prev3.length ? Math.round(prev3.reduce((s, a) => s + (displayScore(a) ?? 0), 0) / prev3.length) : null;

  // category averages across the shared results
  const cats = last3.length
    ? groupScores(last3[0], criteria).map((g) => {
        const vals = last3.map((a) => groupScores(a, criteria).find((x) => x.id === g.id)?.score).filter((v): v is number => v != null);
        return { label: g.label, v: vals.length ? Math.round(vals.reduce((s, v) => s + v, 0) / vals.length) : null };
      }).filter((c) => c.v != null)
    : [];

  return (
    <>
      <div className="head">
        <div className="col">
          <span className="muted" style={{ fontSize: 13, fontWeight: 600 }}>Welcome back, {me.studentId}</span>
          <h1 style={{ fontSize: 34 }}>Keyboard Assessment</h1>
          <span className="sub">Track your progress and improve your performance.</span>
        </div>
        <div className="row">
          <button className="btn" onClick={() => go("history")}>View Previous Assessments</button>
          <button className="btn primary" onClick={() => go("new")}><Icon name="plus" size={16} stroke={2.2} />Start New Assessment</button>
        </div>
      </div>

      {list.error && <ErrorNote error={list.error} onRetry={list.reload} />}

      <div className="dash-grid">
        <div className="card" style={{ paddingBottom: 12 }}>
          <div className="between" style={{ paddingBottom: 12 }}>
            <h2>Recent Assessments</h2>
            {items.length > 5 && <a onClick={() => go("history")} style={{ fontSize: 13.5, fontWeight: 600 }}>See all</a>}
          </div>
          {list.loading && !list.data ? <Loading /> : items.length === 0 ? (
            <div className="empty">No assessments yet. Record yourself playing a song and upload it to start.</div>
          ) : <AssessmentRows items={items.slice(0, 5)} />}
        </div>

        <div className="card col" style={{ gap: 20 }}>
          <h2>Performance Overview</h2>
          <div className="row" style={{ gap: 18 }}>
            <Ring value={avg} size={96} width={8}>
              <span style={{ fontSize: 24, fontWeight: 800 }}>{avg ?? "–"}</span>
            </Ring>
            <div className="col" style={{ gap: 4 }}>
              <span style={{ fontSize: 15, fontWeight: 700 }}>Overall progress</span>
              <span className="muted" style={{ fontSize: 13, lineHeight: 1.45 }}>
                {avg == null ? "Your progress appears here once your teacher has checked an assessment."
                  : `Average of your last ${last3.length} checked assessment${last3.length > 1 ? "s" : ""}.`}
                {avg != null && prevAvg != null && ` ${avg >= prevAvg ? "Up" : "Down"} from ${prevAvg} before.`}
              </span>
            </div>
          </div>
          {cats.length > 0 && (
            <div className="col" style={{ gap: 14, borderTop: "1px solid var(--line-2)", paddingTop: 4 }}>
              {cats.map((c) => (
                <div key={c.label} className="col" style={{ paddingTop: 10 }}>
                  <div className="between" style={{ fontSize: 14 }}><span style={{ fontWeight: 600 }}>{c.label}</span><b className="tnum">{c.v}%</b></div>
                  <Meter value={c.v} />
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </>
  );

  function AssessmentRows({ items }: { items: typeof list.data & {} }) {
    return (
      <>
        {items.map((a) => {
          const shown = visibleToStudent(a);
          const score = shown ? displayScore(a) : null;
          const target = a.status === "queued" || a.status === "processing" ? ["processing", a.assessment_id] : ["results", a.assessment_id];
          return (
            <div key={a.assessment_id} className="list-row" style={{ gridTemplateColumns: "44px minmax(0,1fr) 110px auto 84px 16px" }} onClick={() => go(...target)}>
              <SongThumb seed={a.song_id} />
              <div className="col" style={{ gap: 3, minWidth: 0 }}>
                <span style={{ fontSize: 15, fontWeight: 700 }}>{songTitle(a.song_id)}</span>
                <span className="muted" style={{ fontSize: 12.5 }}>Keyboard Performance</span>
              </div>
              {score != null ? (
                <div className="row" style={{ gap: 8 }}>
                  <div style={{ flex: 1 }}><Meter value={score} /></div>
                  <b className="tnum" style={{ fontSize: 15 }}>{score}%</b>
                </div>
              ) : <span className="muted small">–</span>}
              <StatusPill a={a} view="student" />
              <span className="muted tnum" style={{ fontSize: 13.5 }}>{fmtDate(a.created_at)}</span>
              <span style={{ color: "#9a948a" }}><Icon name="chev" size={16} stroke={2} /></span>
            </div>
          );
        })}
      </>
    );
  }
}

export function StudentHistory() {
  const { me, go, songTitle } = useApp();
  const list = useLoad(() => api.assessments({ student_id: me.studentId, limit: 200 }), [me.studentId]);
  return (
    <>
      <div className="col">
        <a className="back" onClick={() => go("home")}>← Dashboard</a>
        <h1>My results</h1>
      </div>
      {list.error && <ErrorNote error={list.error} onRetry={list.reload} />}
      <div className="card" style={{ paddingBottom: 12 }}>
        {list.loading && !list.data ? <Loading /> : (list.data ?? []).length === 0 ? <div className="empty">No assessments yet.</div> :
          (list.data ?? []).map((a) => {
            const score = visibleToStudent(a) ? displayScore(a) : null;
            return (
              <div key={a.assessment_id} className="list-row" style={{ gridTemplateColumns: "44px minmax(0,1fr) 80px 140px 110px 16px" }}
                onClick={() => go(a.status === "queued" || a.status === "processing" ? "processing" : "results", a.assessment_id)}>
                <SongThumb seed={a.song_id} />
                <span style={{ fontSize: 15, fontWeight: 700 }}>{songTitle(a.song_id)}</span>
                <b className="tnum">{score != null ? `${score}%` : "–"}</b>
                <StatusPill a={a} view="student" />
                <span className="muted tnum small">{fmtDate(a.created_at)}</span>
                <span style={{ color: "#9a948a" }}><Icon name="chev" size={16} stroke={2} /></span>
              </div>
            );
          })}
      </div>
    </>
  );
}
