import { api, type Assessment } from "../api";
import { Garden, Star, Stars, starsFor } from "../components/kid";
import { ErrorNote, Loading, SongThumb } from "../components/ui";
import { displayScore, fmtDate, groupScores, visibleToStudent, type GroupScore } from "../model";
import { useApp, useLoad } from "../state";

/** What a student sees in a list: stars when shared, otherwise a friendly status. */
function KidStatus({ a }: { a: Assessment }) {
  if (visibleToStudent(a)) return <Stars n={starsFor(displayScore(a))} size={20} />;
  if (a.status === "queued" || a.status === "processing") return <span className="kid-pill wait">Listening…</span>;
  if (a.status === "rejected" || a.status === "failed") return <span className="kid-pill bad">Try again</span>;
  return <span className="kid-pill wait">Teacher checking</span>;
}

function SongRows({ items }: { items: Assessment[] }) {
  const { go, songTitle } = useApp();
  return (
    <>
      {items.map((a) => (
        <div key={a.assessment_id} className="list-row kid-row" style={{ gridTemplateColumns: "44px minmax(0,1fr) auto 90px" }}
          onClick={() => go(a.status === "queued" || a.status === "processing" ? "processing" : "results", a.assessment_id)}>
          <SongThumb seed={a.song_id} />
          <b style={{ fontSize: 15.5 }}>{songTitle(a.song_id)}</b>
          <KidStatus a={a} />
          <span className="muted tnum small">{fmtDate(a.created_at)}</span>
        </div>
      ))}
    </>
  );
}

export function StudentHome() {
  const { me, go, criteria } = useApp();
  const list = useLoad(() => (me.studentId ? api.assessments({ student_id: me.studentId, limit: 50 }) : Promise.resolve([])), [me.studentId]);

  if (!me.studentId) {
    return <div className="card empty">Choose which student you are viewing as, in the panel on the left.</div>;
  }

  const items = list.data ?? [];
  const shared = items.filter(visibleToStudent);
  const totalStars = shared.reduce((s, a) => s + starsFor(displayScore(a)), 0);

  // the garden shows the average of the last three shared results, skill by skill
  const last3 = shared.slice(0, 3);
  const garden: GroupScore[] = last3.length
    ? groupScores(last3[0], criteria).map((g) => {
        const vals = last3.map((a) => groupScores(a, criteria).find((x) => x.id === g.id)?.score).filter((v): v is number => v != null);
        return { ...g, score: vals.length ? Math.round(vals.reduce((s, v) => s + v, 0) / vals.length) : null };
      })
    : [];

  return (
    <div className="kid col" style={{ gap: 24 }}>
      <div className="head">
        <div className="col">
          <h1 className="kid-title" style={{ textAlign: "left" }}>Hi {me.studentId}!</h1>
          <span className="kid-sub" style={{ textAlign: "left", margin: 0 }}>Ready to play some music today?</span>
        </div>
        <button className="btn kid-btn" onClick={() => go("new")}>Play a song</button>
      </div>

      {list.error && <ErrorNote error={list.error} onRetry={list.reload} />}

      <div className="dash-grid">
        <div className="card col" style={{ gap: 6 }}>
          <h2 className="kid-h2">My music garden</h2>
          {garden.length ? (
            <>
              <span className="muted small">Your flowers grow every time you practise.</span>
              <Garden groups={garden} size={72} />
            </>
          ) : (
            <span className="kid-sub" style={{ textAlign: "left", margin: "8px 0" }}>Play your first song and watch your garden grow!</span>
          )}
        </div>
        <div className="card col star-box">
          <Star on size={64} />
          <span className="star-count tnum">{totalStars}</span>
          <span className="kid-sub" style={{ margin: 0 }}>stars collected</span>
        </div>
      </div>

      <div className="card" style={{ paddingBottom: 12 }}>
        <div className="between" style={{ paddingBottom: 12 }}>
          <h2 className="kid-h2">My songs</h2>
          {items.length > 5 && <a onClick={() => go("history")} style={{ fontSize: 14, fontWeight: 700 }}>See all</a>}
        </div>
        {list.loading && !list.data ? <Loading /> : items.length === 0 ? (
          <div className="empty">No songs yet. Record yourself playing and send it in!</div>
        ) : <SongRows items={items.slice(0, 5)} />}
      </div>
    </div>
  );
}

export function StudentHistory() {
  const { me, go } = useApp();
  const list = useLoad(() => api.assessments({ student_id: me.studentId, limit: 200 }), [me.studentId]);
  return (
    <div className="kid col" style={{ gap: 20 }}>
      <div className="col">
        <a className="back" onClick={() => go("home")}>← Home</a>
        <h1 className="kid-title" style={{ textAlign: "left" }}>All my songs</h1>
      </div>
      {list.error && <ErrorNote error={list.error} onRetry={list.reload} />}
      <div className="card" style={{ paddingBottom: 12 }}>
        {list.loading && !list.data ? <Loading /> : (list.data ?? []).length === 0 ? <div className="empty">No songs yet.</div> : <SongRows items={list.data ?? []} />}
      </div>
    </div>
  );
}
