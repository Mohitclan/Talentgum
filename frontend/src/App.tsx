import { useEffect, useState } from "react";
import { api, getApiKey, setApiKey } from "./api";
import { Icon } from "./components/ui";
import { Analysis } from "./pages/Analysis";
import { NewAssessment } from "./pages/NewAssessment";
import { Processing } from "./pages/Processing";
import { Results } from "./pages/Results";
import { StudentHistory, StudentHome } from "./pages/StudentHome";
import { TeacherHome } from "./pages/TeacherHome";
import { initials } from "./model";
import { Redirect, useApp, useLoad } from "./state";

export function App() {
  const { route, me, go } = useApp();
  const [page, a, b] = route;

  // each role has its own home; keep the URL meaningful
  useEffect(() => {
    if (!page) go(me.role === "teacher" ? "teacher" : "home");
  }, [page, me.role, go]);

  let body: JSX.Element | null = null;
  if (page === "home") body = me.role === "teacher" ? <Redirect to={["teacher"]} /> : <StudentHome />;
  else if (page === "history") body = <StudentHistory />;
  else if (page === "new") body = <NewAssessment />;
  else if (page === "processing" && a) body = <Processing key={a} id={a} />;
  else if (page === "results" && a) body = <Results key={a} id={a} />;
  else if (page === "analysis" && a) body = <Analysis key={a} id={a} bar={b ? Number(b) : undefined} />;
  else if (page === "teacher") body = me.role === "teacher" ? <TeacherHome selected={a} /> : <Redirect to={["home"]} />;
  else if (page === "settings") body = <Settings />;

  return (
    <div className="shell">
      <Sidebar />
      <main className="main">{body}</main>
    </div>
  );
}

function Sidebar() {
  const { route, go, me, setMe } = useApp();
  const page = route[0];
  const studentPages = ["home", "history", "new"];
  const students = useLoad(() => (me.role === "student" ? api.assessments({ limit: 200 }) : Promise.resolve([])), [me.role]);
  const ids = [...new Set((students.data ?? []).map((x) => x.student_id).filter(Boolean) as string[])].sort();

  const switchRole = (role: "student" | "teacher") => {
    setMe({ role });
    go(role === "teacher" ? "teacher" : "home");
  };

  return (
    <aside className="side">
      <div className="brand">
        <div className="brand-mark"><i /><i /><i /></div>
        <div><b>TalentGum</b><span>Keyboard Assessment</span></div>
      </div>

      {me.role === "student" ? (
        <>
          <div className="nav-h">Student</div>
          <nav className="nav">
            <button className={page === "home" ? "on" : ""} onClick={() => go("home")}><Icon name="home" />Home</button>
            <button className={page === "new" ? "on" : ""} onClick={() => go("new")}><Icon name="newc" />Play a song</button>
            <button className={page === "history" ? "on" : ""} onClick={() => go("history")}><Icon name="chart" />My songs</button>
          </nav>
        </>
      ) : (
        <>
          <div className="nav-h">Teacher</div>
          <nav className="nav">
            <button className={page === "teacher" || (!studentPages.includes(page) && page !== "settings") ? "on" : ""} onClick={() => go("teacher")}><Icon name="people" />Student performance</button>
            <button className={page === "new" ? "on" : ""} onClick={() => go("new")}><Icon name="newc" />New assessment</button>
            <button onClick={() => { window.location.href = "/ui/"; }}><Icon name="tools" />Songs &amp; rubrics</button>
          </nav>
        </>
      )}

      <div style={{ flex: 1, minHeight: 24 }} />
      <div className="who">
        <div className="seg" style={{ width: "100%" }}>
          <button className={me.role === "student" ? "on" : ""} style={{ flex: 1 }} onClick={() => switchRole("student")}>Student</button>
          <button className={me.role === "teacher" ? "on" : ""} style={{ flex: 1 }} onClick={() => switchRole("teacher")}>Teacher</button>
        </div>
        {me.role === "student" ? (
          <div className="col" style={{ gap: 4 }}>
            <label className="lbl" htmlFor="me-sid">Viewing as student</label>
            <input id="me-sid" className="field" list="known-students" placeholder="Student ID" value={me.studentId}
              onChange={(e) => setMe({ studentId: e.target.value.trim() })} />
            <datalist id="known-students">{ids.map((s) => <option key={s} value={s} />)}</datalist>
          </div>
        ) : (
          <div className="who-row">
            <div className="avatar">{initials(me.teacherName || "T")}</div>
            <div className="col" style={{ gap: 0, minWidth: 0 }}><b style={{ fontSize: 13.5 }}>{me.teacherName || "Teacher"}</b><span className="muted" style={{ fontSize: 12 }}>Teacher</span></div>
          </div>
        )}
        <button className="linkbtn small" style={{ textAlign: "left", display: "flex", gap: 6, alignItems: "center" }} onClick={() => go("settings")}>
          <Icon name="key" size={14} />Connection settings
        </button>
      </div>
    </aside>
  );
}

function Settings() {
  const { me, setMe, go, reloadSongs, clearAuthError } = useApp();
  const [key, setKey] = useState(getApiKey());
  const health = useLoad(() => api.health(), []);
  const [saved, setSaved] = useState(false);
  return (
    <div className="col" style={{ gap: 20, maxWidth: 640 }}>
      <div className="col"><a className="back" onClick={() => go(me.role === "teacher" ? "teacher" : "home")}>← Back</a><h1>Connection settings</h1></div>
      <div className="card col" style={{ gap: 14 }}>
        <div className="between">
          <b>Assessment server</b>
          {health.data ? <span className="pill ok">Connected</span> : health.error ? <span className="pill bad">Not reachable</span> : <span className="pill wait">Checking</span>}
        </div>
        {health.data && <span className="muted small">Transcriber: {health.data.transcriber} · Access: {health.data.auth === "keys" ? "API key required" : "open (development)"}</span>}
        <div className="col">
          <label className="lbl" htmlFor="key">API key (X-API-Key)</label>
          <input id="key" className="field" type="password" value={key} onChange={(e) => { setKey(e.target.value); setSaved(false); }} placeholder="Leave empty when the server runs open" />
        </div>
        <span className="muted small">Kept only in this browser. Create a key on the server with <code>ras tenant-create</code>.</span>
        <div className="row">
          <button className="btn primary" onClick={() => { setApiKey(key.trim()); clearAuthError(); reloadSongs(); health.reload(); setSaved(true); }}>Save</button>
          {saved && <span className="muted small">Saved.</span>}
        </div>
      </div>
      <div className="card col" style={{ gap: 10 }}>
        <b>Teacher name</b>
        <input className="field" value={me.teacherName} onChange={(e) => setMe({ teacherName: e.target.value })} />
        <span className="muted small">Recorded with each review and shown to the student.</span>
      </div>
    </div>
  );
}
