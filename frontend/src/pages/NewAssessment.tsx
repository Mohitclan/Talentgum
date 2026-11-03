import { useEffect, useRef, useState } from "react";
import { api, uploadAudio, type Song } from "../api";
import { ErrorNote, Icon } from "../components/ui";
import { useApp, useLoad } from "../state";

const ACCEPT = ".m4a,.mp3,.wav,.aac,.flac,.ogg,audio/*";
const MAX_MB = 50;

function songFacts(s: Song) {
  const ref = s.references[s.references.length - 1];
  if (!ref) return { bars: "–", tempo: "–" };
  return { bars: `${ref.measures} bars · ${ref.notes} notes`, tempo: ref.bpm ? `${Math.round(ref.bpm)} bpm` : "Not set" };
}

export function NewAssessment() {
  const { me, setMe, go, songs, reportError } = useApp();
  const rubrics = useLoad(() => api.rubrics(), []);
  const [songId, setSongId] = useState<string>("");
  const [query, setQuery] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [studentId, setStudentId] = useState(me.role === "student" ? me.studentId : "");
  const [progress, setProgress] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [over, setOver] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  useEffect(() => { if (!songId && songs.length) setSongId(songs[0].song_id); }, [songs, songId]);

  // the client's full criteria rubric when it is loaded, otherwise the newest one
  const rubric = (rubrics.data ?? []).filter((r) => r.rubric_id === "talentgum-criteria").sort((a, b) => b.version - a.version)[0]
    ?? (rubrics.data ?? [])[0];
  const song = songs.find((s) => s.song_id === songId);
  const shown = songs.filter((s) => s.title.toLowerCase().includes(query.toLowerCase()));
  const tooBig = file != null && file.size > MAX_MB * 1024 * 1024;
  const ready = song && file && !tooBig && rubric && studentId.trim() && progress == null;

  const pick = (f: File | undefined | null) => { if (f) { setFile(f); setError(null); } };

  async function begin() {
    if (!ready) return;
    setError(null);
    setProgress(0);
    try {
      const up = await uploadAudio(file!, setProgress);
      const sid = studentId.trim();
      const res = await api.submit({ song_id: song!.song_id, recording_file_id: up.file_id, rubric_id: rubric!.rubric_id, student_id: sid },
        `${up.file_id}:${song!.song_id}:${sid}`);
      if (me.role === "student" && sid !== me.studentId) setMe({ studentId: sid });
      go("processing", res.assessment_id);
    } catch (e) {
      setError(reportError(e));
      setProgress(null);
    }
  }

  return (
    <div className="col" style={{ gap: 28, maxWidth: 1040 }}>
      <div className="col">
        <a className="back" onClick={() => go("home")}>← Dashboard</a>
        <h1>Start a New Assessment</h1>
      </div>

      <div className="new-grid">
        <div className="card col" style={{ gap: 14 }}>
          <div className="row" style={{ gap: 8 }}><span className="step">1</span><h2 style={{ fontSize: 17 }}>Choose a song</h2></div>
          {songs.length > 5 && <input className="field" placeholder="Search songs" value={query} onChange={(e) => setQuery(e.target.value)} />}
          {songs.length === 0 ? (
            <div className="empty">No songs yet. A teacher adds songs (MusicXML or MIDI) in the admin tools.</div>
          ) : (
            <div className="col">
              {shown.map((s) => (
                <button key={s.song_id} className={`song-opt ${s.song_id === songId ? "on" : ""}`} onClick={() => setSongId(s.song_id)}>
                  <span className="col" style={{ gap: 2 }}>
                    <b style={{ fontSize: 14.5 }}>{s.title}</b>
                    <span className="muted" style={{ fontSize: 12.5 }}>{songFacts(s).bars}</span>
                  </span>
                  <span className="radio" />
                </button>
              ))}
            </div>
          )}
          {song && (
            <div className="facts">
              <div><span>Length</span><b>{songFacts(song).bars.split(" · ")[0]}</b></div>
              <div><span>Target tempo</span><b>{songFacts(song).tempo}</b></div>
              <div><span>Assessment type</span><b>Keyboard</b></div>
            </div>
          )}
        </div>

        <div className="col" style={{ gap: 16 }}>
          <div className="card col" style={{ gap: 16 }}>
            <div className="row" style={{ gap: 8, alignItems: "flex-start" }}>
              <span className="step">2</span>
              <div className="col" style={{ gap: 3 }}>
                <h2 style={{ fontSize: 17 }}>Upload your performance</h2>
                <span className="muted" style={{ fontSize: 14 }}>Upload an audio recording of your keyboard performance.</span>
              </div>
            </div>
            {me.role === "teacher" && (
              <div className="col">
                <label className="lbl" htmlFor="sid">Student</label>
                <input id="sid" className="field" placeholder="Student ID" value={studentId} onChange={(e) => setStudentId(e.target.value)} />
              </div>
            )}
            <div className={`drop ${over ? "over" : ""}`}
              onDragOver={(e) => { e.preventDefault(); setOver(true); }}
              onDragLeave={() => setOver(false)}
              onDrop={(e) => { e.preventDefault(); setOver(false); pick(e.dataTransfer.files[0]); }}>
              <div className="ico-box"><Icon name="upload" size={22} stroke={2} /></div>
              <b style={{ fontSize: 15 }}>Drag and drop your recording here</b>
              <span className="muted small">M4A, MP3 or WAV · one performance of the song · up to 5 minutes</span>
              <button className="btn sm" onClick={() => input.current?.click()}>Browse files</button>
              <input ref={input} type="file" accept={ACCEPT} hidden onChange={(e) => pick(e.target.files?.[0])} />
            </div>
            {file && (
              <div className="row" style={{ gap: 14, padding: "14px 16px", border: "1px solid var(--line)", borderRadius: 12 }}>
                <div className="thumb" style={{ width: 40, height: 40, alignItems: "center", padding: 0 }}>
                  {[10, 18, 13, 20].map((h, i) => <i key={i} style={{ height: h, width: 3, background: "var(--ink)", opacity: 1 }} />)}
                </div>
                <div className="col" style={{ flex: 1, gap: 7, minWidth: 0 }}>
                  <div className="between" style={{ fontSize: 14 }}>
                    <b style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{file.name}</b>
                    <span className="muted tnum">{(file.size / 1024 / 1024).toFixed(1)} MB</span>
                  </div>
                  <div className="meter"><i style={{ width: `${progress ?? 0}%`, background: tooBig ? "var(--bad)" : undefined }} /></div>
                  <span className="muted tnum" style={{ fontSize: 12.5 }}>
                    {tooBig ? `Too large: the limit is ${MAX_MB} MB` : progress == null ? "Ready to upload" : progress < 100 ? `Uploading… ${Math.round(progress)}%` : "Uploaded, starting the assessment…"}
                  </span>
                </div>
                {progress == null && <button className="linkbtn" onClick={() => setFile(null)}>Remove</button>}
              </div>
            )}
          </div>
          <div className="note-banner">
            <Icon name="info" />
            <span>Your performance will be compared with the music, note by note, for accuracy, timing and rhythm. Your teacher checks the result before you see your score.</span>
          </div>
          {error && <ErrorNote error={error} />}
          {rubrics.data && !rubric && <ErrorNote error="No scoring rubric is set up yet. A teacher needs to add one in the admin tools." />}
          <div className="row" style={{ justifyContent: "flex-end" }}>
            <button className="btn primary" disabled={!ready} onClick={begin} style={{ padding: "13px 24px", fontSize: 15 }}>Begin Assessment</button>
          </div>
        </div>
      </div>
    </div>
  );
}
