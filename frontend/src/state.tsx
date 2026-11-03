// App-wide state: the hash route, who is viewing (no logins yet), and data every screen needs.
import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from "react";
import { api, ApiError, type Criterion, type Song } from "./api";

export type Role = "student" | "teacher";

export interface Identity {
  role: Role;
  studentId: string; // the RAS student_id the student view is for
  teacherName: string; // recorded as the reviewer
}

const ID_STORE = "tg.identity";

function loadIdentity(): Identity {
  const fallback: Identity = { role: "teacher", studentId: "", teacherName: "Teacher" };
  try {
    return { ...fallback, ...JSON.parse(localStorage.getItem(ID_STORE) ?? "{}") };
  } catch {
    return fallback;
  }
}

/** "#/results/asm_123" -> ["results", "asm_123"] */
function parseHash(): string[] {
  return window.location.hash.replace(/^#\/?/, "").split("/").filter(Boolean).map(decodeURIComponent);
}

interface Ctx {
  route: string[];
  go: (...parts: string[]) => void;
  me: Identity;
  setMe: (patch: Partial<Identity>) => void;
  criteria: Criterion[];
  songs: Song[];
  songTitle: (id: string) => string;
  reloadSongs: () => void;
  authError: boolean;
  reportError: (e: unknown) => string;
  clearAuthError: () => void;
}

const AppCtx = createContext<Ctx | null>(null);

export function AppProvider({ children }: { children: ReactNode }) {
  const [route, setRoute] = useState(parseHash);
  const [me, setMeState] = useState(loadIdentity);
  const [criteria, setCriteria] = useState<Criterion[]>([]);
  const [authError, setAuthError] = useState(false);
  const [songs, setSongs] = useState<Song[]>([]);
  const [songTick, setSongTick] = useState(0);

  useEffect(() => {
    const on = () => { setRoute(parseHash()); window.scrollTo(0, 0); };
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);

  const reportError = useCallback((e: unknown) => {
    if (e instanceof ApiError && e.status === 401) setAuthError(true);
    return e instanceof Error ? e.message : String(e);
  }, []);

  useEffect(() => { api.criteria().then(setCriteria).catch(reportError); }, [reportError, authError]);
  useEffect(() => { api.songs().then(setSongs).catch(reportError); }, [reportError, authError, songTick]);
  const songTitle = useCallback((id: string) => songs.find((s) => s.song_id === id)?.title ?? id.replace(/_/g, " "), [songs]);

  const go = useCallback((...parts: string[]) => {
    window.location.hash = "/" + parts.map(encodeURIComponent).join("/");
  }, []);

  const setMe = useCallback((patch: Partial<Identity>) => {
    setMeState((prev) => {
      const next = { ...prev, ...patch };
      try { localStorage.setItem(ID_STORE, JSON.stringify(next)); } catch { /* private mode */ }
      return next;
    });
  }, []);

  return (
    <AppCtx.Provider value={{ route, go, me, setMe, criteria, songs, songTitle, reloadSongs: () => setSongTick((t) => t + 1), authError, reportError, clearAuthError: () => setAuthError(false) }}>
      {children}
    </AppCtx.Provider>
  );
}

export function useApp(): Ctx {
  const c = useContext(AppCtx);
  if (!c) throw new Error("useApp outside AppProvider");
  return c;
}

/** Load data, reload on demand; errors are kept for display. */
export function useLoad<T>(fn: () => Promise<T>, deps: unknown[]): { data: T | null; error: string | null; loading: boolean; reload: () => void } {
  const { reportError } = useApp();
  const [state, setState] = useState<{ data: T | null; error: string | null; loading: boolean }>({ data: null, error: null, loading: true });
  const [tick, setTick] = useState(0);
  useEffect(() => {
    let live = true;
    setState((s) => ({ ...s, loading: true }));
    fn().then(
      (data) => live && setState({ data, error: null, loading: false }),
      (e) => live && setState({ data: null, error: reportError(e), loading: false }),
    );
    return () => { live = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, tick]);
  const reload = useCallback(() => setTick((t) => t + 1), []);
  return { ...state, reload };
}

/** Navigate after render (never during it). */
export function Redirect({ to }: { to: string[] }) {
  const { go } = useApp();
  const target = to.join("/");
  useEffect(() => { go(...target.split("/")); }, [go, target]);
  return null;
}
