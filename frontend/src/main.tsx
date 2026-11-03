import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { AppProvider, useApp } from "./state";
import "./styles.css";

function AuthBanner() {
  const { authError, go } = useApp();
  if (!authError) return null;
  return (
    <div className="note-banner bad" style={{ borderRadius: 0, justifyContent: "center" }}>
      The assessment server needs an API key. <a onClick={() => go("settings")}>Add it in connection settings</a>.
    </div>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <AppProvider>
      <AuthBanner />
      <App />
    </AppProvider>
  </StrictMode>,
);
