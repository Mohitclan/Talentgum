import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Development: `npm run dev` serves on :5173 and forwards /v1 to `ras serve` on :8000.
// Production: `npm run build` writes into ras/service/app, which the API serves at /app/.
export default defineConfig({
  base: "/app/",
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/v1": process.env.RAS_API ?? "http://127.0.0.1:8000" },
  },
  build: { outDir: "../ras/service/app", emptyOutDir: true },
});
