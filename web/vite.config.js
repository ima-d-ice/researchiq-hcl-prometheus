import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev: `npm run dev` here, with the API on :8000 (uvicorn researchiq.api.server:app).
// Prod: `npm run build`, then the API server serves web/dist itself.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { "/api": "http://127.0.0.1:8000" } },
  build: { outDir: "dist", sourcemap: false },
});
