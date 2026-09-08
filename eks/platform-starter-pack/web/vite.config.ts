import { readFileSync } from "node:fs";
import { Agent } from "node:https";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { fileURLToPath, URL } from "node:url";

export default defineConfig(({ command }) => ({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) } },
  ...(command === "serve"
    ? {
        server: {
          host: "127.0.0.1",
          port: 5173,
          strictPort: true,
          origin: "https://web.localhost:5173",
          https: {
            key: readFileSync(
              new URL("../.local/tls/localhost.key", import.meta.url),
            ),
            cert: readFileSync(
              new URL("../.local/tls/localhost.crt", import.meta.url),
            ),
          },
          proxy: {
            "/api": {
              target: "https://127.0.0.1:8000",
              agent: new Agent({
                ca: readFileSync(
                  new URL("../.local/tls/ca.crt", import.meta.url),
                ),
              }),
              secure: true,
              changeOrigin: true,
            },
          },
        },
      }
    : {}),
}));
