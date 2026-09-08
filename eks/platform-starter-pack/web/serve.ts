import { join } from "node:path";

const files = new Map<string, ReturnType<typeof Bun.file>>();
for await (const path of new Bun.Glob("**/*").scan("dist")) {
  files.set(`/${path}`, Bun.file(join("dist", path)));
}
const index = files.get("/index.html");
if (!index) throw new Error("Build web assets before starting the server");

Bun.serve({
  port: 8080,
  hostname: "0.0.0.0",
  fetch(request) {
    const path = new URL(request.url).pathname;
    if (request.method !== "GET" && request.method !== "HEAD") {
      return new Response(null, { status: 405 });
    }
    if (path === "/livez") return new Response("ok");
    if (path === "/config.json") {
      return Response.json({ environment: process.env.APP_ENV ?? "local" }, {
        headers: { "Cache-Control": "no-store" },
      });
    }
    const file = files.get(path) ?? (!path.includes(".") ? index : undefined);
    if (!file) return new Response("Not found", { status: 404 });
    return new Response(file, {
      headers: {
        "Cache-Control": path.startsWith("/assets/") ? "public, max-age=31536000, immutable" : "no-cache",
        "X-Content-Type-Options": "nosniff",
        "Referrer-Policy": "same-origin",
        "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'self'; form-action 'self'",
      },
    });
  },
});
