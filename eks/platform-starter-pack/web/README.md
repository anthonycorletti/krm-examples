# Web workspace

React, Vite, TypeScript, and generated shadcn/ui components provide a stone-themed experimentation workspace with square edges and System / Light / Dark appearance. Projects contain agent tasks, conversations, runs, and reports; the System view shows live platform probes and outstanding integrations.

React Router loaders, actions, fetchers, and revalidation use the Hey API Fetch SDK generated from FastAPI OpenAPI. No TanStack or separate request-cache dependency. Active tasks poll while the document is visible. Tokens remain in memory and clear on reload or sign-out; theme preference can persist in browser storage.

The full local platform runs through `bin/local-up` and `bin/local-forward`. Connect at https://web.localhost:5173 with `bin/local-token`. For native frontend development, `bin/web-dev` proxies the separately running API with the project CA. OIDC authorization-code/PKCE support is present but remote-provider sign-in remains unverified.

Use `bin/web-install` for the frozen lockfile, `bin/web-ui add COMPONENT` for the pinned shadcn CLI, `bin/generate-client` after API contract changes, `bin/client-check` for drift, `bin/web-format` for formatting, and `bin/web-build` for TypeScript and the production build. See [component provenance](src/components/ui/README.md).
