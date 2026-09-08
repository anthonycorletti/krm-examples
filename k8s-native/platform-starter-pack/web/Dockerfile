FROM oven/bun:1.4.0-slim AS build
WORKDIR /web
COPY package.json bun.lock ./
RUN bun install --frozen-lockfile
COPY . .
RUN bun run build

FROM oven/bun:1.4.0-slim
WORKDIR /web
COPY --from=build /web/dist ./dist
COPY serve.ts ./serve.ts
USER bun
EXPOSE 8080
CMD ["bun", "serve.ts"]
