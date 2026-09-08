# Routes

console.tsx loads identity, components, migration status, and verification evidence through the generated SDK. Its action executes a bounded verification or disconnects the session. Fetcher completion triggers route revalidation. Request signals are passed to SDK calls.

No endpoint types or fetch implementations are duplicated here. Run bin/web-typecheck and bin/web-build.

The agent picker, task view, and System page expose model configuration. The
inspector uses Pydantic AI TestModel: tools run against the platform, but no
external LLM is called and token counts are synthetic. Provider-backed runs do
not yet persist model identity; the UI distinguishes current configuration from
historical run evidence and marks those run models as not recorded.
