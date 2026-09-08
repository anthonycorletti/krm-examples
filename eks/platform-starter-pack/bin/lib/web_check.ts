// Optional local browser smoke check. Uses installed Chrome and Bun, no browser package.
import { mkdtemp, rm, mkdir, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import { X509Certificate, createHash } from "node:crypto";

const root = resolve(import.meta.dir, "../..");
const profile = await mkdtemp(`${tmpdir()}/platform-browser-`);
const output = `${root}/.local/qa`;
await mkdir(output, { recursive: true });
// Pin only this project certificate in the temporary browser; no trust-store changes.
const cert = new X509Certificate(
  await readFile(`${root}/.local/tls/localhost.crt`),
);
const spki = createHash("sha256")
  .update(cert.publicKey.export({ type: "spki", format: "der" }))
  .digest("base64");
const chrome = Bun.spawn(
  [
    process.env.CHROME_BIN ||
      "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "--headless=new",
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-background-networking",
    `--ignore-certificate-errors-spki-list=${spki}`,
    "--remote-debugging-port=0",
    `--user-data-dir=${profile}`,
    "about:blank",
  ],
  { stdout: "ignore", stderr: Bun.file(`${output}/chrome.log`) },
);
let socket: WebSocket | undefined;
try {
  let port = "";
  for (let i = 0; !port; i++) {
    if (i > 100) throw new Error("Chrome did not start");
    try {
      port = (await readFile(`${profile}/DevToolsActivePort`, "utf8")).split(
        "\n",
      )[0];
    } catch {
      await Bun.sleep(100);
    }
  }
  console.log("Chrome started; connecting to its temporary page.");
  const pages = await (
    await fetch(`http://127.0.0.1:${port}/json/list`, {
      signal: AbortSignal.timeout(10000),
    })
  ).json();
  socket = new WebSocket(
    pages.find((p: any) => p.type === "page").webSocketDebuggerUrl,
  );
  await new Promise<void>((ok, fail) => {
    socket!.onopen = () => ok();
    socket!.onerror = fail;
  });
  console.log("Browser connection established.");
  let id = 0;
  const pending = new Map<
    number,
    { ok: (value: any) => void; fail: (err: Error) => void }
  >();
  const errors: string[] = [];
  socket.onmessage = (event) => {
    const message = JSON.parse(String(event.data));
    if (message.method === "Runtime.exceptionThrown")
      errors.push(message.params.exceptionDetails.text);
    const call = pending.get(message.id);
    if (call) {
      pending.delete(message.id);
      message.error
        ? call.fail(new Error(message.error.message))
        : call.ok(message.result);
    }
  };
  const send = (method: string, params = {}): Promise<any> => {
    const key = ++id;
    return new Promise((ok, fail) => {
      const timer = setTimeout(() => {
        pending.delete(key);
        fail(new Error(`Timed out: ${method}`));
      }, 15000);
      pending.set(key, {
        ok: (v) => {
          clearTimeout(timer);
          ok(v);
        },
        fail: (e) => {
          clearTimeout(timer);
          fail(e);
        },
      });
      socket!.send(JSON.stringify({ id: key, method, params }));
    });
  };
  const evaluate = async (expression: string) => {
    const result = await send("Runtime.evaluate", {
      expression,
      returnByValue: true,
    });
    if (result.exceptionDetails) throw new Error(result.exceptionDetails.text);
    return result.result.value;
  };
  const wait = async (expression: string, attempts = 100) => {
    for (let i = 0; i < attempts; i++) {
      if (await evaluate(expression)) return;
      await Bun.sleep(150);
    }
    throw new Error(`UI condition did not become true: ${expression}`);
  };
  const screenshot = async (name: string) => {
    const shot = await send("Page.captureScreenshot", { format: "png" });
    await Bun.write(`${output}/${name}.png`, Buffer.from(shot.data, "base64"));
  };
  console.log("Opening local workspace.");
  await send("Runtime.enable");
  await send("Page.enable");
  await send("Emulation.setDeviceMetricsOverride", {
    width: 1440,
    height: 1000,
    deviceScaleFactor: 1,
    mobile: false,
  });
  await send("Page.navigate", { url: "https://web.localhost:5173" });
  await wait(
    "Array.from(document.querySelectorAll('button')).some(b => b.textContent.trim() === 'Sign in')",
  );
  await evaluate(
    "Array.from(document.querySelectorAll('button')).find(b => b.textContent.trim() === 'Sign in').click()",
  );
  await wait("!!document.querySelector('#username')");
  console.log("Keycloak sign-in form loaded.");
  const credentials = Bun.spawn(
    [
      "kubectl",
      "--context",
      "colima",
      "-n",
      "platform-cluster",
      "get",
      "secret",
      "keycloak-users",
      "-o",
      "jsonpath={.data.KEYCLOAK_DEVELOPER_PASSWORD}",
    ],
    { stdout: "pipe", stderr: "inherit" },
  );
  const encoded = await new Response(credentials.stdout).text();
  if (await credentials.exited)
    throw new Error("Could not read local demo credentials");
  await evaluate("document.querySelector('#username').focus()");
  await send("Input.insertText", { text: "developer" });
  await evaluate("document.querySelector('#password').focus()");
  await send("Input.insertText", {
    text: Buffer.from(encoded, "base64").toString(),
  });
  await evaluate("document.querySelector('#password').form.requestSubmit()");
  await wait("!!document.querySelector('.sidebar')");
  const project = await evaluate(
    "Array.from(document.querySelectorAll('a')).find(a => a.textContent.includes('Platform walkthrough'))?.getAttribute('href')",
  );
  if (!project)
    throw new Error(
      "Run bin/local-experiment first to create walkthrough evidence",
    );
  await evaluate(
    `document.querySelector('a[href=' + CSS.escape(${JSON.stringify(project)}) + ']').click()`,
  );
  await wait(
    "Array.from(document.querySelectorAll('a')).some(a => a.textContent.includes('Inspect the running platform'))",
  );
  const task = await evaluate(
    "Array.from(document.querySelectorAll('a')).find(a => a.textContent.includes('Inspect the running platform')).getAttribute('href')",
  );
  await evaluate(
    `document.querySelector('a[href=' + CSS.escape(${JSON.stringify(task)}) + ']').click()`,
  );
  await wait("document.querySelectorAll('.run').length >= 2");
  console.log("Project, task, conversation, and run evidence rendered.");
  for (const theme of ["light", "dark", "system"]) {
    await evaluate(
      `(() => { const select = document.querySelector('select[aria-label="Appearance"]'); select.value = ${JSON.stringify(theme)}; select.dispatchEvent(new Event('change', { bubbles: true })); })()`,
    );
    await wait(
      `document.querySelector('select[aria-label="Appearance"]').value === ${JSON.stringify(theme)}`,
    );
    if (theme !== "system") {
      await wait(
        `document.documentElement.classList.contains('dark') === ${theme === "dark"}`,
      );
      await screenshot(theme);
    }
  }
  await evaluate(`document.querySelector('a[href="/?view=system"]').click()`);
  await wait("!!document.querySelector('button[value=export]')");
  const before = await evaluate(
    "Array.from(document.querySelectorAll('[data-export-id]')).map(e => e.dataset.exportId)",
  );
  await evaluate("document.querySelector('button[value=export]').click()");
  await wait(
    `Array.from(document.querySelectorAll('[data-export-id]')).some(e => !${JSON.stringify(before)}.includes(e.dataset.exportId) && e.textContent.includes('completed'))`,
    400,
  );
  await screenshot("warehouse");
  console.log("Export now completed through the browser.");
  await send("Emulation.setDeviceMetricsOverride", {
    width: 390,
    height: 844,
    deviceScaleFactor: 1,
    mobile: true,
  });
  await wait("window.innerWidth === 390");
  const overflow = await evaluate(
    "document.documentElement.scrollWidth > innerWidth",
  );
  await screenshot("mobile");
  if (overflow) throw new Error("Mobile viewport overflows horizontally");
  if (errors.length)
    throw new Error(`Browser runtime errors: ${errors.join(", ")}`);
  console.log(
    `Light, dark, system, and mobile checks passed. Screenshots: ${output}`,
  );
} finally {
  socket?.close();
  chrome.kill();
  await chrome.exited;
  await rm(profile, { recursive: true, force: true });
}
