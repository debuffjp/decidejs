// Static server for the spike: node serve.mjs [port]
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { extname, join } from "node:path";
const types = { ".html": "text/html", ".js": "text/javascript" };
const root = new URL(".", import.meta.url).pathname.replace(/^\/(\w:)/, "$1");
createServer(async (req, res) => {
  const path = join(root, decodeURIComponent(new URL(req.url, "http://x").pathname));
  let body;
  try { body = await readFile(path); } catch { return res.writeHead(404).end(); }
  res.writeHead(200, { "cross-origin-opener-policy": "same-origin", "cross-origin-embedder-policy": "credentialless", "content-type": types[extname(path)] || "application/octet-stream" }).end(body);
}).listen(Number(process.argv[2] || 5199), () => console.log("serving on " + (process.argv[2] || 5199)));
