// Head files: JSON with base64 weights. Kept apart from index.ts so the browser entry can load heads without
// pulling in the encoder.
import type { Head } from "./index.ts";

/** Serialises a head to JSON with base64 weights, for a static file next to the app. */
export function saveHead(head: Head): string {
  const b64 = (a: Float32Array) => {
    const bytes = new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
    let s = "";
    for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
    return btoa(s);
  };
  return JSON.stringify({ ...head, weights: b64(head.weights), bias: b64(head.bias) });
}

export function loadHead<L extends string = string>(json: string): Head<L> {
  const raw = JSON.parse(json);
  const f32 = (s: string) => new Float32Array(Uint8Array.from(atob(s), (c) => c.charCodeAt(0)).buffer);
  return { ...raw, weights: f32(raw.weights), bias: f32(raw.bias) };
}
