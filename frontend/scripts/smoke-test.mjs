import { readFileSync } from "node:fs";
import { dirname } from "node:path";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");

function read(path) {
  return readFileSync(resolve(root, path), "utf8");
}

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

const api = read("src/lib/api.ts");
assert(
  api.includes("VITE_API_BASE_URL") && api.includes("http://127.0.0.1:8000"),
  "src/lib/api.ts should keep the FastAPI base URL fallback."
);

const kgExplorer = read("src/components/KnowledgeGraphExplorer.tsx");
assert(
  kgExplorer.includes('const DEFAULT_QUERY = "Pants"'),
  "KnowledgeGraphExplorer should keep Pants as the default query."
);
assert(
  kgExplorer.includes("http://127.0.0.1:5000/search/"),
  "KnowledgeGraphExplorer should call the local Flask KG search service."
);

const envExample = read(".env.example");
assert(
  envExample.includes("VITE_API_BASE_URL=http://127.0.0.1:8000"),
  ".env.example should document the FastAPI base URL."
);

console.log("Frontend smoke tests passed.");
