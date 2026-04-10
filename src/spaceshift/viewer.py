import os
import json
import socket
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs


def _parse_frontmatter(text):
    """Parse YAML frontmatter from markdown. Returns (meta_dict, body_string)."""
    if not text.startswith("---"):
        return {}, text
    end = text.find("---", 3)
    if end == -1:
        return {}, text
    raw = text[3:end].strip()
    body = text[end + 3:].lstrip("\n")
    meta = {}
    current_key = None
    multiline_lines = []
    for line in raw.splitlines():
        if current_key and line.startswith("  "):
            multiline_lines.append(line[2:])
        else:
            if current_key and multiline_lines:
                meta[current_key] = "\n".join(multiline_lines)
                multiline_lines = []
                current_key = None
            if ": |" == line[-3:] if len(line) >= 3 else False:
                current_key = line[:-3].strip()
            elif ": " in line:
                k, v = line.split(": ", 1)
                meta[k.strip()] = v.strip()
            elif line.endswith(":"):
                current_key = line[:-1].strip()
    if current_key and multiline_lines:
        meta[current_key] = "\n".join(multiline_lines)
    return meta, body


def _collect_files(root):
    """Walk root and return list of .md and .svg files with metadata."""
    files = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if not (fname.endswith(".md") or fname.endswith(".svg")):
                continue
            full = os.path.join(dirpath, fname)
            rel = os.path.relpath(full, root)
            stat = os.stat(full)
            files.append({
                "path": rel,
                "size": stat.st_size,
                "modified": stat.st_mtime,
            })
    files.sort(key=lambda f: f["path"])
    return files


def _safe_path(root, requested):
    """Resolve requested path under root, reject traversal."""
    joined = os.path.normpath(os.path.join(root, requested))
    if not joined.startswith(os.path.normpath(root)):
        return None
    if ".." in os.path.relpath(joined, root):
        return None
    return joined


_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>spaceshift viewer</title>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; display: flex; height: 100vh; background: #fff; color: #1a1a1a; }

/* Sidebar */
.sidebar { width: 280px; min-width: 280px; background: #1a1a1a; color: #ccc; overflow-y: auto; display: flex; flex-direction: column; }
.sidebar-header { padding: 20px 16px 12px; border-bottom: 1px solid #333; }
.sidebar-header h1 { font-size: 14px; font-weight: 600; color: #fff; letter-spacing: 0.5px; text-transform: uppercase; }
.sidebar-header .path { font-size: 11px; color: #888; margin-top: 4px; word-break: break-all; }
.file-list { flex: 1; overflow-y: auto; padding: 8px 0; }
.group-label { font-size: 10px; font-weight: 600; color: #666; text-transform: uppercase; letter-spacing: 0.5px; padding: 12px 16px 4px; }
.file-item { padding: 6px 16px; cursor: pointer; font-size: 13px; color: #aaa; transition: background 0.1s, color 0.1s; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.file-item:hover { background: #252525; color: #ddd; }
.file-item.active { background: #2a2a2a; color: #fff; border-left: 2px solid #5b8def; padding-left: 14px; }
.file-item .ext { color: #555; font-size: 11px; }

/* Content */
.content { flex: 1; overflow-y: auto; padding: 40px 60px; max-width: 900px; }
.content.wide { max-width: 100%; }
.meta-card { background: #f7f8fa; border: 1px solid #e4e6ea; border-radius: 8px; padding: 16px 20px; margin-bottom: 28px; }
.meta-row { display: flex; gap: 8px; margin-bottom: 6px; font-size: 13px; line-height: 1.5; }
.meta-row:last-child { margin-bottom: 0; }
.meta-key { color: #666; font-weight: 600; min-width: 80px; flex-shrink: 0; }
.meta-val { color: #1a1a1a; word-break: break-word; }
.meta-val.score { font-weight: 600; }

/* Markdown */
.md-body { line-height: 1.7; font-size: 15px; }
.md-body h1, .md-body h2, .md-body h3, .md-body h4 { margin-top: 1.4em; margin-bottom: 0.6em; font-weight: 600; }
.md-body h1 { font-size: 1.6em; border-bottom: 1px solid #eee; padding-bottom: 0.3em; }
.md-body h2 { font-size: 1.3em; }
.md-body h3 { font-size: 1.1em; }
.md-body p { margin-bottom: 0.8em; }
.md-body ul, .md-body ol { margin-bottom: 0.8em; padding-left: 1.5em; }
.md-body li { margin-bottom: 0.3em; }
.md-body code { background: #f0f1f3; padding: 2px 5px; border-radius: 3px; font-size: 0.9em; }
.md-body pre { background: #f6f7f9; border: 1px solid #e4e6ea; border-radius: 6px; padding: 14px 18px; overflow-x: auto; margin-bottom: 1em; }
.md-body pre code { background: none; padding: 0; }
.md-body blockquote { border-left: 3px solid #ddd; padding-left: 16px; color: #555; margin-bottom: 0.8em; }
.md-body hr { border: none; border-top: 1px solid #eee; margin: 1.5em 0; }
.md-body table { border-collapse: collapse; margin-bottom: 1em; width: 100%; }
.md-body th, .md-body td { border: 1px solid #ddd; padding: 8px 12px; text-align: left; font-size: 14px; }
.md-body th { background: #f7f8fa; font-weight: 600; }
.md-body strong { font-weight: 600; }

.svg-container { text-align: center; }
.svg-container svg { max-width: 100%; height: auto; }

.empty { color: #999; font-size: 14px; padding: 40px; }
</style>
</head>
<body>
<div class="sidebar">
  <div class="sidebar-header">
    <h1>spaceshift</h1>
    <div class="path" id="root-path"></div>
  </div>
  <div class="file-list" id="file-list"></div>
</div>
<div class="content" id="content">
  <div class="empty">Loading files...</div>
</div>

<script>
const DIRECTION_ORDER = { root: 0, sub: 1, super: 2, side: 3 };

function sortKey(path) {
  const name = path.split('/').pop().replace(/\.\w+$/, '');
  // research_tree: direction_L{depth}_{index}
  const treeMatch = name.match(/^(root|sub|super|side)(?:_L(\d+)_(\d+))?$/);
  if (treeMatch) {
    const dir = DIRECTION_ORDER[treeMatch[1]] ?? 9;
    const depth = parseInt(treeMatch[2] || '0');
    const idx = parseInt(treeMatch[3] || '0');
    return `A_${dir}_${String(depth).padStart(3,'0')}_${String(idx).padStart(4,'0')}`;
  }
  // grid_search ranked: name_rank_transform__model
  const rankMatch = name.match(/_(\d+)_/);
  if (rankMatch) {
    return `B_${String(parseInt(rankMatch[1])).padStart(4,'0')}_${name}`;
  }
  return `C_${name}`;
}

function groupFiles(files) {
  const groups = {};
  for (const f of files) {
    const parts = f.path.split('/');
    const group = parts.length > 1 ? parts.slice(0, -1).join('/') : '.';
    if (!groups[group]) groups[group] = [];
    groups[group].push(f);
  }
  for (const g of Object.keys(groups)) {
    groups[g].sort((a, b) => sortKey(a.path).localeCompare(sortKey(b.path)));
  }
  return groups;
}

function renderFileList(files) {
  const list = document.getElementById('file-list');
  const groups = groupFiles(files);
  const keys = Object.keys(groups).sort();
  const multiGroup = keys.length > 1 || (keys.length === 1 && keys[0] !== '.');

  list.innerHTML = '';
  for (const g of keys) {
    if (multiGroup) {
      const label = document.createElement('div');
      label.className = 'group-label';
      label.textContent = g === '.' ? 'root' : g;
      list.appendChild(label);
    }
    for (const f of groups[g]) {
      const el = document.createElement('div');
      el.className = 'file-item';
      el.dataset.path = f.path;
      const name = f.path.split('/').pop();
      const base = name.replace(/\.\w+$/, '');
      const ext = name.includes('.') ? '.' + name.split('.').pop() : '';
      el.innerHTML = `${base}<span class="ext">${ext}</span>`;
      el.onclick = () => loadFile(f.path, el);
      list.appendChild(el);
    }
  }
  // auto-select first
  const first = list.querySelector('.file-item');
  if (first) first.click();
}

async function loadFile(path, el) {
  document.querySelectorAll('.file-item').forEach(e => e.classList.remove('active'));
  if (el) el.classList.add('active');

  const content = document.getElementById('content');

  if (path.endsWith('.svg')) {
    content.className = 'content wide';
    const resp = await fetch('/api/svg?path=' + encodeURIComponent(path));
    if (!resp.ok) { content.innerHTML = '<div class="empty">Failed to load SVG</div>'; return; }
    content.innerHTML = '<div class="svg-container">' + await resp.text() + '</div>';
    return;
  }

  content.className = 'content';
  const resp = await fetch('/api/file?path=' + encodeURIComponent(path));
  if (!resp.ok) { content.innerHTML = '<div class="empty">Failed to load file</div>'; return; }
  const data = await resp.json();

  let html = '';
  if (data.meta && Object.keys(data.meta).length > 0) {
    html += '<div class="meta-card">';
    for (const [k, v] of Object.entries(data.meta)) {
      const cls = (k === 'score' || k === 'rank') ? ' score' : '';
      const display = String(v).includes('\n') ? '<pre style="margin:0;background:none;border:none;padding:0;font-size:12px;">' + escHtml(v) + '</pre>' : escHtml(v);
      html += `<div class="meta-row"><span class="meta-key">${escHtml(k)}</span><span class="meta-val${cls}">${display}</span></div>`;
    }
    html += '</div>';
  }
  html += '<div class="md-body">' + marked.parse(data.body) + '</div>';
  content.innerHTML = html;
  renderMath(content);
}

function renderMath(el) {
  // Block math: $$...$$
  el.querySelectorAll('p, div, li').forEach(node => {
    if (!node.innerHTML.includes('$$')) return;
    node.innerHTML = node.innerHTML.replace(/\$\$([\s\S]*?)\$\$/g, (_, tex) => {
      try { return katex.renderToString(tex.trim(), { displayMode: true, throwOnError: false }); }
      catch(e) { return '$$' + tex + '$$'; }
    });
  });
  // Inline math: $...$  (not inside code/pre)
  el.querySelectorAll('p, li, td, th, span, em, strong').forEach(node => {
    if (!node.innerHTML.includes('$')) return;
    if (node.closest('pre') || node.closest('code')) return;
    node.innerHTML = node.innerHTML.replace(/\$([^\$\n]+?)\$/g, (_, tex) => {
      try { return katex.renderToString(tex.trim(), { displayMode: false, throwOnError: false }); }
      catch(e) { return '$' + tex + '$'; }
    });
  });
}

function escHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

async function init() {
  document.getElementById('root-path').textContent = document.title.replace('spaceshift viewer — ', '');
  const resp = await fetch('/api/files');
  const files = await resp.json();
  if (files.length === 0) {
    document.getElementById('file-list').innerHTML = '<div class="empty">No .md or .svg files found</div>';
    document.getElementById('content').innerHTML = '<div class="empty">No files to display</div>';
    return;
  }
  renderFileList(files);
}

init();
</script>
</body>
</html>"""


def _make_handler(root):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            path = parsed.path
            params = parse_qs(parsed.query)

            if path == "/":
                self._html()
            elif path == "/api/files":
                self._files()
            elif path == "/api/file":
                self._file(params)
            elif path == "/api/svg":
                self._svg(params)
            else:
                self._send(404, "text/plain", "Not found")

        def _html(self):
            self._send(200, "text/html; charset=utf-8", _HTML)

        def _files(self):
            files = _collect_files(root)
            self._send(200, "application/json", json.dumps(files))

        def _file(self, params):
            rel = params.get("path", [None])[0]
            if not rel:
                self._send(400, "text/plain", "Missing path")
                return
            full = _safe_path(root, rel)
            if not full or not full.endswith(".md") or not os.path.isfile(full):
                self._send(403, "text/plain", "Forbidden")
                return
            with open(full, "r", encoding="utf-8") as f:
                text = f.read()
            meta, body = _parse_frontmatter(text)
            self._send(200, "application/json", json.dumps({"meta": meta, "body": body}))

        def _svg(self, params):
            rel = params.get("path", [None])[0]
            if not rel:
                self._send(400, "text/plain", "Missing path")
                return
            full = _safe_path(root, rel)
            if not full or not full.endswith(".svg") or not os.path.isfile(full):
                self._send(403, "text/plain", "Forbidden")
                return
            with open(full, "r", encoding="utf-8") as f:
                svg = f.read()
            self._send(200, "image/svg+xml", svg)

        def _send(self, code, content_type, body):
            data = body.encode("utf-8") if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, fmt, *args):
            pass  # silent

    return Handler


def view_background(path=".", port=8383):
    """Start viewer in a background daemon thread. Returns immediately after opening browser."""
    import threading

    root = os.path.abspath(path)
    if not os.path.isdir(root):
        raise ValueError(f"Not a directory: {root}")

    for attempt in range(10):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("127.0.0.1", port + attempt))
            sock.close()
            port = port + attempt
            break
        except OSError:
            continue
    else:
        raise RuntimeError(f"Could not find open port near {port}")

    handler = _make_handler(root)
    server = HTTPServer(("127.0.0.1", port), handler)
    url = f"http://127.0.0.1:{port}"

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    webbrowser.open(url)
    return url, server


def view(path=".", port=8383, no_open=False):
    """Serve a directory of markdown files in the browser.

    Args:
        path: Directory containing .md/.svg files to view.
        port: Port to serve on (auto-increments if taken).
        no_open: If True, don't auto-open the browser.
    """
    root = os.path.abspath(path)
    if not os.path.isdir(root):
        raise ValueError(f"Not a directory: {root}")

    # find open port
    for attempt in range(10):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("127.0.0.1", port + attempt))
            sock.close()
            port = port + attempt
            break
        except OSError:
            continue
    else:
        raise RuntimeError(f"Could not find open port near {port}")

    handler = _make_handler(root)
    server = HTTPServer(("127.0.0.1", port), handler)
    url = f"http://127.0.0.1:{port}"
    print(f"spaceshift viewer → {url}")
    print(f"Serving: {root}")
    print("Press Ctrl+C to stop")

    if not no_open:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
