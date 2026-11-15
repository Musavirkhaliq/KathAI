import json
import base64
import os
import random
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from pathlib import Path

WORKSPACE = Path('/home/musa/Kashmiri Project')
MANIFEST_PATH = WORKSPACE / 'kashmiri_dataset/manifests/source_manifests/manifest_quran_kashmiri.jsonl'
JUDGMENTS_LOG = WORKSPACE / 'kashmiri_dataset/manifests/human_judgments.jsonl'
HTML_TEMPLATE_PATH = Path('/home/musa/.gemini/antigravity/brain/1361f90f-1463-44e4-bf5a-a8707b9c3e7c/human_judge_ui.html')

from generate_human_eval_ui import SURAH_NAMES, sample_and_build

# Cache records in memory for instant sampling
print("Loading master manifest into memory...")
with open(MANIFEST_PATH, 'r', encoding='utf-8') as f:
    ALL_RECORDS = [json.loads(line) for line in f]
print(f"Loaded {len(ALL_RECORDS)} records from manifest.")

class HumanEvalHandler(BaseHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == '/' or parsed.path == '/index.html':
            if HTML_TEMPLATE_PATH.exists():
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.end_headers()
                with open(HTML_TEMPLATE_PATH, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"UI HTML not found.")
        elif parsed.path == '/api/sample':
            qs = parse_qs(parsed.query)
            n = int(qs.get('n', [10])[0])
            clips = sample_and_build(n)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'ok', 'clips': clips}).encode('utf-8'))
        elif parsed.path == '/api/stats':
            judgments = []
            if JUDGMENTS_LOG.exists():
                with open(JUDGMENTS_LOG, 'r', encoding='utf-8') as f:
                    judgments = [json.loads(line) for line in f if line.strip()]
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                'status': 'ok',
                'total_judgments': len(judgments),
                'total_corpus_clips': len(ALL_RECORDS)
            }).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == '/api/save_judgments':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            try:
                data = json.loads(body)
                items = data.get('judgments', [])
                with open(JUDGMENTS_LOG, 'a', encoding='utf-8') as f:
                    for item in items:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'ok', 'saved_count': len(items)}).encode('utf-8'))
            except Exception as e:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'error', 'message': str(e)}).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

def run(port=8765):
    server_address = ('0.0.0.0', port)
    httpd = HTTPServer(server_address, HumanEvalHandler)
    print(f"Human Eval UI Server running at http://localhost:{port}/")
    httpd.serve_forever()

if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    run(port)
