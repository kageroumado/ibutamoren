#!/usr/bin/env python3
"""Local dev server for Ibutamoren.

Plain `http.server` plus the two headers that make the page cross-origin
isolated (so ONNX Runtime's WASM backend can use threads) and correct MIME
types for .mjs/.wasm/.onnx. Serves this directory:

    python3 serve.py [port]   →  http://127.0.0.1:8613/
"""
import http.server
import os
import socketserver
import sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8613
ROOT = os.path.dirname(os.path.abspath(__file__))


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        '.mjs': 'text/javascript',
        '.wasm': 'application/wasm',
        '.onnx': 'application/octet-stream',
        '.avif': 'image/avif',
        '.heic': 'image/heic',
        '.woff2': 'font/woff2',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT, **kwargs)

    def end_headers(self):
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        self.send_header('Cache-Control', 'no-cache')
        super().end_headers()


if __name__ == '__main__':
    with socketserver.TCPServer(('127.0.0.1', PORT), Handler) as httpd:
        print(f'http://127.0.0.1:{PORT}/')
        httpd.serve_forever()
