"""Serve only the local review package, with byte ranges for accurate video seeking."""
import argparse
import functools
import re
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def byte_range(header,size):
    match=re.fullmatch(r'bytes=(\d*)-(\d*)',header)
    if not match or not any(match.groups()) or size<=0:
        raise ValueError('Unsupported byte range')
    left,right=match.groups()
    if left:
        start=int(left); end=min(int(right),size-1) if right else size-1
    else:
        start=max(0,size-int(right)); end=size-1
    if start<0 or start>=size or end<start:
        raise ValueError('Unsatisfiable byte range')
    return start,end


class ReviewHandler(SimpleHTTPRequestHandler):
    def list_directory(self,path):
        self.send_error(404); return None

    def send_head(self):
        self.requested_range=None
        path=Path(self.translate_path(self.path))
        root=Path(self.directory).resolve()
        if path.is_dir(): path=path/'index.html'
        allowed=path.name in ['index.html','review_manifest.json'] or path.parent.name=='media'
        if not path.resolve().is_relative_to(root) or not path.is_file() or not allowed:
            self.send_error(404); return None
        stream=path.open('rb'); size=path.stat().st_size
        if self.headers.get('Range'):
            try: start,end=byte_range(self.headers['Range'],size)
            except ValueError:
                stream.close(); self.send_response(416)
                self.send_header('Content-Range',f'bytes */{size}'); self.send_header('Content-Length','0')
                self.end_headers(); return None
            self.requested_range=(start,end); stream.seek(start)
            self.send_response(206); self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
            self.send_header('Content-Length',str(end-start+1))
        else:
            self.send_response(200); self.send_header('Content-Length',str(size))
        self.send_header('Content-Type',self.guess_type(str(path)))
        self.send_header('Accept-Ranges','bytes'); self.send_header('Cache-Control','no-store')
        self.end_headers(); return stream

    def copyfile(self,source,outputfile):
        remaining=None if self.requested_range is None else self.requested_range[1]-self.requested_range[0]+1
        try:
            while remaining is None or remaining>0:
                block=source.read(65536 if remaining is None else min(65536,remaining))
                if not block: break
                outputfile.write(block)
                if remaining is not None: remaining-=len(block)
        except (BrokenPipeError,ConnectionResetError,ConnectionAbortedError):
            pass  # A paused or reselected video can cancel an in-flight range.

    def log_message(self,format,*args):
        pass


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--port',type=int,default=8765)
    args=parser.parse_args()
    folder=ROOT/'outputs/manual_validation'
    if not (folder/'index.html').is_file(): raise FileNotFoundError('Prepare the review package first')
    server=ThreadingHTTPServer(('127.0.0.1',args.port),functools.partial(ReviewHandler,directory=str(folder)))
    print(f'Review page: http://127.0.0.1:{args.port}/index.html',flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()


if __name__=='__main__': main()
