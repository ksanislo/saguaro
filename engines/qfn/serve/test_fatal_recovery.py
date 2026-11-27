"""Real pipe/process protocol regression for a native ERR that leaves its process alive.

P124x (d): the engine releases a timed-out window's GPU waits itself; when they DRAINED and its verifier
re-armed, the line says "released and the GPU finished (#267)" and the server ends the request only - a
did-not-drain line, a timed-out line without the note, and a drained-but-still-closed (split) line end the
engine.  The fake speaks each of those shapes, in its --mode."""
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
import json
import urllib.request
from serve.server import StrataEngine, Service, ChatTemplate, EngineDied, fatal_engine_line, serve
from serve.test_server import ByteTokenizer

FAKE=r'''
import sys
from pathlib import Path
state=Path(sys.argv[sys.argv.index('--state')+1])
mode=sys.argv[sys.argv.index('--mode')+1]
print('READY 4096 stop',flush=True)
for line in sys.stdin:
 if line.startswith('QUIT'):break
 if line.startswith(('GEN','BGEN')):
  if not state.exists():
   state.write_text('first request was attempted')
   if mode=='invalid':print('ERR prompt exceeds context',flush=True)
   else:
    kind='verify batch' if mode=='batch' else 'verify'
    if mode=='recovered':
     # the engine drained its waits and re-armed: the request ends with this line, the engine idles for
     # the next GEN (the engine's timeout path answers ERR and returns to stdin - no DONE follows)
     print(f'ERR {kind}: timed out at layer 18; its GPU waits were released and the GPU finished (#267)',flush=True)
    elif mode=='split':
     # drained, but the split verifier stays closed: the fatal shape P124x review F3 must keep fatal
     print(f'ERR {kind}: timed out at layer 18; its GPU waits were released and the GPU finished, but the verifier cannot take another window (#267)',flush=True)
     print('DONE 0 1 0 1 error',flush=True)
    else:
     # no drain (or no note at all from the DONE it still prints): the verifier cannot take another window
     print(f'ERR {kind}: timed out at layer 18; its GPU waits were released but the GPU did not finish within 5 s (#267)',flush=True)
     print('DONE 0 1 0 1 error',flush=True)
   continue
  print('T 90\nDONE 1 1 0 1 stop',flush=True)
'''

class FatalRecovery(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);py=self.root/'native.py'
        if os.name=='nt':   # no shebangs on Windows: a .cmd runs the same fake engine
            py.write_text(FAKE);self.script=self.root/'native.cmd'
            self.script.write_text(f'@"{sys.executable}" "{py}" %*\r\n')
        else:
            self.script=py;py.write_text('#!'+sys.executable+'\n'+FAKE);py.chmod(0o700)
    def engine(self,mode='fatal'):
        e=StrataEngine(str(self.script),['--state',str(self.root/'state'),'--mode',mode])
        self.addCleanup(e.close);return e
    def service(self,e):
        return Service(e,ByteTokenizer(),ChatTemplate(Path(__file__).parent/'chat_template.jinja'))
    def wait_dead(self,e):
        end=time.monotonic()+2
        while e.alive() and time.monotonic()<end:time.sleep(.01)
        self.assertFalse(e.alive(),'unrecoverable native ERR must invalidate health before a new request')
    def run_request(self,svc):
        return list(svc.run([1],False,[],8,{},threading.Event()))
    def test_fatal_error_is_not_a_success_and_next_request_reloads(self):
        e=self.engine();svc=self.service(e)
        with self.assertRaises(EngineDied):self.run_request(svc)
        self.wait_dead(e)
        self.assertIs(svc.engine,e,'the failed request must not be retried automatically')
        self.assertEqual(svc.history[-1]['finish'],'error')
        result=self.run_request(svc);self.addCleanup(svc.engine.close)
        self.assertTrue(svc.engine.alive())
        self.assertTrue(svc.loaded());self.assertEqual(result[-1][1]['completion_tokens'],1)
    def test_http_health_and_next_request_recovery(self):
        old=self.engine();svc=self.service(old);http=serve(svc,port=0)
        base=f'http://127.0.0.1:{http.server_address[1]}'
        body={'messages':[{'role':'user','content':'hi'}],'max_tokens':8,
              'chat_template_kwargs':{'enable_thinking':False},'stream':True}
        try:
            def post():
                req=urllib.request.Request(base+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
                with urllib.request.urlopen(req,timeout=5) as r:return r.read().decode()
            text=post()
            self.assertIn('"error"',text);self.assertTrue(text.rstrip().endswith('data: [DONE]'))
            with urllib.request.urlopen(base+'/health',timeout=5) as r:self.assertFalse(json.load(r)['loaded'])
            body['stream']=False
            self.assertEqual(json.loads(post())['usage']['completion_tokens'],1)
            with urllib.request.urlopen(base+'/health',timeout=5) as r:self.assertTrue(json.load(r)['loaded'])
        finally:
            http.shutdown();http.server_close();svc.engine.close()
    def test_batch_timeout_signature_also_invalidates_engine(self):
        e=self.engine('batch')
        with self.assertRaises(EngineDied):list(e.generate([1],8,{},threading.Event()))
        self.wait_dead(e)

    def test_a_drained_267_timeout_ends_the_request_not_the_engine(self):
        # P124x (d): waits released AND the GPU finished - the engine re-armed its verifier.  The request ends
        # with the engine's error and the SAME engine takes the next one: no kill, no minute-long reload.
        e=self.engine('recovered');svc=self.service(e)
        with self.assertRaises(ValueError):self.run_request(svc)     # the ERR ends this request
        self.assertEqual(svc.history[-1]['finish'],'error')
        self.assertTrue(e.alive(),'a drained #267 release must not end the engine');self.assertIs(svc.engine,e)
        result=self.run_request(svc)                                 # and it did take the next one
        self.assertIs(svc.engine,e);self.assertEqual(result[-1][1]['completion_tokens'],1)
    def test_the_drained_but_still_closed_split_note_is_fatal(self):
        # P124x review F3: a split verifier that re-poisons on drain says so - keeping it (server-wise) would
        # only let the next request die on "an earlier window never finished".
        e=self.engine('split')
        self.feed(e,['ERR verify: timed out at layer 18; its GPU waits were released and the GPU finished, '
                     'but the verifier cannot take another window (#267)\n','DONE 0 1 0 1 error\n'],0)
    def test_fatal_engine_line(self):
        # the pump's classifier, on the engine's exact words
        self.assertFalse(fatal_engine_line('INFO park_pool=1 park_swap_mib=256'))
        self.assertFalse(fatal_engine_line('ERR prompt exceeds context'))
        self.assertFalse(fatal_engine_line('ERR verify: timed out at layer 42; its GPU waits were released and the GPU finished (#267)'))
        self.assertFalse(fatal_engine_line('ERR verify batch: timed out at layer 20; its GPU waits were released and the GPU finished (#267)'))
        self.assertTrue(fatal_engine_line('ERR verify: timed out at layer 42; its GPU waits were released but the GPU did not finish within 5 s (#267)'))
        self.assertTrue(fatal_engine_line('ERR verify batch: timed out at layer 20'))   # no note (an older engine): fatal
        self.assertTrue(fatal_engine_line('ERR verify: timed out at layer 42; its GPU waits were released and the GPU finished, but the verifier cannot take another window (#267)'))
        self.assertTrue(fatal_engine_line('ERR verify: an earlier window never finished on the GPU (#267); restart the engine'))
        self.assertTrue(fatal_engine_line('ERR verify batch: an earlier window never finished (#267)'))

    def feed(self,e,lines,slots):
        import queue
        class P:
            stdout=iter(lines)
            killed=False
            def kill(self):self.killed=True     # the engine ignores SIGTERM in --serve
        old=e.proc;e.proc=P();e.lines=queue.Queue();e.slot_q=[queue.Queue() for _ in range(slots)];e.ended=False
        e._pump()
        self.assertTrue(e.ended);self.assertTrue(e.proc.killed)
        self.assertIsNone(e.lines.get_nowait());self.assertTrue(e.lines.empty())   # EOF, never the trailing DONE
        e.proc=old
    def test_fatal_lines_are_checked_before_batch_routing(self):
        e=self.engine('fatal')
        self.feed(e,['ERR verify batch: timed out at layer 3\n','BDONE 0 1 0 1 error\n'],2)
        self.assertTrue(all(q.get_nowait() is None and q.empty() for q in e.slot_q))   # only the EOF reached a slot, not the BDONE
    def test_an_earlier_window_never_finished_is_fatal(self):
        e=self.engine('fatal')
        self.feed(e,['ERR verify: an earlier window never finished on the GPU (#267); restart the engine\n','DONE 0 1 0 1 error\n'],0)
        self.feed(e,['ERR verify batch: an earlier window never finished on the GPU (#267)\n'],2)
    def test_generate_batched_refuses_a_dead_engine(self):
        e=self.engine('fatal');e.close()
        with self.assertRaises(EngineDied):list(e.generate_batched([1],8,{},threading.Event()))

    def test_validation_error_does_not_kill_engine(self):
        e=self.engine('invalid')
        with self.assertRaises(ValueError):list(e.generate([1],8,{},threading.Event()))
        self.assertTrue(e.alive())
        self.assertEqual(list(e.generate([1],8,{},threading.Event())),[90])
    def test_closed_serial_engine_rejects_without_accessing_closed_pipe(self):
        e=self.engine();e.close()
        with self.assertRaises(EngineDied):list(e.generate([1],8,{},threading.Event()))
if __name__=='__main__':unittest.main()
