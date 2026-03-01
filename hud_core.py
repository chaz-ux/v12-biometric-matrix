
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math, time, csv, textwrap
import numpy as np
from collections import deque, defaultdict
from datetime import datetime
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# SENSITIVITY PRESETS
# vote_to_di = minimum votes for DI verdict
# vote_to_inc = minimum votes for INC
# ─────────────────────────────────────────────────────────────────────────────
SENSITIVITY = {
    'LOW' : {'di': 2, 'inc': 1,
             'label': 'LOW  — casual / party mode (flags more, more false positives)'},
    'MED' : {'di': 3, 'inc': 2,
             'label': 'MED  — general purpose'},
    'HIGH': {'di': 4, 'inc': 3,
             'label': 'HIGH — serious investigation (conservative, fewer false positives)'},
}

# Fixed irrelevant truth calibration questions
TRUTH_QS = [
    "Say your first name.",
    "Name the city you are currently in.",
    "Say what you last ate or drank.",
    "Say today's day of the week.",
]

# Paired control questions (universal guilt prompts)
CONTROL_QS = [
    "Have you ever lied to avoid getting in trouble?",
    "Have you ever taken something that wasn't yours?",
    "Have you ever said something hurtful behind someone's back?",
    "Have you ever broken a promise you made to someone?",
    "Have you ever cheated at something, even something small?",
]

MODEL_PATH     = 'face_landmarker.task'
BLINK_THRESH   = 0.22
SMOOTH_A       = 0.25
NEUTRAL_SEC    = 8.0
OF_SIGMA       = 2.2      # slightly lower = more sensitive
OF_MIN_F       = 2
OF_MAX_F       = 12
BLINK_CONS     = 4.0
BLINK_FLUT     = 2.5
RUNS           = 2

# ─────────────────────────────────────────────────────────────────────────────
# ASYNC
# ─────────────────────────────────────────────────────────────────────────────
latest_result = None
def result_callback(result: Any, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRY
# ─────────────────────────────────────────────────────────────────────────────
def d3(p1,p2,w,h):
    return math.sqrt(((p1.x-p2.x)*w)**2+((p1.y-p2.y)*h)**2+((p1.z-p2.z)*w)**2)

def get_ear(eye,w,h):
    v1=d3(eye[1],eye[5],w,h); v2=d3(eye[2],eye[4],w,h); hw=d3(eye[0],eye[3],w,h)
    return (v1+v2)/(2*hw) if hw>0 else 0.3

def face_roi(lm,w,h,pad=0.12):
    xs=[int(l.x*w) for l in lm]; ys=[int(l.y*h) for l in lm]
    return (max(0,min(xs)-int(w*pad)),max(0,min(ys)-int(h*pad)),
            min(w,max(xs)+int(w*pad)),min(h,max(ys)+int(h*pad)))

def au_raw(lm,w,h):
    fw=d3(lm[33],lm[263],w,h)+1e-6; fh=d3(lm[10],lm[152],w,h)+1e-6
    return {
        'au4' :d3(lm[107],lm[336],w,h)/fw,
        'au1' :(d3(lm[159],lm[52],w,h)+d3(lm[386],lm[282],w,h))/(2*fh),
        'au12':d3(lm[61],lm[291],w,h)/fw,
        'au6' :(d3(lm[117],lm[111],w,h)+d3(lm[346],lm[340],w,h))/(2*fh),
        'au23':d3(lm[13],lm[14],w,h)/fh,
    }

AU_KEYS=['au4','au1','au12','au6','au23']

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL ENGINES (same as v19, slightly tuned)
# ─────────────────────────────────────────────────────────────────────────────
class OFEngine:
    def __init__(self):
        self.prev=None; self.hist=deque(maxlen=12)
        self.bm=0.0; self.bs=0.01; self.sf=0; self.ev=0; self.ready=False
    def update(self,frame,roi):
        x1,y1,x2,y2=roi
        if x2<=x1 or y2<=y1: return 0.0,False
        gray=cv2.resize(cv2.cvtColor(frame[y1:y2,x1:x2],cv2.COLOR_BGR2GRAY),(120,90))
        mag=0.0; spike=False
        if self.prev is not None and self.prev.shape==gray.shape:
            flow=cv2.calcOpticalFlowFarneback(self.prev,gray,None,0.5,2,10,2,5,1.1,0)
            mag=float(np.mean(np.sqrt(flow[...,0]**2+flow[...,1]**2)))
        self.prev=gray.copy(); self.hist.append(mag)
        if self.ready:
            z=(mag-self.bm)/max(self.bs,1e-6)
            if z>OF_SIGMA:
                self.sf+=1
                if self.sf>=OF_MIN_F: spike=True
            else:
                if OF_MIN_F<=self.sf<=OF_MAX_F: self.ev+=1
                self.sf=0
        return mag,spike
    def calibrate(self):
        if len(self.hist)>=5:
            a=np.array(self.hist); self.bm=float(np.mean(a)); self.bs=float(np.std(a))+0.001
        self.ready=True
    def pop(self): n=self.ev; self.ev=0; return n

class rPPGEngine:
    def __init__(self):
        self.buf=[]; self.tbuf=[]; self.bpm=0; self.var=0.0; self.lt=0.0
    def update(self,frame,lm,w,h,now):
        fhx=int(lm[151].x*w); fhy=int(lm[151].y*h)
        bw=int(w*0.05); bh=int(h*0.04)
        y1=max(0,fhy-bh); y2=min(frame.shape[0],fhy+bh)
        x1=max(0,fhx-bw); x2=min(frame.shape[1],fhx+bw)
        if y2>y1 and x2>x1:
            self.buf.append(float(np.mean(frame[y1:y2,x1:x2,1]))); self.tbuf.append(now)
            if len(self.buf)>300: self.buf.pop(0); self.tbuf.pop(0)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,160,0),1)
        if now-self.lt>0.5 and len(self.buf)>150: self._calc(); self.lt=now
    def _calc(self):
        el=self.tbuf[-1]-self.tbuf[0]
        if el<=0: return
        fps=len(self.buf)/el
        sig=(np.array(self.buf)-np.mean(self.buf))*np.hanning(len(self.buf))
        freqs=np.fft.rfftfreq(len(sig),d=1/fps); mag=np.abs(np.fft.rfft(sig))
        lo=np.searchsorted(freqs,0.8); hi=np.searchsorted(freqs,3.0)
        if lo<hi<len(mag):
            self.bpm=int(freqs[lo+np.argmax(mag[lo:hi])]*60)
            fft=np.fft.rfft(sig); m=np.zeros_like(fft); m[lo:hi]=fft[lo:hi]
            self.var=float(np.var(np.fft.irfft(m,n=len(sig))[-90:]))

class BlinkEngine:
    def __init__(self):
        self.bl=False; self.ts=deque()
        self.tc=[]; self.tf=[]
        self.tc_thr=None; self.tf_thr=None
    def update(self,ear,now):
        if ear<BLINK_THRESH:
            if not self.bl: self.ts.append(now); self.bl=True
        else: self.bl=False
        while self.ts and now-self.ts[0]>60: self.ts.popleft()
    def rate(self,t0,t1):
        d=t1-t0; return sum(1 for t in self.ts if t0<=t<=t1)/d if d>0 else 0.0
    def record_truth(self,a0,a1):
        self.tc.append(self.rate(a0,a0+BLINK_CONS))
        self.tf.append(self.rate(a1,a1+BLINK_FLUT))
    def set_thresh(self):
        if len(self.tc)>=2:
            m=np.mean(self.tc); s=np.std(self.tc)+1e-6; self.tc_thr=m-1.0*s
            m=np.mean(self.tf); s=np.std(self.tf)+1e-6; self.tf_thr=m+1.5*s
    def flags(self,a0,a1,now):
        if self.tc_thr is None: return False,False
        return (self.rate(a0,a0+BLINK_CONS)<self.tc_thr,
                self.rate(a1,min(now,a1+BLINK_FLUT))>self.tf_thr)
    def recent(self,now):
        return int(sum(1 for t in self.ts if now-t<15)/15*60)

class FACSEngine:
    def __init__(self):
        self.sm={k:0.0 for k in AU_KEYS}; self.anchor={}; self.buf=[]
        self.tscores=[]; self.thr=None
    def smooth(self,raw):
        for k in AU_KEYS:
            self.sm[k]=raw[k] if self.sm[k]==0.0 else raw[k]*SMOOTH_A+self.sm[k]*(1-SMOOTH_A)
    def add(self): self.buf.append(dict(self.sm))
    def clear(self): self.buf=[]
    def tension(self):
        if not self.buf or not self.anchor: return 0.0
        a4=np.mean([f['au4'] for f in self.buf]); a23=np.mean([f['au23'] for f in self.buf])
        n4=self.anchor.get('au4',a4); n23=self.anchor.get('au23',a23)
        return float((n4-a4)/(n4+1e-6)*0.6+(n23-a23)/(n23+1e-6)*0.4)
    def duchenne(self):
        if not self.buf or not self.anchor: return 0.0
        n12=self.anchor.get('au12',0); n6=self.anchor.get('au6',0)
        c=sum(1 for f in self.buf if
              (f['au12']-n12)/(n12+1e-6)>0.06 and (f['au6']-n6)/(n6+1e-6)<0.04)
        return c/len(self.buf)
    def record_truth(self): self.tscores.append(self.tension())
    def set_thresh(self):
        if len(self.tscores)>=2:
            m=np.mean(self.tscores); s=np.std(self.tscores)+1e-6; self.thr=m+1.5*s

# ─────────────────────────────────────────────────────────────────────────────
# CQT COMPARISON  (sensitivity-aware)
# ─────────────────────────────────────────────────────────────────────────────
def compare(test_sig, ctrl_sig, sens_key):
    sp = SENSITIVITY[sens_key]
    votes = sum([
        test_sig['of']      - ctrl_sig['of']      > 0.3,
        test_sig['rppg']    - ctrl_sig['rppg']     > 0.00015,
        test_sig['blink_c'] - ctrl_sig['blink_c']  > 0.03,
        test_sig['tension'] - ctrl_sig['tension']  > 0.008,
        test_sig['duc']     - ctrl_sig['duc']       > 0.02,
    ])
    verd = ('DI'  if votes >= sp['di']  else
            'INC' if votes >= sp['inc'] else 'NDI')
    return verd, votes

def sig_of_window(S, now):
    return {
        'of'     : S['of'].pop(),
        'rppg'   : S['rppg'].var,
        'blink_c': S['blink'].rate(S['ans_start'], S['ans_start']+BLINK_CONS),
        'tension': S['facs'].tension(),
        'duc'    : S['facs'].duchenne(),
    }

# ─────────────────────────────────────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────────────────────────────────────
def build_steps(test_qs):
    steps=[{'type':'neutral'}]
    for i,q in enumerate(TRUTH_QS):
        steps.append({'type':'truth','q':q,'idx':i})
    for run in range(RUNS):
        for qi,tq in enumerate(test_qs):
            cq=CONTROL_QS[qi % len(CONTROL_QS)]
            steps.append({'type':'test','q':tq,'qi':qi,'run':run})
            steps.append({'type':'control','q':cq,'qi':qi,'run':run})
    steps.append({'type':'done'})
    return steps

def fresh_subject(test_qs, sens, subject_num):
    return dict(
        num=subject_num, sens=sens,
        steps=build_steps(test_qs),
        step_idx=0,
        phase_t=time.time(),
        neutral_done=False,
        in_answer=False, ans_start=None, ans_end=None,
        pending_test=None,
        pair_results=[],      # (run, qi, q_text, verdict, votes)
        cumvotes=0, totpairs=0,
        of=OFEngine(), rppg=rPPGEngine(), blink=BlinkEngine(), facs=FACSEngine(),
    )

# ─────────────────────────────────────────────────────────────────────────────
# SETUP SCREEN
# ─────────────────────────────────────────────────────────────────────────────
def run_setup_screen(cap):
    """
    Full-screen setup: type questions, pick sensitivity.
    Returns (questions_list, sensitivity_key) or (None,None) if quit.
    """
    questions = []
    current_input = ""
    sens_idx = 1   # 0=LOW, 1=MED, 2=HIGH
    sens_keys = ['LOW','MED','HIGH']
    subject_name = ""
    typing_name = True   # first type subject/scenario name

    FONT  = cv2.FONT_HERSHEY_SIMPLEX
    MONO  = cv2.FONT_HERSHEY_PLAIN

    while True:
        ok, frame = cap.read()
        if not ok: return None, None
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Dark overlay
        overlay = np.zeros_like(frame); overlay[:] = (12, 6, 22)
        frame[:] = cv2.addWeighted(frame, 0.08, overlay, 0.92, 0)

        # Title
        cv2.putText(frame, "POLYGRAPH  V20", (30, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, (180, 100, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "REAL SESSION SETUP", (30, 80),
                    FONT, 0.6, (100, 60, 140), 1, cv2.LINE_AA)
        cv2.line(frame, (30,92),(w-30,92),(40,20,60),1)

        # Sensitivity selector
        sy = 108
        # UI TWEAK: Changed instruction to CTRL+S so 's' can be typed
        cv2.putText(frame,"SENSITIVITY  [ CTRL+S to cycle ]",(30,sy),FONT,0.5,(120,120,120),1)
        sy += 22
        for i,k in enumerate(sens_keys):
            col = (0,220,150) if i==sens_idx else (50,50,70)
            cv2.putText(frame, ("▶ " if i==sens_idx else "  ")+SENSITIVITY[k]['label'],
                        (30,sy),FONT,0.5,col,1)
            sy+=20
        cv2.line(frame,(30,sy+2),(w-30,sy+2),(40,20,60),1); sy+=14

        # Scenario / subject name
        cv2.putText(frame,"SCENARIO / SUBJECT LABEL:",(30,sy+16),FONT,0.5,(120,120,120),1)
        
        # Highlighting the active field
        name_col = (255, 230, 100) if typing_name else (140, 120, 180)
        cursor = "█" if typing_name and int(time.time()*2)%2==0 else ""
        cv2.putText(frame, f"  {subject_name}{cursor}", (30,sy+36), FONT, 0.65, name_col, 1)
        sy += 56
        cv2.line(frame,(30,sy),(w-30,sy),(40,20,60),1); sy+=10

        # Questions list
        cv2.putText(frame,f"TEST QUESTIONS  ({len(questions)}/8)  [ ENTER to add ]",(30,sy+14),FONT,0.48,(120,120,120),1); sy+=28
        for i,q in enumerate(questions):
            q_short = q[:70]+'…' if len(q)>70 else q
            cv2.putText(frame,f"  {i+1}. {q_short}",(30,sy),FONT,0.5,(200,200,220),1)
            sy += 20
            if sy > h-120: break

        # Current input field
        input_col = (0, 220, 255) if not typing_name else (60, 60, 80)
        cursor2 = "█" if not typing_name and int(time.time()*2)%2==0 else ""
        cv2.putText(frame,f"> {current_input}{cursor2}",(30,h-80), FONT, 0.6, input_col, 1)

        # UI Upgrade: Dynamic Start Indicator
        is_ready_to_start = (not typing_name and current_input == "" and len(questions) > 0)
        
        if is_ready_to_start:
            pulse = int((math.sin(time.time() * 6) + 1) * 75) + 100
            cv2.putText(frame, "SYSTEM READY", (w-260, h-100), FONT, 0.5, (0, pulse, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, "[ PRESS SPACE TO START ]", (w-290, h-75), FONT, 0.65, (0, pulse, 0), 2, cv2.LINE_AA)
        elif len(questions) > 0:
            cv2.putText(frame, "Clear input to start...", (w-230, h-80), FONT, 0.5, (80, 80, 80), 1)

        # Footer instructions (Removed 'Q' so user can type the letter 'q')
        cv2.line(frame,(30,h-92),(w-30,h-92),(40,20,60),1)
        instr = ("TAB = switch field  |  ENTER = add question  |  BACKSPACE = delete  |  ESC = quit")
        cv2.putText(frame,instr,(30,h-12),FONT,0.38,(70,70,70),1)

        cv2.imshow('Polygraph V20 — Setup',frame)
        key = cv2.waitKey(30) & 0xFF

        # --- EVENT HANDLING LOGIC ---
        if key == 255: continue # Optimization: skip logic if no key pressed

        if key == 27:  # ESC key only for quit (Freeing up 'q')
            return None, None

        elif key == 9:  # TAB — switch between name and question input
            typing_name = not typing_name

        elif key == 19: # CTRL+S (Freeing up 's' and 'S')
            sens_idx = (sens_idx + 1) % 3

        elif key in (8, 127):  # BACKSPACE 
            if typing_name and subject_name:
                subject_name = subject_name[:-1]
            elif not typing_name and current_input:
                current_input = current_input[:-1]

        elif key == 13:  # ENTER
            if typing_name:
                typing_name = False
            else:
                q = current_input.strip()
                if q and len(questions) < 8:
                    questions.append(q)
                    current_input = ""
                # Bonus: Enter on empty line also starts session
                elif not q and len(questions) > 0:
                    break 

        # THE START TRIGGER FIX: Must be checked before printable characters
        elif key == 32 and is_ready_to_start:
            break

        elif 32 <= key <= 126:  # ALL printable characters
            target_len = len(subject_name) if typing_name else len(current_input)
            if target_len < 60:
                if typing_name: 
                    subject_name += chr(key)
                else: 
                    current_input += chr(key)

    cv2.destroyWindow('Polygraph V20 — Setup')
    return questions, sens_keys[sens_idx], subject_name.strip() or "Subject"
# ─────────────────────────────────────────────────────────────────────────────
# REVEAL SCREEN
# ─────────────────────────────────────────────────────────────────────────────
def show_reveal(cap, all_subjects, test_qs, sens_key):
    """Multi-subject comparison screen."""
    FONT=cv2.FONT_HERSHEY_SIMPLEX
    while True:
        ok,frame=cap.read()
        if not ok: break
        frame=cv2.flip(frame,1); h,w,_=frame.shape
        overlay=np.zeros_like(frame); overlay[:]=(10,5,18)
        frame[:]=cv2.addWeighted(frame,0.05,overlay,0.95,0)

        cv2.putText(frame,"SESSION RESULTS",(30,45),cv2.FONT_HERSHEY_DUPLEX,1.1,(180,100,255),2)
        sp=SENSITIVITY[sens_key]
        cv2.putText(frame,f"Sensitivity: {sens_key}  |  {len(all_subjects)} subject(s)",
                    (30,72),FONT,0.5,(80,60,100),1)
        cv2.line(frame,(30,84),(w-30,84),(40,20,60),1)

        y=100
        # Per-subject cumulative scores
        scores=[]
        for sub in all_subjects:
            tot=sub['totpairs']; cum=sub['cumvotes']
            pct=cum/(tot*5)*100 if tot>0 else 0
            scores.append((sub['num'],sub.get('name','?'),pct,sub['pair_results']))

        # Sort by deception score descending
        scores.sort(key=lambda x:-x[2])

        # Subject summary bar
        for rank,(num,name,pct,results) in enumerate(scores):
            bar_col=(0,0,220) if pct>55 else (0,140,220) if pct>35 else (0,200,80)
            label='MOST SUSPICIOUS' if rank==0 and pct>35 else ''
            bar_w=int((w*0.4)*min(pct,100)/100)
            cv2.rectangle(frame,(30,y),(30+bar_w,y+22),bar_col,-1)
            cv2.rectangle(frame,(30,y),(int(w*0.4)+30,y+22),(40,20,60),1)
            cv2.putText(frame,f"{name}  {pct:.0f}%  {label}",
                        (35,y+15),FONT,0.5,(220,220,220),1)
            y+=30

        y+=10; cv2.line(frame,(30,y),(w-30,y),(40,20,60),1); y+=14

        # Per-question breakdown across subjects
        cv2.putText(frame,"PER-QUESTION BREAKDOWN:",(30,y),FONT,0.5,(100,100,120),1); y+=20
        for qi,tq in enumerate(test_qs):
            q_short=tq[:48]+'…' if len(tq)>48 else tq
            cv2.putText(frame,f"Q{qi+1}: {q_short}",(30,y),FONT,0.48,(160,160,180),1); y+=18
            for _,name,_,results in scores:
                # Find verdicts for this question across runs
                q_res=[r for r in results if r[1]==qi]
                if q_res:
                    v_str=" ".join(f"{r[2]}({r[3]}/5)" for r in q_res)
                    # Aggregate
                    avg=np.mean([r[3] for r in q_res])
                    col=(0,0,220) if avg>=sp['di'] else (0,140,220) if avg>=sp['inc'] else (0,200,80)
                    final='DI' if avg>=sp['di'] else 'INC' if avg>=sp['inc'] else 'NDI'
                    cv2.putText(frame,f"    {name}: {final} [{v_str}]",
                                (30,y),FONT,0.44,col,1)
                    y+=16
            y+=4
            if y>h-60: break

        cv2.putText(frame,"Q=quit  R=restart",(30,h-12),FONT,0.4,(60,60,60),1)
        cv2.imshow('Polygraph V20 — Results',frame)
        key=cv2.waitKey(30)&0xFF
        if key==ord('q') or key==27: break
        if key==ord('r') or key==ord('R'): return 'restart'

    cv2.destroyWindow('Polygraph V20 — Results')
    return 'quit'

# ─────────────────────────────────────────────────────────────────────────────
# MEDIAPIPE + CAMERA
# ─────────────────────────────────────────────────────────────────────────────
base_options=python.BaseOptions(model_asset_path=MODEL_PATH)
opts=vision.FaceLandmarkerOptions(base_options=base_options,num_faces=1,
     running_mode=vision.RunningMode.LIVE_STREAM,result_callback=result_callback)
detector=vision.FaceLandmarker.create_from_options(opts)
cap=cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
cap.set(cv2.CAP_PROP_FPS,60)

# ─────────────────────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────
FONT=cv2.FONT_HERSHEY_SIMPLEX

def hud_panel(img,w,h):
    ov=img.copy(); cv2.rectangle(ov,(0,0),(w,145),(12,6,22),-1)
    img[:]=cv2.addWeighted(img,0.12,ov,0.88,0)
    cv2.line(img,(0,145),(w,145),(40,20,60),1)

def put(img,text,x,y,scale=0.6,col=(220,220,220),thick=1):
    cv2.putText(img,text,(x,y),FONT,scale,col,thick,cv2.LINE_AA)

def pulsing_dot(img,now,x,y,col=(0,0,255)):
    r=int(10+3*math.sin(now*5))
    cv2.circle(img,(x,y),r,col,-1)

def verdict_color(v):
    return {
        'DI' :(0,0,255),
        'INC':(0,140,255),
        'NDI':(0,220,80),
    }.get(v,(150,150,150))

def wrap_text(img,text,x,y,max_w,scale,col,thick=1):
    """Word-wrap text into image."""
    words=text.split(); line=""; yl=y
    for word in words:
        test=line+(" " if line else "")+word
        tw=cv2.getTextSize(test,FONT,scale,thick)[0][0]
        if tw>max_w and line:
            put(img,line,x,yl,scale,col,thick); yl+=int(22*scale/0.5); line=word
        else: line=test
    if line: put(img,line,x,yl,scale,col,thick)
    return yl+int(22*scale/0.5)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN SESSION LOOP
# ─────────────────────────────────────────────────────────────────────────────
def run_session(test_qs, sens_key, first_subject_name):
    global latest_result
    all_subjects=[]
    subject_num=1
    S=fresh_subject(test_qs,sens_key,subject_num)
    S['name']=first_subject_name
    prev_t=0.0
    sess_id=datetime.now().strftime('%Y%m%d_%H%M%S')

    while True:
        ok,image=cap.read()
        if not ok: break

        key=cv2.waitKey(1)&0xFF
        now=time.time()

        if key==ord('q') or key==27:
            all_subjects.append(S)
            break

        if key in (ord('r'),ord('R')):
            all_subjects.append(S)
            break

        if key==ord('n') or key==ord('N'):
            # Next subject
            all_subjects.append(S)
            subject_num+=1
            S=fresh_subject(test_qs,sens_key,subject_num)
            # Ask for name via console
            name=input(f"\nSubject {subject_num} name (press ENTER to skip): ").strip()
            S['name']=name or f"Subject {subject_num}"
            latest_result=None
            print(f"Starting calibration for {S['name']}")
            continue

        step=S['steps'][S['step_idx']] if S['step_idx']<len(S['steps']) else {'type':'done'}

        # SPACE = mark answer done
        if key==ord(' '):
            stype=step.get('type')
            if stype in ('truth','test','control'):
                if not S['in_answer']:
                    S['in_answer']=True; S['ans_start']=now
                    S['of'].pop(); S['facs'].clear()
                else:
                    S['in_answer']=False; S['ans_end']=now
                    if now-S['ans_start']<1.5:
                        S['in_answer']=False  # too short, retry
                    else:
                        sig=sig_of_window(S,now)
                        if stype=='truth':
                            S['facs'].record_truth(); S['blink'].record_truth(S['ans_start'],now)
                            S['facs'].set_thresh(); S['blink'].set_thresh()
                            S['step_idx']+=1
                        elif stype=='test':
                            S['pending_test']=sig; S['step_idx']+=1
                        elif stype=='control':
                            if S['pending_test']:
                                verd,votes=compare(S['pending_test'],sig,sens_key)
                                qi=step['qi']; run=step['run']
                                qt=test_qs[qi]
                                S['pair_results'].append((run,qi,qt,verd,votes))
                                S['cumvotes']+=votes; S['totpairs']+=1
                                S['pending_test']=None
                                print(f"  {S['name']} Q{qi+1} R{run+1}: {verd} ({votes}/5)")
                            S['step_idx']+=1

        # ── FRAME ─────────────────────────────────────────────────────────────
        dt=max(now-prev_t,0.001); fps=1/dt; prev_t=now
        image=cv2.flip(image,1); h,w,_=image.shape

        detector.detect_async(
            mp.Image(image_format=mp.ImageFormat.SRGB,
                     data=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)),
            int(now*1000))

        if not (latest_result and latest_result.face_landmarks):
            hud_panel(image,w,h)
            put(image,"NO FACE DETECTED — position subject in frame",30,80,0.7,(0,0,220),2)
            cv2.imshow('Polygraph V20',image); continue

        lm=latest_result.face_landmarks[0]
        roi=face_roi(lm,w,h)

        S['of'].update(image,roi)
        S['rppg'].update(image,lm,w,h,now)
        ear_now=(get_ear([lm[i] for i in [33,160,158,133,153,144]],w,h)+
                 get_ear([lm[i] for i in [362,385,387,263,373,380]],w,h))/2
        S['blink'].update(ear_now,now)
        S['facs'].smooth(au_raw(lm,w,h))
        if S['in_answer']: S['facs'].add()

        # Face mesh
        mc=(0,160,60) if S['in_answer'] else (35,35,55)
        for lmk in lm:
            cv2.circle(image,(int(lmk.x*w),int(lmk.y*h)),1,mc,-1)

        # ── NEUTRAL ───────────────────────────────────────────────────────────
        if step['type']=='neutral':
            elapsed=now-S['phase_t']
            pct=int(min(elapsed/NEUTRAL_SEC,1.0)*100)
            if elapsed>=NEUTRAL_SEC and not S['neutral_done']:
                S['facs'].anchor={k:S['facs'].sm[k] for k in AU_KEYS}
                S['of'].calibrate(); S['neutral_done']=True; S['step_idx']=1

            hud_panel(image,w,h)
            put(image,f"SUBJECT: {S['name']}  |  SENSITIVITY: {sens_key}",30,30,0.55,(140,80,180))
            put(image,"PHASE 1 — NEUTRAL LOCK",30,58,0.75,(0,200,255),2)
            put(image,"Subject: look at camera. Do not speak or move.",30,85,0.55,(200,200,200))
            put(image,f"Calibrating baseline... {pct}%",30,112,0.6,(100,180,100))
            bw=int((w-60)*pct/100)
            cv2.rectangle(image,(30,128),(w-30,138),(30,15,40),-1)
            cv2.rectangle(image,(30,128),(30+bw,138),(0,200,255),-1)

        # ── TRUTH ─────────────────────────────────────────────────────────────
        elif step['type']=='truth':
            nd=step['idx']; tot=len(TRUTH_QS)
            hud_panel(image,w,h)
            put(image,f"SUBJECT: {S['name']}  |  CALIBRATION  {nd+1}/{tot}",30,30,0.55,(140,80,180))
            put(image,"PHASE 2 — TRUTH CALIBRATION",30,58,0.75,(0,220,120),2)
            sq=step['q']
            if not S['in_answer']:
                y_=wrap_text(image,f'Ask: "{sq}"',30,85,w-60,0.6,(220,220,220))
                put(image,"Press SPACE when subject finishes answering.",30,y_+8,0.52,(100,100,120))
            else:
                dur=now-S['ans_start']
                put(image,f'Answering: "{sq}"',30,85,0.55,(200,200,200))
                put(image,f"Recording...  {dur:.1f}s   SPACE when done",30,112,0.6,(0,220,255))
                pulsing_dot(image,now,w-30,25)

        # ── TEST / CONTROL ────────────────────────────────────────────────────
        elif step['type'] in ('test','control'):
            qi=step['qi']; run=step['run']; is_test=step['type']=='test'
            col_t=(0,140,255) if is_test else (0,200,180)
            label="TEST QUESTION" if is_test else "CONTROL QUESTION"
            sp_=SENSITIVITY[sens_key]

            hud_panel(image,w,h)
            put(image,f"SUBJECT: {S['name']}  |  Run {run+1}/{RUNS}  Pair {qi+1}/{len(test_qs)}  |  {sens_key}",
                30,30,0.5,(140,80,180))
            put(image,f"PHASE 3 — {label}",30,58,0.72,col_t,2)

            sq2=step['q']
            if not S['in_answer']:
                note="(test topic)" if is_test else "(everyone has done this)"
                y_=wrap_text(image,f'Ask: "{sq2}"',30,85,w-60,0.6,(220,220,220))
                put(image,note,30,y_+4,0.45,(80,80,100))
                put(image,"SPACE when subject finishes.",30,y_+24,0.52,(100,100,120))
            else:
                dur=now-S['ans_start']
                wrap_text(image,f'"{sq2}"',30,85,w-60,0.55,(180,180,200))
                put(image,f"Recording... {dur:.1f}s  |  SPACE when done",30,115,0.6,(0,200,255))
                pulsing_dot(image,now,w-30,25)

            # Live channels mini-display
            y0=h-105
            items=[f"OF:{S['of'].ev}ev",f"HR:{S['rppg'].bpm}bpm",
                   f"BLINK:{S['blink'].recent(now)}/m",f"TENS:{S['facs'].tension():.3f}"]
            put(image,"  ".join(items),20,y0,0.4,(60,80,60))

            # Last results
            if S['pair_results']:
                y0=h-85
                put(image,"Recent:",20,y0,0.42,(60,60,80)); y0+=18
                for r in S['pair_results'][-4:]:
                    run_,qi_,qt_,verd_,v_=r
                    q_s=qt_[:35]+'…' if len(qt_)>35 else qt_
                    col=verdict_color(verd_)
                    put(image,f"Q{qi_+1}R{run_+1} [{verd_} {v_}/5] {q_s}",22,y0,0.4,col)
                    y0+=16

        # ── DONE ──────────────────────────────────────────────────────────────
        elif step['type']=='done':
            tot=S['totpairs']; cum=S['cumvotes']
            pct=cum/(tot*5)*100 if tot>0 else 0
            overall_col=(0,0,220) if pct>55 else (0,140,220) if pct>35 else (0,200,80)
            overall_str="HIGH DECEPTION SIGNAL" if pct>55 else "MODERATE SIGNAL" if pct>35 else "NO DECEPTION SIGNAL"

            hud_panel(image,w,h)
            put(image,f"SUBJECT: {S['name']} — COMPLETE",30,35,0.65,(180,100,255),2)
            put(image,f"{overall_str}   {pct:.0f}% cumulative signal",30,70,0.65,overall_col,2)
            put(image,f"Votes: {cum}/{tot*5}  |  Pairs: {tot}",30,100,0.5,(120,120,140))
            put(image,f"N = next subject  |  Q = view results & quit  |  R = full restart",
                30,128,0.45,(80,80,100))

            # Per-question summary
            by_q=defaultdict(list)
            for r in S['pair_results']:
                by_q[r[1]].append(r)
            y0=160
            for qi2 in sorted(by_q.keys()):
                entries=by_q[qi2]
                avg_v=np.mean([e[4] for e in entries])
                sp_=SENSITIVITY[sens_key]
                fv='DI' if avg_v>=sp_['di'] else 'INC' if avg_v>=sp_['inc'] else 'NDI'
                col=verdict_color(fv)
                q_s=test_qs[qi2][:55]+'…' if len(test_qs[qi2])>55 else test_qs[qi2]
                put(image,f"Q{qi2+1}: [{fv}] {q_s}",30,y0,0.5,col); y0+=22
                if y0>h-40: break

        # FPS indicator
        put(image,f"fps:{int(fps)}",(w-72),(h-8),0.38,(30,30,30))
        cv2.imshow('Polygraph V20',image)

    cv2.destroyWindow('Polygraph V20')

    # Save CSV
    all_rows=[]
    for sub in all_subjects:
        for r in sub['pair_results']:
            run_,qi_,qt_,verd_,v_=r
            all_rows.append([sub.get('name','?'),run_+1,qi_+1,qt_,verd_,v_,''])
    if all_rows:
        fn=f"session_{sess_id}.csv"
        with open(fn,'w',newline='') as f:
            cw=csv.writer(f)
            cw.writerow(["Subject","Run","Q","Question","Verdict","Votes","GroundTruth"])
            cw.writerows(all_rows)
        print(f"Saved {fn}")

    return all_subjects

# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
print("Polygraph V20 — opening setup screen...")
print("In the setup window: type your scenario name, then TAB to questions.")
print("Press ENTER after each question. SPACE twice on empty line to start.")

while True:
    test_qs, sens_key, first_name = run_setup_screen(cap)
    if test_qs is None:
        break

    all_subs = run_session(test_qs, sens_key, first_name)

    action = show_reveal(cap, all_subs, test_qs, sens_key)
    if action != 'restart':
        break

cap.release()
cv2.destroyAllWindows()
print("Done.")