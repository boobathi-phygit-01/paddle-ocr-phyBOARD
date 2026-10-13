import os, time, math, argparse, threading
from collections import deque
import cv2
import numpy as np
try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
except ImportError:
    from ai_edge_litert.interpreter import Interpreter, load_delegate  # newer name of tflite_runtime
from PIL import Image, ImageDraw, ImageFont

# ---------------- args ----------------
ap = argparse.ArgumentParser()
ap.add_argument("--det", default="models/det_int8.tflite")
ap.add_argument("--rec", default="models/inference_rec_logits_float32.tflite")
ap.add_argument("--dict", default="rec_dict.txt")
ap.add_argument("--font", default=None)
ap.add_argument("--cam", type=int, default=0)
ap.add_argument("--hd", action="store_true", help="capture at 1280x720")
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--label-size", type=int, default=32, help="font size of recognized-text labels")
ap.add_argument("--fps-size", type=int, default=36, help="font size of the FPS overlay")
ap.add_argument("--delegate", nargs="?", const="/usr/lib/libvx_delegate.so", default="/usr/lib/libvx_delegate.so",
                help="load an NPU delegate; with no value uses /usr/lib/libvx_delegate.so")
args = ap.parse_args()

# ---------------- config ----------------
OCR_EVERY  = 2          # run OCR every N frames
DET_SIZE   = 640
DET_THRESH = 0.2
BOX_THRESH = 0.4
UNCLIP     = 1.8
REC_H, REC_W = 48, 320
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD  = np.array([0.229, 0.224, 0.225], np.float32)

# tracking / voting
MIN_READ   = 0.25       # ignore single reads below this confidence
MIN_SHOW   = 0.55       # only display a track whose best read is at least this
HISTORY    = 8          # reads kept per box
MAX_MISS   = 4          # OCR cycles a box may disappear before its track is dropped
SHOW_MISS  = 2          # still draw a box for this many cycles after it was last seen
IOU_MATCH  = 0.3

# ---------------- load models ----------------
delegates = []
if args.delegate:
    try:
        delegates = [load_delegate(args.delegate)]
        print(f"delegate loaded: {args.delegate}")
    except Exception as e:
        print(f"WARNING: could not load delegate {args.delegate} ({e}); running on CPU.")

def load(path, use_delegate=False):
    it = Interpreter(model_path=path, num_threads=args.threads,
                     experimental_delegates=(delegates or None) if use_delegate else None)
    it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    print(f"{path} [{'NPU' if use_delegate and delegates else 'CPU'}]: in {np.dtype(i['dtype']).name} {i['shape']} q={i['quantization']} | "
          f"out {np.dtype(o['dtype']).name} {o['shape']} q={o['quantization']}")
    return it, i, o

det, det_in, det_out = load(args.det, use_delegate=True)   # detection on NPU
rec, rec_in, rec_out = load(args.rec)                      # recognition on CPU

def quant(x, d):
    if d["dtype"] == np.float32:
        return x.astype(np.float32)
    s, zp = d["quantization"]
    info = np.iinfo(d["dtype"])
    return np.clip(np.round(x / s + zp), info.min, info.max).astype(d["dtype"])

def dequant(y, d):
    if y.dtype == np.float32:
        return y
    s, zp = d["quantization"]
    return (y.astype(np.float32) - zp) * s

# ---------------- dictionary ----------------
with open(args.dict, encoding="utf-8") as f:
    chars = ["<blank>"] + [l.rstrip("\n") for l in f] + [" "]
n_cls = rec_out["shape"][-1]
print(f"dict classes: {len(chars)}  model classes: {n_cls}")
if len(chars) != n_cls:
    print("WARNING: dictionary size does not match the model output.")

# ---------------- detection ----------------
def order_points(p):
    s = p.sum(1); d = np.diff(p, axis=1).ravel()
    return np.array([p[np.argmin(s)], p[np.argmin(d)], p[np.argmax(s)], p[np.argmax(d)]], np.float32)

def run_det(frame):
    h0, w0 = frame.shape[:2]
    img = cv2.resize(frame, (DET_SIZE, DET_SIZE)).astype(np.float32) / 255.0
    x = ((img - MEAN) / STD)[None]
    det.set_tensor(det_in["index"], quant(x, det_in))
    det.invoke()
    prob = dequant(det.get_tensor(det_out["index"]), det_out)[0, :, :, 0]

    bitmap = (prob > DET_THRESH).astype(np.uint8)
    cnts, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        if len(c) < 4:
            continue
        (cx, cy), (w, h), ang = cv2.minAreaRect(c)
        if min(w, h) < 3:
            continue
        mask = np.zeros(prob.shape, np.uint8)
        cv2.drawContours(mask, [c], -1, 1, -1)
        if (prob * mask).sum() / max(mask.sum(), 1) < BOX_THRESH:
            continue
        dist = (w * h) * UNCLIP / (2 * (w + h))
        pts = cv2.boxPoints(((cx, cy), (w + 2 * dist, h + 2 * dist), ang))
        pts[:, 0] = np.clip(pts[:, 0] * w0 / DET_SIZE, 0, w0 - 1)
        pts[:, 1] = np.clip(pts[:, 1] * h0 / DET_SIZE, 0, h0 - 1)
        boxes.append(order_points(pts))
    return boxes

def crop(frame, pts):
    w = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])))
    h = int(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2])))
    if w < 2 or h < 2:
        return None
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    M = cv2.getPerspectiveTransform(pts, dst)
    img = cv2.warpPerspective(frame, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if h / w >= 1.5:
        img = np.rot90(img)
    return np.ascontiguousarray(img)

# ---------------- recognition ----------------
def run_rec(img):
    h, w = img.shape[:2]
    new_w = min(REC_W, max(1, int(math.ceil(REC_H * w / h))))
    x = (cv2.resize(img, (new_w, REC_H)).astype(np.float32) / 255.0 - 0.5) / 0.5
    canvas = np.zeros((REC_H, REC_W, 3), np.float32)
    canvas[:, :new_w] = x
    rec.set_tensor(rec_in["index"], quant(canvas[None], rec_in))
    rec.invoke()
    out = dequant(rec.get_tensor(rec_out["index"]), rec_out)[0]

    if not (out.min() >= -0.05 and out.max() <= 1.05 and abs(out.sum(-1).mean() - 1) < 0.1):
        e = np.exp(out - out.max(-1, keepdims=True))
        out = e / e.sum(-1, keepdims=True)

    idx, probs = out.argmax(-1), out.max(-1)
    text, conf, last = [], [], 0
    for i, p in zip(idx, probs):
        if i != 0 and i != last and i < len(chars):
            text.append(chars[i]); conf.append(p)
        last = i
    return "".join(text), float(np.mean(conf)) if conf else 0.0

# ---------------- tracking + voting ----------------
class Track:
    _next = 1
    def __init__(self, pts):
        self.id = Track._next
        Track._next += 1
        self.pts = pts
        self.hist = deque(maxlen=HISTORY)     # (text, conf)
        self.miss = 0
        self.shown = ""

    def add(self, text, conf):
        if text and conf >= MIN_READ:
            self.hist.append((text, conf))

    def best(self):
        """Pick the read with the best score: highest confidence, plus a bonus
        for each repeat of the same text. Returns (text, conf) or None."""
        if not self.hist:
            return None
        top, cnt = {}, {}
        for t, c in self.hist:
            top[t] = max(top.get(t, 0.0), c)
            cnt[t] = cnt.get(t, 0) + 1
        t = max(top, key=lambda k: top[k] + 0.1 * (cnt[k] - 1))
        return t, top[t]

def rect_of(pts):
    x, y, w, h = cv2.boundingRect(pts.astype(np.int32))
    return (x, y, x + w, y + h)

def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0

tracks = []

def update_tracks(dets):
    """dets: list of (pts, text, conf) from this OCR cycle."""
    for tr in tracks:
        tr.miss += 1
    pairs = []
    for ti, tr in enumerate(tracks):
        r1 = rect_of(tr.pts)
        for di, (pts, _, _) in enumerate(dets):
            v = iou(r1, rect_of(pts))
            if v >= IOU_MATCH:
                pairs.append((v, ti, di))
    pairs.sort(reverse=True)
    t_used, d_used = set(), set()
    for v, ti, di in pairs:
        if ti in t_used or di in d_used:
            continue
        t_used.add(ti); d_used.add(di)
        pts, text, conf = dets[di]
        tr = tracks[ti]
        tr.pts, tr.miss = pts, 0
        tr.add(text, conf)
    for di, (pts, text, conf) in enumerate(dets):
        if di not in d_used:
            tr = Track(pts)
            tr.add(text, conf)
            tracks.append(tr)
    tracks[:] = [t for t in tracks if t.miss <= MAX_MISS]

def current_results():
    res = []
    for tr in tracks:
        b = tr.best()
        if b is None or tr.miss > SHOW_MISS or b[1] < MIN_SHOW:
            continue
        if b[0] != tr.shown:
            tr.shown = b[0]
            print(f"[#{tr.id}] {b[1]:.2f}  {b[0]}")
        res.append((tr.pts, b[0], b[1]))
    return res

# ---------------- drawing ----------------
def get_font(size=80):
    for p in (args.font, "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"):
        if p and os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    print("WARNING: no TTF font found, non-ASCII text will be tiny. Use --font /path/to/font.ttf")
    return ImageFont.load_default()

font = get_font(args.label_size)   # PIL font, only used for non-ASCII labels
CV_FONT = cv2.FONT_HERSHEY_SIMPLEX

def put_label(img, text, org_x, org_y, size, fg, bg=(0, 0, 0)):
    """Fast ASCII text with a solid background box using cv2 only."""
    scale = size / 30.0
    th_line = max(1, int(round(scale * 2)))
    (tw, th), base = cv2.getTextSize(text, CV_FONT, scale, th_line)
    x = int(min(max(0, org_x), max(0, img.shape[1] - tw - 8)))
    y = int(min(max(th + 6, org_y), img.shape[0] - base - 4))
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 8, y + base + 2), bg, -1)
    cv2.putText(img, text, (x + 4, y), CV_FONT, scale, fg, th_line, cv2.LINE_AA)

def draw(frame, results, disp_fps, ocr_ms, ocr_fps):
    non_ascii = []
    for pts, text, conf in results:
        cv2.polylines(frame, [pts.astype(np.int32)], True, (0, 255, 0), 3)
        label = f"{text} {conf:.2f}"
        x = float(pts[:, 0].min()); y = float(pts[:, 1].min())
        if label.isascii():
            ty = y - 10
            if ty < args.label_size + 6:
                ty = float(pts[:, 1].max()) + args.label_size + 6
            put_label(frame, label, x, ty, args.label_size, (0, 255, 0))
        else:
            non_ascii.append((x, y, pts, label))

    # PIL only when a label really needs unicode glyphs
    if non_ascii:
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(pil)
        for x, y, pts, label in non_ascii:
            l, t, r, b = d.textbbox((0, 0), label, font=font)
            tw, th = r - l, b - t
            ty = y - th - 12
            if ty < 0:
                ty = float(pts[:, 1].max()) + 4
            ty = min(ty, max(0, pil.height - th - 8))
            tx = min(max(0, x), max(0, pil.width - tw - 8))
            d.rectangle([tx, ty, tx + tw + 8, ty + th + 8], fill=(0, 0, 0))
            d.text((tx + 4 - l, ty + 4 - t), label, font=font, fill=(0, 255, 0))
        frame = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    info = f"FPS: {disp_fps:.1f}   OCR: {ocr_ms:.0f} ms ({ocr_fps:.1f}/s)"
    put_label(frame, info, 8, args.fps_size + 8, args.fps_size, (0, 255, 255))
    return frame

# ---------------- threads ----------------
class Camera(threading.Thread):
    """Reads continuously and keeps only the newest frame (no stale buffer lag)."""
    def __init__(self, idx, hd):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # much higher fps than YUYV
        if hd:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.lock = threading.Lock()
        self.frame, self.seq, self.alive = None, 0, True
        print(f"camera: {int(self.cap.get(3))}x{int(self.cap.get(4))} @ {self.cap.get(cv2.CAP_PROP_FPS):.0f} fps")

    def run(self):
        while self.alive:
            ok, f = self.cap.read()
            if not ok:
                self.alive = False
                break
            with self.lock:
                self.frame, self.seq = f, self.seq + 1

    def latest(self, last_seq=-1):
        with self.lock:
            if self.frame is None or self.seq == last_seq:
                return None, last_seq
            return self.frame.copy(), self.seq

    def stop(self):
        self.alive = False
        self.join(timeout=1)
        self.cap.release()

class OcrWorker(threading.Thread):
    """Runs detection + recognition on the newest frame, as fast as it can,
    without ever blocking the display loop."""
    def __init__(self, cam):
        super().__init__(daemon=True)
        self.cam = cam
        self.lock = threading.Lock()
        self.results, self.ms, self.rate, self.alive = [], 0.0, 0.0, True

    def run(self):
        seq, t_prev = -1, None
        while self.alive:
            frame, seq = self.cam.latest(seq)
            if frame is None:
                time.sleep(0.003)
                continue
            t0 = time.perf_counter()
            dets = []
            for pts in run_det(frame):
                c = crop(frame, pts)
                if c is None:
                    continue
                text, conf = run_rec(c)
                dets.append((pts, text, conf))
            update_tracks(dets)
            res = current_results()
            t1 = time.perf_counter()
            dt = (t1 - t0) * 1000
            inst = 1.0 / max(t1 - t_prev, 1e-6) if t_prev else 0.0
            t_prev = t1
            with self.lock:
                self.results = res
                self.ms = dt if self.ms == 0 else 0.7 * self.ms + 0.3 * dt
                self.rate = inst if self.rate == 0 else 0.7 * self.rate + 0.3 * inst

    def get(self):
        with self.lock:
            return self.results, self.ms, self.rate

# ---------------- main loop (display only) ----------------
cam = Camera(args.cam, args.hd)
cam.start()
worker = OcrWorker(cam)
worker.start()

disp_fps, t_prev, seq = 0.0, time.perf_counter(), -1
while cam.alive:
    frame, seq = cam.latest(seq)
    if frame is None:
        time.sleep(0.002)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
        continue

    now = time.perf_counter()
    inst = 1.0 / max(now - t_prev, 1e-6)
    t_prev = now
    disp_fps = inst if disp_fps == 0 else 0.9 * disp_fps + 0.1 * inst

    results, ocr_ms, ocr_fps = worker.get()
    cv2.imshow("PaddleOCR", draw(frame, results, disp_fps, ocr_ms, ocr_fps))
    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
        break

worker.alive = False
cam.stop()
cv2.destroyAllWindows()
