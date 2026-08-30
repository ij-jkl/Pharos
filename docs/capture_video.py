"""Render the 4:5 video from Pharos's own captured terminal output.

Nothing here is mocked up. Every terminal line comes from docs/shot-*.txt, which
capture_cli.py wrote by running the real CLI against the real fixture. This script
parses the ANSI those files already carry, lays it out on a 4:5 canvas, and animates
the reveal so a reader watches the numbers arrive rather than reading a still.

    uv run --with pillow python docs/capture_video.py

Needs ffmpeg on PATH. Writes pharos-video.mp4, pharos-check.gif and pharos-cover.png
beside this file.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

DOCS = Path(__file__).resolve().parent
OUT = DOCS

W, H = 1080, 1350
FPS = 24
PAD_X = 26
BAR_H = 44
CAP_H = 196
TERM_TOP = BAR_H + 16
TERM_BOT = H - CAP_H
COLS = 102

BG = (11, 14, 20)
BAR_BG = (17, 21, 28)
FG = (201, 209, 217)
ACCENT = (57, 197, 207)

NORMAL = {
    30: (72, 79, 88), 31: (241, 112, 106), 32: (63, 185, 80), 33: (210, 153, 34),
    34: (88, 166, 255), 35: (188, 140, 255), 36: (57, 197, 207), 37: (201, 209, 217),
}
BRIGHT = {
    90: (110, 118, 129), 91: (255, 133, 133), 92: (86, 211, 100), 93: (227, 179, 65),
    94: (121, 192, 255), 95: (210, 168, 255), 96: (86, 212, 221), 97: (240, 246, 252),
}

SGR = re.compile(r"\x1b\[([0-9;]*)m")
OTHER_ESC = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def face(*names: str) -> str:
    """First of `names` found in the platform's font directories."""
    roots = [Path(r"C:\Windows\Fonts"), Path("/usr/share/fonts"), Path("/Library/Fonts"),
             Path("/System/Library/Fonts"), Path.home() / ".local/share/fonts"]
    for name in names:
        for root in roots:
            if root.exists():
                hit = next(iter(root.rglob(name)), None)
                if hit is not None:
                    return str(hit)
    raise SystemExit(f"no font found, tried: {', '.join(names)}")


MONO_TTF = face("consola.ttf", "DejaVuSansMono.ttf", "LiberationMono-Regular.ttf")
MONO_B_TTF = face("consolab.ttf", "DejaVuSansMono-Bold.ttf", "LiberationMono-Bold.ttf")
UI_TTF = face("segoeui.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf")
UI_B_TTF = face("segoeuib.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf")


# --- fonts -----------------------------------------------------------------

def pick_size() -> int:
    for size in range(22, 10, -1):
        f = ImageFont.truetype(MONO_TTF, size)
        if f.getlength("M") * COLS <= W - 2 * PAD_X:
            return size
    raise SystemExit("no font size fits")


SIZE = pick_size()
MONO = ImageFont.truetype(MONO_TTF, SIZE)
MONO_B = ImageFont.truetype(MONO_B_TTF, SIZE)
ADV = MONO.getlength("M")
LH = round(SIZE * 1.44)
ROWS = int((TERM_BOT - TERM_TOP - 8) // LH)

UI = ImageFont.truetype(UI_B_TTF, 40)
UI_S = ImageFont.truetype(UI_TTF, 27)
UI_T = ImageFont.truetype(MONO_TTF, 20)
BIG = ImageFont.truetype(UI_B_TTF, 60)
BIG_S = ImageFont.truetype(UI_TTF, 34)


# --- ANSI ------------------------------------------------------------------

def blend(c, bg, a):
    return tuple(round(x * a + y * (1 - a)) for x, y in zip(c, bg, strict=True))


def resolve(fg, bold, dim):
    if fg is None:
        col = (240, 246, 252) if bold else FG
    elif fg in NORMAL:
        col = BRIGHT[fg + 60] if bold else NORMAL[fg]
    else:
        col = BRIGHT[fg]
    return blend(col, BG, 0.44) if dim else col


def parse(line: str):
    """One raw line -> [(text, rgb, bold)] spans, with column positions implicit."""
    line = OTHER_ESC.sub(lambda m: m.group(0) if m.group(0).endswith("m") else "", line)
    spans, pos, fg, bold, dim = [], 0, None, False, False
    for m in SGR.finditer(line):
        text = line[pos:m.start()]
        if text:
            spans.append((text, resolve(fg, bold, dim), bold))
        for raw in (m.group(1) or "0").split(";"):
            code = int(raw or 0)
            if code == 0:
                fg, bold, dim = None, False, False
            elif code == 1:
                bold = True
            elif code == 2:
                dim = True
            elif code == 22:
                bold = dim = False
            elif code == 39:
                fg = None
            elif code in NORMAL or code in BRIGHT:
                fg = code
        pos = m.end()
    tail = line[pos:]
    if tail:
        spans.append((tail, resolve(fg, bold, dim), bold))
    return spans


def load(name):
    return DOCS.joinpath(name).read_text(encoding="utf-8").split("\n")


def find(lines, needle, start=0):
    for i in range(start, len(lines)):
        if needle in lines[i]:
            return i
    raise SystemExit(f"not found: {needle}")


# --- drawing ---------------------------------------------------------------

def tick(d, x, y, col):
    """Consolas has no U+2713, so the check is two strokes on the cell's own grid."""
    a = ADV
    d.line([(x + a * 0.10, y + LH * 0.52), (x + a * 0.40, y + LH * 0.78)], fill=col, width=2)
    d.line([(x + a * 0.40, y + LH * 0.78), (x + a * 0.92, y + LH * 0.24)], fill=col, width=2)


def cross(d, x, y, col):
    a = ADV
    d.line([(x + a * 0.12, y + LH * 0.26), (x + a * 0.88, y + LH * 0.76)], fill=col, width=2)
    d.line([(x + a * 0.88, y + LH * 0.26), (x + a * 0.12, y + LH * 0.76)], fill=col, width=2)


SHAPES = {"\u2713": tick, "\u2717": cross}
# Consolas Bold ships no rounded box corners even though Consolas Regular does, so a
# bold border draws as tofu. Line art has no weight to lose: route these to the regular
# face whatever the span's intensity says.
REG_ONLY = set("\u256d\u256e\u256f\u2570")
SPECIAL = set(SHAPES) | REG_ONLY


def draw_line(d, spans, y):
    col = 0
    for text, rgb, bold in spans:
        font = MONO_B if bold else MONO
        if any(ch in SPECIAL for ch in text):
            for ch in text:
                x = PAD_X + col * ADV
                if ch in SHAPES:
                    SHAPES[ch](d, x, y, rgb)
                elif ch in REG_ONLY:
                    d.text((x, y), ch, font=MONO, fill=rgb)
                elif ch != " ":
                    d.text((x, y), ch, font=font, fill=rgb)
                col += 1
        else:
            d.text((PAD_X + col * ADV, y), text, font=font, fill=rgb)
            col += len(text)


def chrome(d):
    d.rectangle([0, 0, W, BAR_H], fill=BAR_BG)
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse([20 + i * 24, BAR_H // 2 - 6, 32 + i * 24, BAR_H // 2 + 6], fill=c)
    d.text((110, BAR_H // 2 - 11), "pharos", font=UI_T, fill=(110, 118, 129))
    who = "Isaac Jordan"
    d.text((W - PAD_X - UI_T.getlength(who), BAR_H // 2 - 11), who,
           font=UI_T, fill=(92, 100, 112))
    d.line([(0, BAR_H), (W, BAR_H)], fill=(30, 36, 46), width=1)


def caption(d, main, sub, progress):
    top = TERM_BOT
    d.rectangle([0, top, W, H], fill=BAR_BG)
    d.line([(0, top), (W, top)], fill=(30, 36, 46), width=1)
    d.rectangle([PAD_X, top + 34, PAD_X + 5, top + 34 + 44], fill=ACCENT)
    d.text((PAD_X + 22, top + 30), main, font=UI, fill=(240, 246, 252))
    for i, s in enumerate(sub.split("\n")[:2]):
        d.text((PAD_X + 22, top + 84 + i * 34), s, font=UI_S, fill=(139, 148, 158))
    d.rectangle([0, H - 6, W, H], fill=(30, 36, 46))
    d.rectangle([0, H - 6, int(W * progress), H], fill=ACCENT)


def render(buf, scroll, cap, sub, progress):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    chrome(d)
    first = int(scroll)
    frac = scroll - first
    for r in range(ROWS + 1):
        i = first + r
        if 0 <= i < len(buf):
            y = TERM_TOP + (r - frac) * LH
            if TERM_TOP - LH < y < TERM_BOT:
                draw_line(d, buf[i], y)
    d.rectangle([0, TERM_BOT - 1, W, TERM_BOT + 1], fill=BAR_BG)
    caption(d, cap, sub, progress)
    return img


def card(lines, sub_lines, progress):
    """A full-bleed title/end frame: no terminal, just the claim."""
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    y = H // 2 - (len(lines) * 84 + len(sub_lines) * 48) // 2
    d.rectangle([PAD_X + 30, y - 4, PAD_X + 36, y + len(lines) * 84 - 20], fill=ACCENT)
    for ln in lines:
        d.text((PAD_X + 62, y), ln, font=BIG, fill=(240, 246, 252))
        y += 84
    y += 26
    for i, ln in enumerate(sub_lines):
        d.text((PAD_X + 62, y), ln, font=BIG_S,
               fill=(214, 222, 232) if i == 0 else (139, 148, 158))
        y += 48
    d.rectangle([0, H - 6, W, H], fill=(30, 36, 46))
    d.rectangle([0, H - 6, int(W * progress), H], fill=ACCENT)
    return img


GREEN = (63, 185, 80)
RED = (241, 112, 106)
UI_H = ImageFont.truetype(UI_B_TTF, 30)
UI_K = ImageFont.truetype(MONO_TTF, 21)
UI_V = ImageFont.truetype(MONO_B_TTF, 21)
UI_N = ImageFont.truetype(MONO_TTF, 18)

LEFT = {
    "head": "game fixture",
    "meta": "6 files  ·  --semantic  ·  qwen3.5:9b",
    "verdict": "COMPLETE",
    "colour": GREEN,
    "rows": [
        ("coverage", "100%   6 of 6 files"),
        ("verification", "OK  syntax"),
        ("parts", "3, each wrote both its files"),
        ("exit", "0"),
    ],
    "why": "Every assigned file changed,\nand the project still builds.",
    "src": "DESKTOP_VALIDATION.md",
}
RIGHT = {
    "head": "shop fixture",
    "meta": "25 files  ·  position packing  ·  qwen3.5:9b",
    "verdict": "FAILED",
    "colour": RED,
    "rows": [
        ("coverage", "88%   22 of 25 files"),
        ("verification", "3 checks broke"),
        ("damage", "parts 4, 6 and 7 named"),
        ("exit", "1"),
    ],
    "why": "A part could not finish, and\nchecks that passed now fail.",
    "src": "docs/shot-run.txt",
}


def column(d, spec, x0, shown):
    """One run's result. `shown` is how many rows have been revealed so far."""
    cx = x0 + 30
    d.rectangle([x0 + 14, 170, x0 + 19, 170 + 98], fill=spec["colour"])
    d.text((cx, 172), spec["head"], font=UI_H, fill=(240, 246, 252))
    d.text((cx, 214), spec["meta"], font=UI_N, fill=(120, 128, 140))
    if shown >= 1:
        d.text((cx, 292), spec["verdict"], font=BIG, fill=spec["colour"])
    y = 414
    for i, (k, v) in enumerate(spec["rows"]):
        if shown >= i + 2:
            d.text((cx, y), k, font=UI_K, fill=(130, 138, 150))
            d.text((cx, y + 30), v, font=UI_V, fill=(216, 224, 232))
        y += 112
    if shown >= len(spec["rows"]) + 2:
        yy = y + 18
        for ln in spec["why"].split("\n"):
            d.text((cx, yy), ln, font=UI_N, fill=spec["colour"])
            yy += 28
        d.text((cx, yy + 22), spec["src"], font=UI_N, fill=(96, 104, 116))


def panel(shown_l, shown_r, progress):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((PAD_X + 30, 62), "Two runs that really happened.",
           font=UI, fill=(240, 246, 252))
    d.line([(W // 2, 158), (W // 2, TERM_BOT - 34)], fill=(30, 36, 46), width=1)
    column(d, LEFT, 0, shown_l)
    column(d, RIGHT, W // 2, shown_r)
    caption(d, "Same tool. Same rules.",
            "The verdict follows the code, not the effort.\nNothing above was decided by a model.",
            progress)
    return img


# --- timeline --------------------------------------------------------------

class Film:
    def __init__(self, proc):
        self.proc = proc
        self.buf = []
        self.scroll = 0.0
        self.cap = ""
        self.sub = ""
        self.n = 0
        self.total = 1
        self._last = None

    def push(self, img):
        self.proc.stdin.write(img.tobytes())
        self.n += 1

    def frame(self):
        target = max(0.0, len(self.buf) - ROWS)
        self.scroll += (target - self.scroll) * 0.22
        if abs(target - self.scroll) < 0.01:
            self.scroll = target
        self.push(render(self.buf, self.scroll, self.cap, self.sub, self.n / self.total))

    def wait(self, seconds):
        for _ in range(round(seconds * FPS)):
            self.frame()

    def clear(self):
        self.buf, self.scroll = [], 0.0

    def say(self, main, sub=""):
        self.cap, self.sub = main, sub

    def type(self, cmd, cps=34):
        self.buf.append([])
        shown = ""
        for ch in cmd:
            shown += ch
            self.buf[-1] = parse("\x1b[32m>\x1b[0m \x1b[1m" + shown + "\x1b[0m")
            for _ in range(max(1, round(FPS / cps))):
                self.frame()
        self.buf.append([])

    def emit(self, lines, lps=40.0):
        step = 1.0 / lps
        carry = 0.0
        for raw in lines:
            self.buf.append(parse(raw))
            carry += step * FPS
            while carry >= 1:
                self.frame()
                carry -= 1
        if carry > 0:
            self.frame()

    def card(self, lines, subs, seconds):
        for _ in range(round(seconds * FPS)):
            self.push(card(lines, subs, self.n / self.total))

    def panel(self, hold):
        steps = len(LEFT["rows"]) + 2
        for i in range(steps + 1):
            for _ in range(round(0.30 * FPS)):
                self.push(panel(i, max(0, i - 1), self.n / self.total))
        for _ in range(round(hold * FPS)):
            self.push(panel(steps, steps, self.n / self.total))


def build(film: Film):
    check = load("shot-check.txt")
    split = load("shot-split.txt")
    run = load("shot-run.txt")

    chk = check[: find(check, "=====") - 1]
    sp_start = find(split, "Split plan")
    sp = split[sp_start : find(split, "part 2/2", sp_start) + 8]
    run_head = run[: find(run, "Scorecard") - 1]
    sc_start = find(run, "Scorecard")
    sc_end = find(run, "Review \u2014 one model's opinion")
    scorecard = run[sc_start - 1 : sc_end - 1]
    review = run[sc_end - 1 : find(run, "It did not decide anything") + 2]

    film.card(
        ["A 9B coding model fills", "a 12 GB card.", "What is left will not", "hold the prompt."],
        ["The usual answer is a smaller model, or less work.",
         "Pharos measures what actually fits."],
        3.0,
    )

    film.say("Does it fit?",
             "Counted with the model's own tokenizer.\n"
             "Every line labelled exact, not estimated.")
    film.type("pharos check --file prompt.txt")
    film.wait(0.35)
    film.emit(chk[:30], lps=33)
    film.emit(chk[30:], lps=10)
    film.say("No.", "The floor alone is 509 tokens over the usable budget.")
    film.wait(2.0)

    film.clear()
    film.say("So cut it.",
             "Every part repeats the task and narrows the scope.\n"
             "A part cannot open a file it was not given.")
    film.type("pharos split --file prompt.txt")
    film.wait(0.35)
    film.emit(sp, lps=9)
    film.wait(2.6)

    film.clear()
    film.say("Then carry it out.", "One part, one conversation, one window.")
    film.type("pharos run --compact --review")
    film.wait(0.35)
    film.emit(run_head[:7], lps=6)
    film.emit(run_head[7:], lps=2.9)
    film.wait(1.0)

    film.say("And say what happened.", "Six questions. None of them asked of a model.")
    film.emit(scorecard, lps=8.5)
    film.wait(1.6)

    film.say("The verdict is measured.",
             "Coverage is writes over scope. Damage is parsed.\n"
             "Verification is your own ruff and pytest.")
    film.wait(1.4)

    film.say("The opinion comes last.",
             "Under a verdict it cannot change, with every finding\n"
             "checked against the diff first.")
    film.emit(review, lps=7)
    film.wait(2.2)

    film.panel(hold=3.4)

    film.card(
        ["Pharos", "", "Measure the window.", "Then fill it."],
        ["Isaac Jordan  ·  github.com/ij-jkl/Pharos",
         "750 tests  ·  mypy --strict  ·  MIT"],
        3.8,
    )


def main():
    OUT.mkdir(exist_ok=True)
    mp4 = OUT / "pharos-video.mp4"

    # Pass one counts frames so the progress bar is truthful; pass two encodes.
    class Counter(Film):
        def push(self, _img):
            self.n += 1

    dry = Counter(None)
    build(dry)
    total = dry.n
    print(f"font {SIZE}px  ·  {ROWS} rows  ·  {total} frames  ·  {total / FPS:.1f}s", flush=True)

    cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
        "-c:v", "libx264", "-preset", "slow", "-crf", "20",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(mp4),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    film = Film(proc)
    film.total = total
    build(film)
    proc.stdin.close()
    proc.wait()
    print(f"wrote {mp4} ({mp4.stat().st_size / 1e6:.1f} MB)", flush=True)

    # The README wants the check arc alone: short enough to loop, wide enough to read.
    gif, cover, pal = OUT / "pharos-check.gif", OUT / "pharos-cover.png", OUT / "_pal.png"
    run(["ffmpeg", "-loglevel", "error", "-y", "-ss", "1.6", "-i", str(mp4),
         "-frames:v", "1", str(cover)])
    trim = ["-ss", "3.0", "-to", "9.3", "-i", str(mp4)]
    vf = "fps=10,scale=720:-1:flags=lanczos"
    run(["ffmpeg", "-loglevel", "error", "-y", *trim, "-vf",
         vf + ",palettegen=max_colors=128:stats_mode=diff", str(pal)])
    run(["ffmpeg", "-loglevel", "error", "-y", *trim, "-i", str(pal), "-lavfi",
         f"{vf}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle",
         str(gif)])
    pal.unlink()
    for f in (gif, cover):
        print(f"wrote {f} ({f.stat().st_size / 1e6:.1f} MB)", flush=True)


def run(cmd):
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
