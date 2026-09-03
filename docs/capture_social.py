"""Writes docs/pharos-social.png -- the 1280x640 card GitHub serves as the repository's
Open Graph image, which is what LinkedIn, Slack and X render when the link is pasted.

Separate from capture_video.py because it is a different shape and a different job: that
script films the terminal at 1080x1350 and grabs a frame for a portrait cover, and a frame
of a 102-column terminal is unreadable once a feed has scaled it to thumbnail width. This
draws one static card at the 2:1 GitHub asks for, with type sized to survive that scaling.

Palette and typefaces are imported rather than restated, so the card cannot drift away from
the video and the shots beside it.

GitHub has no API for this: upload the result by hand at
Settings -> General -> Social preview. Without it GitHub generates a card from the repository
name, description and language stats, which is legible but says nothing this one says.

    uv run --with pillow python docs/capture_social.py
"""

from __future__ import annotations

from capture_video import ACCENT, BG, FG, OUT, UI_B_TTF, UI_TTF
from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 640

# GitHub renders the card at full size; a feed scales it to roughly a third of that. Every
# size below was chosen by looking at the result at 420px wide, not at 1280.
EYEBROW = 25
HEAD = 51
SUB = 27
FOOT = 22

MARGIN = 84
RULE_X = MARGIN
TEXT_X = MARGIN + 30

DIM = (139, 148, 158)  # the video's 90-series grey, one step up for contrast at small sizes

HEADLINE = [
    "A 9B coding model fills a 12 GB card.",
    "What is left will not hold the prompt.",
]
BODY = [
    "Pharos measures what actually fits, cuts the work into ordered",
    "parts that do, runs them one at a time, and names the part that",
    "broke your build.",
]
# No test count and no version: this is a PNG, and a number inside it is stale from the moment
# it is uploaded with nothing to notice. The README carries both, and a test checks them.
FOOT_L = "github.com/ij-jkl/Pharos"
FOOT_R = "MIT  ·  mypy --strict  ·  Python 3.12 | 3.13"


def main() -> None:
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    eyebrow = ImageFont.truetype(UI_B_TTF, EYEBROW)
    head = ImageFont.truetype(UI_B_TTF, HEAD)
    sub = ImageFont.truetype(UI_TTF, SUB)
    foot = ImageFont.truetype(UI_TTF, FOOT)

    y = MARGIN + 6
    # Letterspaced by hand: PIL has no tracking, and the eyebrow is the one line where the
    # difference between a label and a word is doing real work.
    x = TEXT_X
    for ch in "PHAROS":
        draw.text((x, y), ch, font=eyebrow, fill=ACCENT)
        x += draw.textlength(ch, font=eyebrow) + 5

    y += EYEBROW + 40
    rule_top = y - 6
    for line in HEADLINE:
        draw.text((TEXT_X, y), line, font=head, fill=(240, 246, 252))
        y += HEAD + 16
    rule_bottom = y - 16

    # The rule spans the headline only, the way the cover frame does it.
    draw.rectangle([RULE_X, rule_top, RULE_X + 5, rule_bottom], fill=ACCENT)

    y += 26
    for line in BODY:
        draw.text((TEXT_X, y), line, font=sub, fill=DIM)
        y += SUB + 13

    foot_y = H - MARGIN - FOOT
    draw.text((TEXT_X, foot_y), FOOT_L, font=foot, fill=FG)
    right = draw.textlength(FOOT_R, font=foot)
    draw.text((W - MARGIN - right, foot_y), FOOT_R, font=foot, fill=DIM)

    # A hairline off the bottom-left corner, the one piece of the cover frame worth keeping:
    # it reads as a crop mark and stops the card looking like a slide.
    draw.rectangle([0, H - 7, 132, H], fill=ACCENT)

    out = OUT / "pharos-social.png"
    img.save(out, optimize=True)
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB, {W}x{H})")


if __name__ == "__main__":
    main()
