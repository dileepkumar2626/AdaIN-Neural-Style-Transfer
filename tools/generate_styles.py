
import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SIZE = 768
OUT = Path(__file__).resolve().parent.parent / "static" / "styles"
THUMBS = OUT / "thumbs"


def rng_for(seed):
    return random.Random(seed), np.random.default_rng(seed)


def hsv(h, s, v):
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, max(0, min(1, s)), max(0, min(1, v)))
    return int(r * 255), int(g * 255), int(b * 255)


def noise_field(nrng, scale):
    """Smooth value noise in [0,1], SIZE x SIZE."""
    n = max(2, SIZE // scale)
    base = nrng.random((n + 1, n + 1))
    img = Image.fromarray((base * 255).astype(np.uint8)).resize((SIZE, SIZE), Image.BICUBIC)
    return np.asarray(img).astype(np.float32) / 255.0


def palette_map(t, stops):
    """Map array t in [0,1] to RGB using colour stops [(pos,(r,g,b)),...]."""
    pos = np.array([p for p, _ in stops])
    cols = np.array([c for _, c in stops], dtype=np.float32)
    out = np.zeros(t.shape + (3,), dtype=np.float32)
    for ch in range(3):
        out[..., ch] = np.interp(t, pos, cols[:, ch])
    return out


def to_img(arr):
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def add_grain(img, nrng, amount=10):
    a = np.asarray(img).astype(np.float32)
    a += nrng.normal(0, amount, a.shape[:2])[..., None]
    return to_img(a)
def swirl_night(seed=1):
    r, nrng = rng_for(seed)
    ang = noise_field(nrng, 160) * math.tau * 2
    img = Image.new("RGB", (SIZE, SIZE), (14, 28, 70))
    d = ImageDraw.Draw(img)
    cols = [(24, 54, 128), (44, 92, 170), (240, 205, 80), (120, 170, 220), (18, 36, 90), (250, 235, 150)]
    for _ in range(5200):
        x, y = r.uniform(0, SIZE), r.uniform(0, SIZE)
        c = r.choice(cols if y > SIZE * .25 else cols[:5])
        pts = [(x, y)]
        for _ in range(14):
            a = ang[int(min(SIZE - 1, max(0, y))), int(min(SIZE - 1, max(0, x)))]
            x += math.cos(a) * 6
            y += math.sin(a) * 6
            pts.append((x, y))
        d.line(pts, fill=c, width=r.randint(4, 9))
    return img.filter(ImageFilter.SMOOTH)


def neon_grid(seed=2):
    r, nrng = rng_for(seed)
    img = Image.new("RGB", (SIZE, SIZE), (10, 6, 28))
    d = ImageDraw.Draw(img)
    for i in range(0, SIZE, 48):
        c1 = hsv(.85 - i / SIZE * .35, .9, 1)
        c2 = hsv(.5 + i / SIZE * .2, .9, 1)
        d.line([(i, 0), (i, SIZE)], fill=c1, width=3)
        d.line([(0, i), (SIZE, i)], fill=c2, width=3)
    for _ in range(40):
        x, y = r.randrange(0, SIZE, 48), r.randrange(0, SIZE, 48)
        d.ellipse([x - 9, y - 9, x + 9, y + 9], fill=hsv(r.random(), .8, 1))
    glow = img.filter(ImageFilter.GaussianBlur(10))
    return Image.blend(img, glow, .55).point(lambda v: min(255, int(v * 1.25)))


def stained_glass(seed=3):
    r, nrng = rng_for(seed)
    n = 70
    pts = nrng.random((n, 2)) * SIZE
    hues = nrng.random(n)
    small = 192
    ys, xs = np.mgrid[0:small, 0:small] * (SIZE / small)
    dist = (xs[..., None] - pts[:, 0]) ** 2 + (ys[..., None] - pts[:, 1]) ** 2
    idx = dist.argmin(-1)
    srt = np.sort(dist, -1)
    edge = (np.sqrt(srt[..., 1]) - np.sqrt(srt[..., 0])) < 6
    cols = np.array([hsv(h, .85, .9) for h in hues], dtype=np.float32)
    arr = cols[idx]
    arr *= (0.8 + 0.2 * nrng.random((small, small, 1)))
    arr[edge] = (18, 14, 20)
    return to_img(arr).resize((SIZE, SIZE), Image.NEAREST).filter(ImageFilter.GaussianBlur(1.1))


def cubist_shards(seed=4):
    r, nrng = rng_for(seed)
    pal = [(196, 150, 96), (122, 84, 58), (226, 200, 150), (70, 92, 100), (160, 60, 48), (40, 36, 34), (210, 170, 110)]
    img = Image.new("RGB", (SIZE, SIZE), (200, 170, 120))
    d = ImageDraw.Draw(img)
    for _ in range(260):
        cx, cy = r.uniform(0, SIZE), r.uniform(0, SIZE)
        k = r.randint(3, 5)
        rad = r.uniform(40, 190)
        a0 = r.uniform(0, math.tau)
        pts = [(cx + math.cos(a0 + j * math.tau / k + r.uniform(-.5, .5)) * rad * r.uniform(.5, 1),
                cy + math.sin(a0 + j * math.tau / k + r.uniform(-.5, .5)) * rad * r.uniform(.5, 1)) for j in range(k)]
        d.polygon(pts, fill=r.choice(pal), outline=(30, 24, 20))
    return img


def pointillism(seed=5):
    r, nrng = rng_for(seed)
    f1, f2 = noise_field(nrng, 200), noise_field(nrng, 120)
    img = Image.new("RGB", (SIZE, SIZE), (245, 238, 220))
    d = ImageDraw.Draw(img)
    for _ in range(26000):
        x, y = r.randrange(SIZE), r.randrange(SIZE)
        h = .08 + f1[y, x] * .75
        c = hsv(h + r.uniform(-.04, .04), .55 + .35 * f2[y, x], .75 + .25 * r.random())
        rad = r.randint(3, 6)
        d.ellipse([x - rad, y - rad, x + rad, y + rad], fill=c)
    return img


def watercolor(seed=6):
    r, nrng = rng_for(seed)
    img = Image.new("RGB", (SIZE, SIZE), (250, 246, 238))
    for _ in range(34):
        layer = Image.new("RGB", (SIZE, SIZE), (255, 255, 255))
        mask = Image.new("L", (SIZE, SIZE), 0)
        d = ImageDraw.Draw(mask)
        x, y, rad = r.uniform(0, SIZE), r.uniform(0, SIZE), r.uniform(80, 260)
        d.ellipse([x - rad, y - rad * .8, x + rad, y + rad * .8], fill=r.randint(150, 230))
        mask = mask.filter(ImageFilter.GaussianBlur(rad * .22))
        col = hsv(r.choice([.52, .58, .92, .05, .13, .75]) + r.uniform(-.03, .03), .85, .9)
        layer = Image.new("RGB", (SIZE, SIZE), col)
        img = Image.composite(Image.blend(img, layer, .75), img, mask)
    return add_grain(img, nrng, 5)


def ink_sketch(seed=7):
    r, nrng = rng_for(seed)
    tone = noise_field(nrng, 140)
    img = Image.new("RGB", (SIZE, SIZE), (243, 237, 222))
    d = ImageDraw.Draw(img)
    for angle, thr in ((35, .35), (-35, .55), (90, .72)):
        ca, sa = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        for off in range(-SIZE, SIZE * 2, 9):
            x0, y0 = off * ca, off * sa
            pts, on = [], False
            for t in range(-SIZE, SIZE * 2, 6):
                x, y = x0 - t * sa, y0 + t * ca
                if 0 <= x < SIZE and 0 <= y < SIZE and tone[int(y), int(x)] > thr:
                    pts.append((x + r.uniform(-.8, .8), y + r.uniform(-.8, .8)))
                else:
                    if len(pts) > 1:
                        d.line(pts, fill=(25, 25, 32), width=1)
                    pts = []
            if len(pts) > 1:
                d.line(pts, fill=(25, 25, 32), width=1)
    return add_grain(img, nrng, 4)


def seigaiha_waves(seed=8):
    r, nrng = rng_for(seed)
    img = Image.new("RGB", (SIZE, SIZE), (20, 40, 90))
    d = ImageDraw.Draw(img)
    rad = 64
    for row in range(-1, SIZE // (rad // 2) + 2):
        y = row * rad // 2
        for col in range(-1, SIZE // rad + 2):
            x = col * rad + (rad // 2 if row % 2 else 0)
            for k, c in enumerate([(235, 240, 245), (40, 80, 150), (235, 240, 245), (30, 62, 130), (225, 232, 240)]):
                rr = rad // 2 + 2 - k * 6
                if rr > 0:
                    d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=c, outline=(15, 30, 70))
    return img


def pop_halftone(seed=9):
    r, nrng = rng_for(seed)
    img = Image.new("RGB", (SIZE, SIZE), (250, 226, 60))
    d = ImageDraw.Draw(img)
    step = 18
    cx, cy = SIZE * .35, SIZE * .4
    for gy in range(0, SIZE + step, step):
        for gx in range(0, SIZE + step, step):
            dd = math.hypot(gx - cx, gy - cy) / SIZE
            rad = max(1.5, step * .52 * (1 - min(1, dd * 1.25)))
            col = (225, 30, 50) if (gx // step + gy // step) % 2 else (20, 110, 210)
            ox = step / 2 if (gy // step) % 2 else 0
            d.ellipse([gx + ox - rad, gy - rad, gx + ox + rad, gy + rad], fill=col)
    d.rectangle([0, SIZE * .72, SIZE, SIZE * .78], fill=(15, 15, 15))
    d.rectangle([0, SIZE * .82, SIZE, SIZE * .84], fill=(15, 15, 15))
    return img


def mosaic_tiles(seed=10):
    r, nrng = rng_for(seed)
    img = Image.new("RGB", (SIZE, SIZE), (60, 55, 50))
    d = ImageDraw.Draw(img)
    f = noise_field(nrng, 150)
    t = 24
    for y in range(0, SIZE, t):
        for x in range(0, SIZE, t):
            h = .5 + f[y, x] * .45
            col = hsv(h + r.uniform(-.03, .03), .6 + r.uniform(-.15, .2), .55 + .4 * r.random())
            j = r.randint(0, 3)
            d.rectangle([x + 2 + j, y + 2, x + t - 2, y + t - 2 - j], fill=col)
    return img


def sunset_strokes(seed=11):
    r, nrng = rng_for(seed)
    stops = [(0, (30, 20, 80)), (.3, (150, 40, 120)), (.55, (240, 90, 60)), (.8, (255, 170, 60)), (1, (255, 230, 150))]
    ys = np.linspace(1, 0, SIZE)[:, None] * np.ones((1, SIZE))
    img = to_img(palette_map(ys, stops))
    d = ImageDraw.Draw(img)
    for _ in range(3800):
        x, y = r.uniform(-30, SIZE), r.uniform(0, SIZE)
        t = 1 - y / SIZE + r.uniform(-.08, .08)
        c = tuple(int(v) for v in palette_map(np.array([max(0, min(1, t))]), stops)[0])
        d.line([(x, y), (x + r.uniform(30, 80), y + r.uniform(-4, 4))], fill=c, width=r.randint(5, 11))
    d.ellipse([SIZE * .55, SIZE * .58, SIZE * .85, SIZE * .88], fill=(255, 238, 170))
    return img.filter(ImageFilter.SMOOTH_MORE)


def psychedelic(seed=12):
    ys, xs = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    cx = cy = SIZE / 2
    dx, dy = xs - cx, ys - cy
    rr, th = np.hypot(dx, dy), np.arctan2(dy, dx)
    t = np.sin(rr / 22 + th * 4) * .5 + .5
    t2 = np.sin(rr / 40 - th * 3 + 2) * .5 + .5
    arr = np.stack([np.sin(t * 6.28) * .5 + .5, np.sin(t2 * 6.28 + 2) * .5 + .5, np.sin((t + t2) * 6.28 + 4) * .5 + .5], -1) * 255
    return to_img(arr)


def impasto(seed=13):
    r, nrng = rng_for(seed)
    f = noise_field(nrng, 180)
    img = Image.new("RGB", (SIZE, SIZE), (90, 50, 40))
    d = ImageDraw.Draw(img)
    for _ in range(9000):
        x, y = r.uniform(0, SIZE), r.uniform(0, SIZE)
        h = (f[int(y), int(x)] * 1.2 + r.uniform(-.06, .06)) % 1
        c = hsv(h * .8 + .02, .75 + r.uniform(-.2, .2), .6 + r.uniform(0, .4))
        a = r.uniform(0, math.tau)
        L = r.uniform(14, 34)
        w = r.randint(7, 14)
        d.line([(x, y), (x + math.cos(a) * L, y + math.sin(a) * L)], fill=c, width=w)
    return img


def pastel_dream(seed=14):
    r, nrng = rng_for(seed)
    n1, n2, n3 = noise_field(nrng, 220), noise_field(nrng, 160), noise_field(nrng, 90)
    arr = np.stack([200 + 55 * n1, 190 + 60 * n2, 215 + 40 * n3], -1)
    arr[..., 0] += 25 * np.sin(n2 * 6)
    img = to_img(arr).filter(ImageFilter.GaussianBlur(3))
    d = ImageDraw.Draw(img, "RGBA")
    for _ in range(140):
        x, y, rad = r.uniform(0, SIZE), r.uniform(0, SIZE), r.uniform(10, 60)
        d.ellipse([x - rad, y - rad, x + rad, y + rad], fill=hsv(r.random(), .25, 1) + (r.randint(50, 110),))
    return add_grain(img.filter(ImageFilter.GaussianBlur(1.5)), nrng, 5)


def bauhaus(seed=15):
    r, nrng = rng_for(seed)
    img = Image.new("RGB", (SIZE, SIZE), (238, 228, 205))
    d = ImageDraw.Draw(img)
    pal = [(200, 40, 40), (240, 190, 40), (30, 70, 150), (25, 25, 25), (60, 130, 90)]
    for _ in range(46):
        x, y, s = r.randrange(0, SIZE, 24), r.randrange(0, SIZE, 24), r.choice([48, 72, 96, 144])
        c, kind = r.choice(pal), r.choice(["circle", "rect", "tri", "half"])
        if kind == "circle":
            d.ellipse([x, y, x + s, y + s], fill=c)
        elif kind == "rect":
            d.rectangle([x, y, x + s, y + s // 2], fill=c)
        elif kind == "tri":
            d.polygon([(x, y + s), (x + s / 2, y), (x + s, y + s)], fill=c)
        else:
            d.pieslice([x, y, x + s, y + s], 180, 360, fill=c)
    for _ in range(14):
        x = r.randrange(0, SIZE, 24)
        d.line([(x, 0), (x + r.randint(-120, 120), SIZE)], fill=(25, 25, 25), width=3)
    return img


def art_deco(seed=16):
    img = Image.new("RGB", (SIZE, SIZE), (12, 12, 16))
    d = ImageDraw.Draw(img)
    cx = SIZE / 2
    gold = [(212, 175, 55), (240, 210, 110), (170, 130, 40)]
    for k in range(14, 0, -1):
        rad = k * 28
        d.ellipse([cx - rad, SIZE - rad * .0 - rad, cx + rad, SIZE + rad], outline=gold[k % 3], width=3)
    for i in range(36):
        a = math.pi + i * math.pi / 35
        d.line([(cx, SIZE), (cx + math.cos(a) * SIZE * 1.2, SIZE + math.sin(a) * SIZE * 1.2)], fill=gold[i % 3], width=2)
    for x in range(0, SIZE, 64):
        d.rectangle([x, 0, x + 6, SIZE * .18], fill=gold[x // 64 % 3])
        d.polygon([(x - 20, SIZE * .18), (x + 3, SIZE * .26), (x + 26, SIZE * .18)], fill=gold[(x // 64 + 1) % 3])
    return img
STYLES = [
    ("swirl-night", "Swirling Night", "Painting", "Flowing brushwork in deep blues and gold", swirl_night),
    ("impasto", "Thick Oil", "Painting", "Heavy, textured oil strokes in rich colour", impasto),
    ("watercolor", "Watercolor Wash", "Painting", "Soft, bleeding washes on paper", watercolor),
    ("pointillism", "Pointillism", "Painting", "Thousands of tiny dots of colour", pointillism),
    ("sunset-strokes", "Sunset Strokes", "Painting", "Warm gradient painted with broad strokes", sunset_strokes),
    ("cubist-shards", "Cubist Shards", "Abstract", "Overlapping angular planes in earth tones", cubist_shards),
    ("bauhaus", "Bauhaus", "Abstract", "Primary-colour geometry on cream paper", bauhaus),
    ("pastel-dream", "Pastel Dream", "Abstract", "Soft, hazy pastel clouds", pastel_dream),
    ("psychedelic", "Psychedelic", "Abstract", "Vivid rainbow spirals", psychedelic),
    ("neon-grid", "Neon Grid", "Digital", "Glowing synthwave lines on midnight", neon_grid),
    ("art-deco", "Art Deco Gold", "Digital", "Gold fans and rays on black", art_deco),
    ("pop-halftone", "Pop Art", "Pattern", "Comic-book halftone dots", pop_halftone),
    ("stained-glass", "Stained Glass", "Pattern", "Jewel colours divided by dark lead lines", stained_glass),
    ("mosaic", "Mosaic Tiles", "Pattern", "Hand-laid ceramic tiles", mosaic_tiles),
    ("waves", "Indigo Waves", "Pattern", "Traditional overlapping wave scales", seigaiha_waves),
    ("ink-sketch", "Ink Sketch", "Sketch", "Cross-hatched pen on cream paper", ink_sketch),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    THUMBS.mkdir(parents=True, exist_ok=True)
    manifest = []
    for sid, name, cat, desc, fn in STYLES:
        img = fn().convert("RGB")
        img.save(OUT / f"{sid}.jpg", quality=90)
        thumb = img.copy()
        thumb.thumbnail((240, 240))
        thumb.save(THUMBS / f"{sid}.jpg", quality=85)
        manifest.append({"id": sid, "name": name, "category": cat, "description": desc, "file": f"{sid}.jpg"})
        print("made", sid)
    (OUT / "styles.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"{len(manifest)} styles written to {OUT}")


if __name__ == "__main__":
    main()
