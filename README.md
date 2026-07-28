# RF-LEGO — project website

Source for the [RF-LEGO](https://github.com/aiot-lab/RF-LEGO) project page.

> **RF-LEGO: Modularized Signal Processing–Deep Learning Co-Design for RF Sensing via Deep Unrolling**
> Luca Jiang-Tao Yu, Chenshu Wu — The University of Hong Kong
> ACM MobiCom 2026, Austin, TX, USA

This branch (`website`) contains **only** the static site — no Python package, no
training code. The implementation lives on [`main`](https://github.com/aiot-lab/RF-LEGO/tree/main).

The page covers the teaser video, the abstract and the method (deep unrolling and
the three bricks). Experimental results are left to the paper.

## Layout

```
.
├── index.html               # the whole page (single-page, long scroll)
├── .nojekyll                # serve files verbatim, skip Jekyll
└── static/
    ├── css/
    │   ├── style.css        # design system
    │   └── fonts.css        # @font-face for the self-hosted webfonts
    ├── js/main.js           # theme toggle, scroll progress, KaTeX, copy-BibTeX
    ├── fonts/               # Source Serif 4 / Inter / JetBrains Mono (latin woff2)
    ├── vendor/katex/        # KaTeX 0.16.11, self-hosted
    ├── video/
    │   ├── teaser_rflego.mp4    # 1080p30 H.264 High / AAC-LC, faststart, 5.5 MB
    │   └── teaser_poster.jpg    # poster frame
    └── figures/             # the five method figures from the paper
        ├── rflego_core.svg      # Fig. 1 — the three principles
        ├── deep_unrolling.svg   # the two unrolling patterns
        ├── lego_ft.svg
        ├── lego_bf.svg
        ├── lego_detector.svg
        ├── favicon.svg
        └── og_preview.jpg       # social card
```

No build step, no dependencies to install, **no third-party requests at runtime** —
KaTeX and the webfonts are vendored, so the page renders identically offline, in
restricted networks, and behind privacy blockers. Open `index.html` directly, or
serve the folder:

```bash
python3 -m http.server 8000     # then visit http://localhost:8000
```

Third-party licences: KaTeX (MIT) in `static/vendor/katex/LICENSE`; the three
typefaces (SIL OFL 1.1) in `static/fonts/LICENSE-*`.

## Colours

The palette is sampled from the three bricks in Fig. 1 of the paper, and each
brick owns one module — matching the colour coding of the paper's own module
figures:

| | hex | module |
|---|---|---|
| yellow | `#FFC001` | RF-LEGO FT |
| blue | `#0070C0` | RF-LEGO Beamformer |
| green | `#00B050` | RF-LEGO Detector |

The blue also serves as the link/UI accent — it is the only one of the three that
clears WCAG AA as text on the paper background. Yellow and green appear as fills,
borders and tints only.

## Assets

Figures are regenerated from the paper's LaTeX sources; text is pre-converted to
paths, so the SVGs carry no font dependency:

```bash
pdftocairo -svg fig.pdf fig.svg
```

The teaser is transcoded from the 4K source to the most broadly supported web
profile (H.264 High L4.0, yuv420p, AAC-LC, `moov` up front):

```bash
ffmpeg -i teaser_video_rflego.mov \
  -vf "scale=1920:-2:flags=lanczos,fps=30" \
  -c:v libx264 -profile:v high -level 4.0 -pix_fmt yuv420p -crf 23 \
  -c:a aac -b:a 64k -ac 1 -ar 48000 -movflags +faststart \
  static/video/teaser_rflego.mp4
```

## Publishing on GitHub Pages

Settings → Pages → *Deploy from a branch* → branch `website`, folder `/ (root)`.
The site then serves at `https://aiot-lab.github.io/RF-LEGO/`.

All asset paths are relative, so the page also works from a subdirectory or from
the local filesystem.

## Citation

```bibtex
@inproceedings{luca2026mobicom_rflego,
  author    = {Luca Jiang-Tao Yu and Chenshu Wu},
  title     = {{RF-LEGO}: Modularized Signal Processing-Deep Learning
               Co-Design for {RF} Sensing via Deep Unrolling},
  booktitle = {The 32nd Annual International Conference on Mobile
               Computing and Networking (MobiCom '26)},
  year      = {2026},
  month     = {Oct},
  address   = {Austin, TX, USA},
  publisher = {ACM},
  doi       = {10.1145/3795866.3796683}
}
```
