# indic-xlit

An HTTP service wrapping [ai4bharat-transliteration](https://github.com/AI4Bharat/IndicXlit)
(IndicXlit) — bidirectional transliteration between Roman script and 9
Indic scripts: Hindi, Bengali, Punjabi (Gurmukhi), Gujarati, Odia, Tamil,
Telugu, Kannada, and Malayalam.

Per-language engines are loaded lazily on first use and cached in memory —
not all 9+ languages upfront, since a given deployment usually only needs
one or two.

## Run it

```bash
bash setup.sh
uv run server.py   # PORT=8006 by default
```

## API

### `GET /health`

```json
{"status": "ok"}
```

### `POST /transliterate`

**Indic → Roman** — convert Indic-script runs to Roman-script
approximations, leaving Latin text/digits/punctuation untouched:

```json
{"mode": "indic-to-roman", "text": "नमस्ते दुनिया"}
```

```json
{"text": "namaste duniya"}
```

**Roman → Indic** — transliterate a batch of Roman-script words (assumed
already language-tagged as `lang` by an upstream word-language-ID step —
this mode does *not* itself decide which words belong to which language)
into their native-script spelling:

```json
{"mode": "roman-to-indic", "words": ["kal", "chalo"], "lang": "hi"}
```

```json
{"translations": {"kal": "कल", "chalo": "चलो"}}
```

A word that fails to transliterate maps to itself unchanged.

**On error**: an HTTP error status (422 for an unknown `mode`, 500 for an
engine failure) with a JSON `{"detail": "..."}` body.

## License

IndicXlit is released by [AI4Bharat](https://ai4bharat.iitm.ac.in/) — see
the [upstream repository](https://github.com/AI4Bharat/IndicXlit) for
licensing terms.
