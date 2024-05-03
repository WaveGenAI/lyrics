# Lyrics Generator

It's a wrapper of the Whisper model to generate lyrics of an audio file.

# Usage

```python
from src.main import LyricGen

lyric_gen = LyricGen()
out = lyric_gen.generate_lyrics("COVEX - Good Side (ft. Delaney Jane).mp3")

print(out)
```

# Installation

```bash
pip install -r requirements.txt
```
