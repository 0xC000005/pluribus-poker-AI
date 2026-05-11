# Terminal Application

This is the legacy terminal application for playing against old offline agents.
It is separate from the current full-deck Deep CFR and Slumbot pipeline.

The characters are a little broken when captured in `asciinema`, but you'll get the idea by watching this video below. Results should be better in your actual terminal!
[![asciicast](https://asciinema.org/a/331234.png)](https://asciinema.org/a/331234)

For current Slumbot evaluation, use:

```bash
python scripts/play_slumbot.py --model models/slumbot_2p_iter1000.pt --hands 300 --greedy
```
