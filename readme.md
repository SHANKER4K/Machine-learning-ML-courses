# Machine-learning-ML-courses

Notes, exercises, and code snippets from various Machine Learning courses collected in one place for reference and practice.

## Repository goals

- Keep course materials organized by provider/course
- Store runnable examples and small experiments
- Track progress and revisit key concepts quickly

## Structure

This repo is organized by course/provider folders. A common layout looks like:

If your folders differ, adjust this README accordingly.

## Getting started

### Prerequisites

- Python 3.9+ (recommended)
- `pip` or `conda`

### Setup (venv)

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

pip install -U pip
```

If a `requirements.txt` exists:

```bash
pip install -r requirements.txt
```

## Usage

- Browse course folders for notes and scripts.
- Run Python files directly:

```bash
python path/to/script.py
```

- If notebooks are included, start Jupyter:

```bash
pip install notebook
jupyter notebook
```

## Notes

- This repository is intended for learning and personal reference.
- Avoid committing course-provided copyrighted content; prefer your own notes and original code.

## Contributing

If you want to extend this repo:

1. Create a new folder for the course/provider
2. Add a short `README.md` inside with:
   - course name
   - topics covered
   - key takeaways
3. Keep scripts small and well-named (one concept per file when possible)

## License

Add a `LICENSE` file if you plan to share this publicly. Until then, treat it as private study material.
