# Cat vs. Dog Image Classification Using Deep Learning in Python
**Carter Susi**

**Dec 02, 2025**

**CAP4613**

**Repo**:\
[https://github.com/cartersusi/susi-ml-bonus](https://github.com/cartersusi/susi-ml-final)

## Requirements

[**python3 (venv)**](requirements.txt)
```sh
pip install -r requirements.txt
```
***or***\
[**python3 (uv)**](pyproject.toml)
```sh
uv sync
```

---

[**Config**](conf.json)
```json
{
  "models_dir": "/[user]/[prj]/models",
  "kaggle_creds": "/[user]/.kaggle/kaggle.json",
  "model_path": "/[user]/[prj]/models/cat_dog_cnn_25.pth"
}
```

---

[**Params**](main.py#L103)


*Visualization*: (optional)
- `-v` `--visualize`
```sh
python3 main.py -v
```

<hr style="width: 66%; margin-left: 0px;"/>

*Inference*: (optional)
- `-i` `--inference`
  - **port**: Port number to serve the model (e.g., `8080`)
  - **filename**: Path to image file for inference (e.g., `cat.jpg`)
```sh
python3 main.py -i 8080      # Serve on port
python3 main.py -i cat.jpg   # Infer on file
```

<hr style="width: 66%; margin-left: 0px;"/>

*Kaggle Credits*: (optional)
- `-k` `--kaggle_creds`
- Overrides **conf["kaggle_creds"]**
  - filename: Path to credential file for kaggle (e.g., `~/.kaggle/kaggle.json`)
```sh
python3 main.py -k ~/.kaggle/kaggle.json
```

<hr style="width: 66%; margin-left: 0px;"/>

*Models Directory*: (optional)
- `-o` `--models_dir`
- Overrides **conf["models_dir"]**
  - dirname: Path to output directory for trained models (e.g., `./models/`)
```sh
python3 main.py -o ./models/
```

<hr style="width: 66%; margin-left: 0px;"/>

*Image Size*: (optional)
- `-im` `--image_size`
- Overrides computed image size based on GPU VRAM
  - image_size: Image size as WIDTHxHEIGHT (e.g., `128x128`)
```sh
python3 main.py -im 128x128
```

---
