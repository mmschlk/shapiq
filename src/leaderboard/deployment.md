# Deployment Guide

We explain how to deploy the Shapiq Living Benchmark Leaderboard both locally and on HuggingFace Spaces. The leaderboard is built using Gradio, which allows for easy deployment.

To deploy, one needs to:
1. **configure** the storage for the leaderboard data
2. **choose** a deployment method (local or HuggingFace Spaces)


## Storage Configuration

Available storage options for the leaderboard data include:
- **Local Storage**: Store the leaderboard data in a local JSONL file directory on your machine. This is suitable for personal use or testing.
- **MongoDB Storage**: Use a MongoDB database to store the leaderboard data. This is suitable for production use and is especially useful for collaborative environments.
- **HuggingFace Datasets**: Use the HuggingFace Datasets library to store and manage the leaderboard data. This is suitable for production use and is especially useful when a stable and cleaned version of the leaderboard data is required.


In order to configure the storage, please set the `LOADING_METHOD` global variable to the preferred storage option in the [leaderboard ui implementation](./ui/ui.py#L55). 


Each storage option requires specific configuration steps, which are detailed in the [Storage Module README](./storage/README.md). Parameter reminder (to be configured in the `.env` file):
- MongoDB: `MONGODB_URI` and `MONGODB_DB`
- HuggingFace Datasets: `HF_DATASET`, `HF_FILE`, and `HF_TOKEN`
- Local file storage: `LOCAL_DB_PATH`


For creating a MongoDB account and database, please refer to the [MongoDB Atlas documentation](https://www.mongodb.com/docs/get-started/?language=python). For creating a HuggingFace account and dataset, please refer to the [HuggingFace documentation](https://huggingface.co/docs/hub/datasets).


### Developer recommendations

We recommend:
- For local development and testing, use **Local Storage**.
- For a collaborative environment, where populating the leaderboard is required, use **MongoDB Storage**.
- For stable, public versions of the leaderboard, use **HuggingFace Datasets**. Manually version the leaderboard data and store it in as a HuggingFace Dataset.


## Deployment Methods

Please choose the deployment method that best suits your needs. The leaderboard can be deployed either locally or on HuggingFace Spaces.

### Local Deployment

For local deployment (deployment through `Gradio`), running the leaderboard ui requires executing the following command in the terminal:

From `src` directory
```bash
python3 -m leaderboard.ui.ui
```

OR 

From project root:
```bash
python3 app.py
```

For local deployment, with public access as provided by `Gradio`, please first set in the Pyhton [leaderboard ui implementation](./ui/ui.py#L1684) the `share` parameter to `True` in the `demo.launch()` function, as shown below:

```python
demo.launch(ssr_mode=False, share=True) 
```

### HuggingFace Spaces Deployment

Deployment to HuggingFace Spaces has been configured as a `GitHub Action`, which automatically deploys the leaderboard to HuggingFace Spaces when a new commit is pushed to the `huggingface-spaces` branch of the repository. The deployment process is triggered by the `huggingface-spaces.yml` workflow file located in the `.github/workflows` directory.

Requirements for HuggingFace Spaces deployment:
- A HuggingFace account and a HuggingFace Space created for the leaderboard. Please refer to the [HuggingFace documentation](https://huggingface.co/docs/hub/spaces) for instructions on creating a HuggingFace Space.
- A README.md file in the root of the repository:

```md
---
title: Shapiq Leaderboard
emoji: 🏆
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "6.14.0"
python_version: "3.12"
app_file: app.py
pinned: false
---
```

- And [`app.py`](../../app.py) file in the root of the repository, which serves as the entry point for the leaderboard application.



The GitHub Action workflow file (`huggingface-spaces.yml`) is configured to:
- Check out the repository code.
- Copy the repository code to the HuggingFace Space - includes only code files (restriction placed on larger files by GitHub and HuggingFace).

