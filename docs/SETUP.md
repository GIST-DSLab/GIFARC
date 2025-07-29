## Requirements & Setup

First, clone this git.

```clone
git clone <GIT_GIFARC_URL>
```

To install requirements, move to the cloned directory and run `pip install`:

```setup
pip install -r requirements.txt
```

Next, install docker and docker compose

```bash
cd ./GIFARC
docker compose up -d
```

[note] it use port 8998 as jupyter note book, if you want to change you can chage
Next turn on the vscode and install devcontainer extention.


Ctrl + p and find
>Dev containers: Open Folder in Container
Click it and open GIFARC folder

Open the container and type

```bash
touch .env # make .env file
# and write below we make it to use wi open ai if you want to work with other api or some other provider or local you can fix under utilitys/llm.py
OPENAI_API_KEY="****"
```

after that put gif in to ./data/GIF
Open, description_executor.ipynb and now run all the ipynb file, it will generate the gif arc under ./results/promblem~~

