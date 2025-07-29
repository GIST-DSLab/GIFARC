# Recommand Env

- Ubuntu 22.04 (If you are using docker then it will be fine)
- GPU( 3060RTX ~ 4080SUPER ): it will not use GPU however because of some library use gpu, it might be a issue to install cuda.

# Requirements & Setup

Initially you needs install docker and docker compose.
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


Ctrl + <Shift> + p and find
>Dev containers: Open Folder in Container

Click it and open GIFARC folder

Open the container and type

```bash

touch .env # make .env file
# and write below we make it to use wi open ai if you want to work with other api or some other provider or local you can fix under utilitys/llm.py
OPENAI_API_KEY="your open ai api key"

```

after that put gif in to ./data/GIF then you can run the code.
READ 

GENERATION.md


