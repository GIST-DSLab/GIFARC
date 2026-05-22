# Recommand Env

- `>= Ubuntu 22.04`

# Requirements & Setup

### 1. Clone git

First, clone this git.

```clone
git clone <GIT_GIFARC_URL>
```

### 2. Setup requirements.

Move to the cloned directory and run `pip install` to install pip requirements :

```setup
pip install -r requirements.txt
```

### 3. Create docker container.

Next, install `docker` and run `docker compose`.

```bash
cd ./GIFARC
docker compose up -d
```

Note. it uses port 8997 for jupyter notebook.

### 4. Open docker container workspace.

Next, turn on the vscode and install extension `Dev Container`.


Type `Ctrl + <Shift> + p` to open workspace command and find

```
> Dev containers: Open Folder in Container
```

Click it, and select GIFARC folder. It will open a workspace in the docker container we created.

### 5. Setup environment variables.

Open the container terminal and type:

```bash
cp .env.example .env
# Fill in the keys for the provider you plan to use.
OPENAI_API_KEY="your open ai api key"
AZURE_OPENAI_ENDPOINT="your azure openai endpoint"
AZURE_OPENAI_API_KEY="your azure openai api key"
DEPLOYMENT_IMAGE_PROCESSOR_NAME="your image processor deployment"
DEPLOYMENT_DATA_GENERATOR_NAME="your data generator deployment"

```

### 6. All Set!

You are all set up. Put GIF files in to ./data/GIF and run the codes.
