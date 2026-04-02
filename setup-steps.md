ssh:
```
ssh -X sdutta60@sol.asu.edu
```
asking for a interactive session:
```
salloc -p general -q class -c 4 --mem=40G -t 0-1  -A class_cse579spring2026
```

cd to scratchpad directory (personal dirs have 100 gb limit, scratchpad has 1tb), change sdutta60 to your asu username

```
cd /scratch/sdutta60
```

git clone https://github.com/SD170/cflow-ad (i've forked it)

```
https://github.com/SD170/cflow-ad
```

initiate python env (sol uses mamba package manager)

```
module load mamba/latest
```

creating a new python env (python 3.8 cuz the readme says that)
```
mamba create -n kaggle-setup -c conda-forge python=3.8
```

activate 
```
source activate kaggle-setup
```

install kaggle
```
python -m pip install -U kaggle
```

creating a new folder inside /cflow-ad called data
```
mkdir -p data
cd data
```

download dataset 
```
export KAGGLE_USERNAME="your_kaggle_username"
export KAGGLE_KEY="your_api_key"
kaggle datasets download -d thtuan/btad-beantech-anomaly-detection -p data --unzip
```

renaming the folder
```
mv ./data ./BTAD
```
now we're in /scratch/sdutta60/cflow-ad/data/BTAD


now exit from the node, and ask for a gpu node, so that we can test stuff out
```
exit
```

asking for 1 a100 for 1 hr
```
salloc -p general -q class -c 2 --mem=8G -t 0-1 -G a100:1 -A class_cse579spring2026
```

theres an exiting base env we can use for cuda
```
module load mamba/latest
source activate pytorch-gpu-2.3.1-cuda-12.1
```

install deps (after cd to the project folder):
```
python -m pip install -U pip
python -m pip install -r requirements.txt
```

install FrEIA (some deps, no idea what it does):
```
python -m pip install "git+https://github.com/vislearn/FrEIA.git@cc5cf5ebee08f9bb762bab5a6535c11d19ccb026"
```



to run:
```
python main.py --gpu 0 --pro -inp 512 --dataset btad --class-name 01
```
