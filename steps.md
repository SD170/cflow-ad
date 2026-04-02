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
mamba create -n cflow-scratch-v1 -c conda-forge python=3.8
```

activate 
```
source activate cflow-scratch-v1
```

installed kaggle
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



