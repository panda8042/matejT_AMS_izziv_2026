# AMS izziv 2026 - U-Mamba za segmentacijo koronarnih arterij

Avtor: Matej Turk  
Metoda: U-Mamba  
Dataset: ImageCAS  
Naloga: avtomatska segmentacija koronarnih arterij na 3D CTA slikah

## Opis

Projekt pripravlja metodo U-Mamba za segmentacijo koronarnih arterij na podatkovni zbirki ImageCAS.

Repozitorij vsebuje:

- Docker okolje s CUDA, PyTorch in U-Mamba odvisnostmi,
- skripte za pretvorbo ImageCAS podatkov v nnU-Net format,
- osnovne CLI skripte za trening, testiranje in inferenco,
- mini dataset workflow za varno preverjanje delovanja pipeline-a na 10 slikah.

Končni trening celotnega modela je predviden na zmogljivejšem računalniku z dovolj prostora in GPU pomnilnika. Na laboratorijskem računalniku se uporablja mini testni pipeline.

## Struktura projekta

```text
matejT_AMS_izziv_2026/
├── Dockerfile
├── README.md
├── run_train.py
├── run_test.py
├── run_inference.py
├── configs/
│   └── params.json
├── scripts/
│   ├── convert_imagecas_to_nnunet.py
│   └── convert_imagecas_mini_to_nnunet.py
├── src/
├── outputs/
├── nnUNet_raw/
├── nnUNet_preprocessed/
└── nnUNet_results/
```



## Osebne beležke

git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/panda8042/matejT_AMS_izziv_2026.git
git push -u origin main


## subsection docker

docker build -t matejt_ams_izziv .

docker ps -> preveri aktivne
docker image ls

docker run --gpus device=0 -it --rm -v "$PWD":/workdir -v /media/FastDataMama/izziv/data:/data matejt_ams_izziv python3 test.py


## Smoke test status

Na laboratorijskem računalniku je bil uspešno izveden mini end-to-end test za U-Mamba pipeline:

- Dataset503_ImageCASMini50: 50 train slik + 10 test slik,
- nnU-Net 3d_fullres preprocessing uspešno izveden,
- U-Mamba trainer se uspešno zažene,
- zaradi omejitve VRAM na RTX 2080 Ti je bil za smoke test uporabljen zmanjšan plan:
  - batch_size = 1,
  - patch_size = [48, 96, 96],
- 1 epoch trening se uspešno zaključi,
- ustvarjena sta checkpoint_final.pth in checkpoint_best.pth,
- validation prediction .nii.gz datoteke se ustvarijo.

Opomba: po 1 epochu so validation predikcije še prazne oziroma vsebujejo samo background, zato ta test potrjuje delovanje pipeline-a, ne pa kakovosti končnega modela. Za uporaben model je potreben daljši trening na večjem GPU oziroma z ustrezno izbranim planom.

## Smoke test status

Na laboratorijskem računalniku je bil uspešno izveden mini end-to-end test za U-Mamba pipeline.

Uporabljen je bil mini dataset:

- Dataset503_ImageCASMini50,
- 50 učnih primerov,
- 10 testnih primerov,
- uradni Split-1 iz ImageCAS split datoteke.

Uspešno izvedeni koraki:

- pretvorba ImageCAS podatkov v nnU-Net format,
- nnU-Net 3d_fullres preprocessing,
- zagon U-Mamba trainerja,
- 1 epoch trening na GPU,
- zapis checkpoint_final.pth in checkpoint_best.pth,
- generiranje validation prediction .nii.gz datotek,
- pregled CTA slike in ground-truth maske v 3D Slicerju.

Na RTX 2080 Ti originalni 3d_fullres plan ni šel skozi zaradi omejitve VRAM. Za smoke test je bil zato uporabljen zmanjšan plan:

- batch_size = 1,
- patch_size = [48, 96, 96].

Opomba: po 1 epochu so validation predikcije še prazne oziroma vsebujejo samo background. Ta test zato potrjuje funkcionalnost pipeline-a, ne pa kakovosti končnega modela. Za uporaben model je potreben daljši trening na večjem GPU oziroma z ustrezno izbranim planom.

## Test 
Dataset504_ImageCASPreprocessed150:
- source: existing preprocessed Dataset001_ImageCAS subset
- split: 140 train / 10 validation, validation IDs 141–150
- model: U-Mamba Enc 3D
- configuration: 3d_fullres
- patch size: [64, 128, 128]
- batch size: 1
- epochs: 20
- result: Mean Validation Dice = 0.4506
- validation predictions: 10/10 generated, all non-empty

## Dataset504 U-Mamba 20 epoch result

Na laboratorijskem računalniku je bil izveden U-Mamba trening na lokalnem Dataset504 subsetu, ki uporablja obstoječe predprocesirane ImageCAS podatke.

Uporabljena konfiguracija:

- dataset: Dataset504_ImageCASPreprocessed150
- split: 140 training / 10 validation
- validation primeri: 141–150
- model: U-Mamba Enc 3D
- konfiguracija: 3d_fullres
- patch size: [64, 128, 128]
- batch size: 1
- število epochov: 20

Rezultat:

- validation complete
- Mean Validation Dice: 0.4506
- ustvarjenih 10/10 validacijskih predikcij
- vse validacijske predikcije so neprazne
- velikost rezultatov: približno 1.3 GB

Ta rezultat kaže, da U-Mamba pipeline ne deluje več samo kot smoke test, ampak se model dejansko uči na ImageCAS podatkih.


## Dataset505 U-Mamba 200 epoch result

Izveden je bil daljši U-Mamba trening na večjem ImageCAS subsetu.

Uporabljena konfiguracija:

- dataset: Dataset505_ImageCASPreprocessed700
- split: 650 training / 50 validation
- validation primeri: 651–700
- model: U-Mamba Enc 3D
- konfiguracija: 3d_fullres
- patch size: [64, 128, 128]
- batch size: 1
- število epochov: 200

Rezultat:

- validation complete
- Mean Validation Dice: 0.6412
- ustvarjenih 50/50 validacijskih predikcij

V primerjavi s prejšnjim 20-epoch baseline testom na Dataset504 se je Mean Validation Dice izboljšal iz 0.4506 na 0.6412.




## Dataset506 U-Mamba 800 epoch patch80 result

Izveden je bil dodaten U-Mamba trening na istem 650/50 splitu kot Dataset505, vendar z večjim patch size in daljšim učenjem.

Konfiguracija:

- dataset: Dataset506_ImageCASPreprocessed700Patch80
- split: 650 training / 50 validation
- validation primeri: 651–700
- model: U-Mamba Enc 3D
- konfiguracija: 3d_fullres
- patch size: [80, 128, 128]
- batch size: 1
- število epochov: 800

Rezultat:

- Mean Validation Dice: 0.6536

V primerjavi z Dataset505 U-Mamba treningom se je rezultat izboljšal iz 0.6412 na 0.6536. Večji patch in daljši trening sta torej prinesla manjše, vendar merljivo izboljšanje.

PRIMERJAVA
Dataset505 patch64, 200 ep:
Mean Dice: 0.6412
Std Dice:  0.0728
Min Dice:  0.4310
Max Dice:  0.7810

Dataset506 patch80, 800 ep:
Mean Dice: 0.6536
Std Dice:  0.0779
Min Dice:  0.3918
Max Dice:  0.7856


## Dataset506 nnU-Net baseline 800 epoch patch80 result

Izveden je bil še osnovni nnU-Net baseline na istem splitu kot Dataset506 U-Mamba eksperiment.

Uporabljena konfiguracija:

- dataset: Dataset506_ImageCASPreprocessed700Patch80
- split: 650 training / 50 validation
- validation primeri: 651–700
- model: nnU-Net baseline
- konfiguracija: 3d_fullres
- patch size: [80, 128, 128]
- batch size: 1
- število epochov: 800

Rezultat:

- Mean Validation Dice: 0.7404

Primerjava glavnih rezultatov:

| Eksperiment | Model | Patch size | Epochs | Train/val | Mean Dice |
|---|---|---:|---:|---:|---:|
| Dataset505 | U-Mamba Enc 3D | [64, 128, 128] | 200 | 650/50 | 0.6412 |
| Dataset506 | U-Mamba Enc 3D | [80, 128, 128] | 800 | 650/50 | 0.6536 |
| Dataset506 | nnU-Net baseline | [80, 128, 128] | 800 | 650/50 | 0.7404 |

Na istem Dataset506 splitu in z enako patch/epoch konfiguracijo je osnovni nnU-Net dosegel višji Mean Validation Dice kot U-Mamba Enc. To kaže, da v trenutni konfiguraciji U-Mamba ni izboljšala segmentacije glede na nnU-Net baseline.

## Dataset506 nnU-Net baseline 800 epoch patch80 result

Izveden je bil osnovni nnU-Net baseline na istem splitu kot Dataset506 U-Mamba eksperiment.

Uporabljena konfiguracija:

- dataset: Dataset506_ImageCASPreprocessed700Patch80
- split: 650 training / 50 validation
- validation primeri: 651–700
- model: nnU-Net baseline
- konfiguracija: 3d_fullres
- patch size: [80, 128, 128]
- batch size: 1
- število epochov: 800

Rezultat na 50 validacijskih primerih:

- Mean Dice: 0.7402
- Std Dice: 0.0646
- Min Dice: 0.5789
- Max Dice: 0.8478
- N: 50

Primerjava glavnih eksperimentov:

| Eksperiment | Model | Patch size | Epochs | Train/val | Mean Dice | Std Dice | Min Dice | Max Dice |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Dataset505 | U-Mamba Enc 3D | [64, 128, 128] | 200 | 650/50 | 0.6412 | 0.0728 | 0.4310 | 0.7810 |
| Dataset506 | U-Mamba Enc 3D | [80, 128, 128] | 800 | 650/50 | 0.6536 | 0.0779 | 0.3918 | 0.7856 |
| Dataset506 | nnU-Net baseline | [80, 128, 128] | 800 | 650/50 | 0.7402 | 0.0646 | 0.5789 | 0.8478 |

Na istem Dataset506 splitu, z enakim patch size, batch size in številom epochov, je osnovni nnU-Net dosegel višji Mean Validation Dice kot U-Mamba Enc. nnU-Net baseline je bil tudi stabilnejši, saj je imel nižji standardni odklon in višji najslabši Dice primer. V trenutni konfiguraciji U-Mamba torej ni izboljšala segmentacije glede na nnU-Net baseline.

Analiza FP/FN kaže, da oba modela še vedno oversegmentirata, vendar nnU-Net praviloma ustvari manj false-positive voxlov in doseže boljše ujemanje z ročnimi maskami.
