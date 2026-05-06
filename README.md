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
