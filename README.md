# matejT_AMS_izziv_2026

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

