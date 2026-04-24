# Uporabimo uradno Pytorch sliko
FROM pytorch/pytorch:2.0.1-cuda11.8-cudnn8-runtime
#FROM pytorch/pytorch:1.7.1-cuda11.-cudnn8-devel

# Install
RUN pip install numpy
RUN pip install nibabel
RUN pip install timm==0.5.4
RUN pip install scipy
RUN pip install torch
#RUN pip install SimpleITK

#COPY source source

WORKDIR /workdir


#CMD ["python", "Code/Test_C2FViT_pairwise.py", "--modelpath", "Model/C2FViT_affine_COM_pairwise_stagelvl3_118000.pth", "--fixed", "Data/image_B.nii.gz", "--moving", "Data/image_A.nii.gz"]
