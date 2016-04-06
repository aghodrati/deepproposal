# News
- generated proposals of our method are added.
- spatial pyramid pooling models are added

# DeepProposal
code for DeepProposal paper presented in ICCV 2015 (http://arxiv.org/abs/1510.04445):

Amir Ghodrati, Ali Diba, Marco Pedersoli, Tinne Tuytelaars, Luc Van Gool, "DeepProposal: Hunting Objects by Cascading Deep Convolutional Layers".

# Installing
- Dependencies should be installed according to their instructions. They should either be in ./deps/ folder or you need to change the paths in get_opts.m file:
  - matconvnet : https://github.com/vlfeat/matconvnet (compile it in gpu-enabled mode)
  - piotr_toolbox_V3.40 : http://vision.ucsd.edu/~pdollar/toolbox/piotr_toolbox_V3.40.zip
  - liblinear svm : https://github.com/cjlin1/liblinear (if you want to train an objectness)
  - modified version of EdgeBox which I have included it.

- Instruction to compile the modified code of edgebox(I have included files for 64-bit linux):
  - mex private/edgesDetectMex.cpp -outdir private [OMPPARAMS]
  - mex private/edgesNmsMex.cpp -outdir private [OMPPARAMS]
  - mex private/spDetectMex.cpp -outdir private [OMPPARAMS]
  - mex private/edgeBoxesMex.cpp -outdir private
  - Here [OMPPARAMS] are parameters for OpenMP and are OS and compiler dependent.
    - Windows: [OMPPARAMS] = '-DUSEOMP' 'OPTIMFLAGS="$OPTIMFLAGS' '/openmp"'
    - Linux V1: [OMPPARAMS] = '-DUSEOMP' CFLAGS="\$CFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp"
    - Linux V2: [OMPPARAMS] = '-DUSEOMP' CXXFLAGS="\$CXXFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp"
  - To compile without OpenMP simply omit [OMPPARAMS]; note that code will be single threaded in this case.

- The compiled files for "nms_c.cpp" and "intg_sub.cpp" under 64-bit linux are added. If you want to compile it, use "mex file_name.cpp"

- The models that are used in DeepProposal (in './models' folder):
  - objectness models: as described in section 3.1 of the paper (trained models are included)
  - contour model: as described in section 4 of the paper (trained model is included)
  - pre-trained CNN model (http://www.vlfeat.org/matconvnet/models/imagenet-caffe-ref.mat): make a symbolic link in ./models/cnn/ that point to the imagenet-caffe-ref.mat or change the path in get_opts.m

# Demo
You can run the "demo.m" for extracting proposals for a given image.

# Datasets
A script called "main_run.m" is included to extract proposals for VOC 2007 and COCO 2014. you should setup VOC07 development kit and/or COCO14 api+dataset according to their instructions and then set paths in get_opts.m

# Notes
- All boxes are in format of [x y x+w y+h]
- This version does not use spatial pyramid representation for second stage. -UPDATE: It is included now!
- For COCO14 dataset, objects with area smaller than 32^2 are considered as difficult so they are omitted during evaluation. you can change it in arrange_imdb.m
- Features can be loaded from './feats/' in case they are already extracted
- Outputs are stored in './outputs/'.
- Logs are stored in './logs/'.

# UPDATE: generated proposals for VOC2007 test set:
- DeepProposal-50: <a href=https://drive.google.com/file/d/0B8gk4ucVr8z_RHRic3dqeGdCLTQ/view?usp=sharing> Google Drive</a>
- DeepProposal-70: <a href=https://drive.google.com/file/d/0B8gk4ucVr8z_TGFKS1ZHM01weDQ/view?usp=sharing> Google Drive</a>
- note: the proposals are in the format of [x, y, width, height, confidence]
