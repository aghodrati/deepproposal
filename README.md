# deepproposal
code for DeepProposal paper presented in ICCV 2015

#Installing
- Dependencies should be installed according to their instructions. They should be in ./deps/ folder:
  - matconvnet-1.0-beta8 : http://www.vlfeat.org/matconvnet/download/matconvnet-1.0-beta8.tar.gz (compile it in gpu-enabled mode)
  - piotr_toolbox_V3.40 : http://vision.ucsd.edu/~pdollar/toolbox/piotr_toolbox_V3.40.zip
  - liblinear-1.94 : https://github.com/cjlin1/liblinear/archive/v194.tar.gz (if you want to train an objectness)
  - modified version of EdgeBox which I have included it.

- Instruction to compile the modified code of edgebox(I have included files for 64-bit linux):
  - mex private/edgesDetectMex.cpp -outdir private [OMPPARAMS]
  - mex private/edgesNmsMex.cpp -outdir private [OMPPARAMS]
  - mex private/spDetectMex.cpp -outdir private [OMPPARAMS]
  - mex private/edgeBoxesMex.cpp -outdir private

  - Here [OMPPARAMS] are parameters for OpenMP and are OS and compiler dependent.
    - Windows: [OMPPARAMS] = '-DUSEOMP' 'OPTIMFLAGS="$OPTIMFLAGS' '/openmp"'
    - Linux V1: [OMPPARAMS] = '-DUSEOMP' CFLAGS="\$CFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp"
    - Linux V2: [OMPPARAMS] = '-DUSEOMP' CXXFLAGS="\$CXXFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp" \n

  - To compile without OpenMP simply omit [OMPPARAMS]; note that code will be single threaded in this case.

- I have added the compiled files for nms_c.cpp and intg_sub.cpp under 64-bit linux. If you want to compile it, use mex file_name.cpp

- The models that are used for DeepProposal (in './models' folder):

- objectness models: as described in section 3.1 of the paper (trained models are included)
  - contour model: as described in section 4 of the paper (trained model is included)
  - pre-trained CNN model (http://www.vlfeat.org/matconvnet/models/imagenet-caffe-ref.mat): make a symbolic link in ./models/cnn/ that point to the imagenet-caffe-ref.mat or change the path in get_opts.m


#Datasets
I have included a script called "main_run.m" to extract proposals for VOC 2007 and COCO 2014. you should setup VOC07 development kit and/or COCO14 api+dataset according to their instructions and then set paths in get_opts.m

#Notes
- All boxes are in format of [x y x+w y+h]
- This version does not use spatial pyramid representation for second stage.
- For COCO14 dataset, I consider objects with area smaller than 32^2 as difficult so they are omitted during evaluation. you can change it in arrange_imdb.m
- Features can be loaded from './feats/' in case they are already extracted
- Outputs are stored in './outputs/'.
- Logs are stored in './logs/'.
