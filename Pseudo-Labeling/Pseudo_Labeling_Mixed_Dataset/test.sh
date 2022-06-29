mkdir results
mkdir results/ep14
mkdir results/ep11
mkdir results/ep9
python -W ignore::UserWarning ./darkface_val_zdce.py --trained_model ./trained/epoch14_dsfd.pth --save_folder ep14
python -W ignore::UserWarning ./darkface_val_zdce.py --trained_model ./trained/epoch11_dsfd.pth --save_folder ep11
python -W ignore::UserWarning ./darkface_val_zdce.py --trained_model ./trained/epoch9_dsfd.pth --save_folder ep9