mv ./WIDER_val ./WIDER_val_real
ln -s ./WIDER_test ./WIDER_val

mv ./val.txt ./val_real.txt
mv ./val_test6.txt ./val.txt

mv ./wider_face_val.mat ./wider_face_val_real.mat
mv ./wider_face_val_test6.mat ./wider_face_val.mat

mv ./wider_face_split/wider_face_val.mat ./wider_face_split/wider_face_val_real.mat
mv ./wider_face_split/wider_face_val_test6.mat ./wider_face_split/wider_face_val.mat

mv ./ground_truth/wider_face_val.mat ./ground_truth/wider_face_val_real.mat
mv ./ground_truth/wider_face_val_test6.mat ./ground_truth/wider_face_val.mat
