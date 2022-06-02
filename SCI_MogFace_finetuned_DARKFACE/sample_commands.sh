bash
conda activate PyEnv
cd /code
pip install -r requirements_MogFace.txt # -i https://pypi.tuna.tsinghua.edu.cn/simple
cd utils/nms && python setup.py build_ext --inplace && cd ../..
cd utils/bbox && python setup.py build_ext --inplace && cd ../..
python -W ignore::UserWarning ./train.py -b 4 -n 1 -r 0 -s ./SCI/weights/difficult.pt
# python -W ignore::UserWarning ./validate_from_train.py -b 4 -r 5 (需要先val2test?)
python -W ignore::UserWarning ./test_multi.py -n 5
python -W ignore::UserWarning ./test_multi.py -n 5 -f True -d ./final_results
python -W ignore::UserWarning ./test_multi.py -n 5 -f True -d ./final_results --test_hard 1
python -W ignore::UserWarning ./test_multi_noeval.py -n 0 -s ./SCI/weights/difficult.pt -f True -d ./final_results --test_hard 1



cd /
mv /code /code-old
cp -r /mnt/test10 /code
cp -r /code /code1
cd /code
pip install -r requirements_MogFace.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
cd ./dataset/WIDERFACE
chmod 777 ./test2val9.sh
./test2val9.sh
cd ..
cd ..
mkdir ./thisis9
python -W ignore::UserWarning ./test_multi.py -n 85 -f True -d ./final_results --test_hard 1

cd /code1
cd ./dataset/WIDERFACE
chmod 777 ./test2val8.sh
./test2val8.sh
cd ..
cd ..
mkdir ./thisis8
python -W ignore::UserWarning ./test_multi.py -n 85 -f True -d ./final_results --test_hard 1