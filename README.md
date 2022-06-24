# KO_CNN_POStagger

한국어 CNN POS-tagger입니다. 아래의 레퍼런스의 구조를 그대로 구현해봤습니다.

데이터는 세종 말뭉치의 형태분석 말뭉치를 사용했고 총 문장수는 83만문장입니다.

=> 전체 83만 문장 중 training 70만, test 13만 문장을 사용했을 때 epoch 1이 채 돌기도 전에 classification 결과 모든 POS class에 비슷한 값이 예측되는 것으로 봐서 overfitting의 문제가 생기는 것으로 추측되었습니다.

=> training 5만, test 1만 문장을 사용했을 때 accuracy가 94~96%까지 상승하는 것으로 확인했습니다.

※ 세종 말뭉치의 형태분석 말뭉치는 형태소 분석이 된 데이터이기 때문에 현 모델에는 형태소 분석기는 포함되어 있지 않습니다. 

### 소스 코드 설명

- Processing_Data.py

data_v5_edit.txt를 이용하여 data, label, indexed_data, dictionary(data,label) pickle 파일을 생성합니다.
레퍼런스를 보시면 크게 character 단위와 word 단위의 CNN 두개를 확인할 수 있는데 그 중 character 단위의 cnn을 구성하기 위해서 한글 데이터를 모두 자소단위로 분리하는 것을 확인하실 수 있습니다.

- CNN_conv3d.py

파일의 이름이 conv3d 인 이유는 character 단위의 CNN에서 input이 5차원으로 이루어져 있기 때문에 명명하였습니다.
모델 구조는 레퍼런스를 따라서 나름대로 구성해보았습니다

- Run_Model.py

학습 된 모델로 training 데이터 외의 데이터를 가지고 실험해 볼 수 있도록 만들었습니다.



### 레퍼런스
A General-Purpose Tagger with Convolutional Neural Networks(Xiang Yu and Agnieszka Falenska and Ngoc Thang Vu)
http://www.aclweb.org/anthology/W17-4118
