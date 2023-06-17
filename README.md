# 2023_Capstone
국민대학교 2023-1 [추천시스템:캡스톤디자인]에서 진행한 프로젝트  

## run
1. templates.py 에서 원하는 param 변경
2. 터미널 or run.ipynb(Jupyter Notebook)에서 ```!python main.py --template 'train_bert'``` 실행  
3. ml-1m 사용하려면 ```1``` 입력. ml-20m 사용하려면 ```20``` 입력.
4. train 완료 후, test 진행하고 싶으면 ```y``` 입력.  

*Data는 없어도 코드 실행될 때, 없으면 다운 받아짐*  

### option 선택
- **Embedding** 방법) 아래 두가지 중 선택
  1. ```args.model_embedding = 'origin_embedding'``` → *기존 embedding 방법*
  2. ```args.model_embedding = 'hyper_embedding'``` → *Hyperbolic Embedding 적용*
  
- **Data Augmentation** 방법) 아래 네가지 중 선택
  1. ```args.data_type = 'origin_dataset'``` → *기존 데이터 셋 생성하는 방법*
  2. ```args.data_type = 'noise_dataset'``` → *noise 추가하는 방법 (user가 사용하지 않은 item 중 추가)*
  3. ```args.data_type = 'similarity'``` → *유사한 item으로 대체하는 방법*
  4. ```args.data_type = 'redundancy'``` → *중복성 추가하는 방법 (user가 사용한 item 중 추가)*
 
- Data Augmentation을 위한 **N_Aug** & **P** 지정) ['noise_dataset', 'similarity', 'redundancy'] 중 선택시 지정
  1. ```args.N_Aug``` → *증강할* ***규모*** *선택 (기존 데이터의 몇 배 증강 시킬지 지정)*  
     ```5, 10, 15``` 등 int 입력
  2. ```args.P``` → *한 user의 item 중* ***몇 %*** *를 변경할 것인지 지정 (None인 경우, 1개 변경)*  
     ```None``` or ```0.1, 0.2``` 등 $0 \le P \le 1$ 사이의 float 입력

- **mode** 선택
  1. ```args.mode = 'train'``` → *train 시킬 경우, 학습 완료 후 test 진행하고 싶은 경우 test 진행 물어보면 ```y``` 입력*
  2. ```args.mode = 'test'``` → *test 시킬 경우,*
     ```args.test_model_path = 'test path'``` → *test할 모델있는 경로 입력 ex) './experiments/Baseline/'*


### Check List
- [X] test mode 추가  
  templates.py에서 아래와 같이 변경후, ```!python main.py --template 'train_bert'``` 실행  
  1. ```args.mode = 'test'```  
  2. ```args.test_model_path = 'experiments/[생성한 폴더명]/'``` 으로 입력
  
- [ ] 각자 맡은 Data Augmentation 추가
  - [ ] noise 추가
  - [ ] 유사한 item으로 대체
  - [ ] 중복성 추가
  
- [X] Hyperbolic Embedding 추가, args에서 선택할 수 있도록 추가


---

## 참고 논문 및 코드
### main BERT4Rec code 참고
- [paper] https://arxiv.org/abs/1904.06690
- [code] https://github.com/jaywonchung/BERT4Rec-VAE-Pytorch.git  
  해당 코드를 BERT4Rec의 baseline으로 사용하여 실험 진행

### Data Augmentation 참고 논문
- [paper] https://arxiv.org/abs/2203.14037

### Hyperbolic Embedding
- [paper] https://arxiv.org/abs/2104.03869
- [code] https://github.com/leymir/hyperbolic-image-embeddings
