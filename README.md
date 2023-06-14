# 2023_Capstone
국민대학교 2023-1 추천시스템:캡스톤디자인  



## run
1. templates.py 에서 원하는 param 변경
2. 터미널 or run.ipynb(Jupyter Notebook)에서 ```!python main.py --template 'train_bert'``` 실행  
3. ml-1m 사용하려면 ```1``` 입력. ml-20m 사용하려면 ```20``` 입력.
4. train 완료 후, test 진행하고 싶으면 ```y``` 입력.  

*Data는 없어도 코드 실행될 때, 없으면 다운 받아짐*  

### test mode 추가
- templates.py에서 아래와 같이 변경후, ```!python main.py --template 'train_bert'``` 실행  
  1. args.mode = 'test'
  2. args.test_model_path = 'experiments/[생성한 폴더명]/' 으로 입력

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
