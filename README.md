# Be Careful : 도로 보행 위험 행동 인식 시스템

- 카메라를 통해 보행자를 인식하고 보행자의 사진을 수집한다.
- 수집된 보행자 사진을 기반으로 보행자의 행동을 인식하여 위험 행동 (차도 위를 보행하는 행위 또는 스마트폰을 보며 보행하는 행위) 여부를 판단한다.
- 보행자가 위험 행위를 하는 것으로 판단되면 보행자에게 경고 알림을 줌으로써 보행 중 교통사고를 예방한다.

## 1. 프로젝트 개요
---
한국은 현재 교통 패러다임이 차량 중심에서 보행 중심의 사회로 변화하고 있다. 따라서 보행자의 안전 및 환경 계선의 관심 또한 증가하였다. 하지만 민식이법 개정 이후에도 교통사고 건 수와 부상자 수 변함이 없었다.

<img src="https://github.com/qor6/SoftWare/assets/88486391/bab2f84c-7f51-45e0-b24e-faef1cbc7b43" alt="교통사고"  width="70%" height="60%" ></img>

도로 보행 중 위험 행동을 하는 보행자가 감지되었을 때 이들에게 경고하는 알림을 준다면 보다 효율적으로 안전한 보행 환경을 조성할 수 있을 것이다.


## 2. 개발 모듈
---
(1) YOLOv5
  - 개발 과정
 > 1.  AI HUB에서 제공한 900장의 이미지
 > 2. YOLOv5s Model로 이미지 학습
 > 3. 차도 위로 보행하는 행위를 인식
 
  - 개발 결과
  > Precision : 97.1% / Recall : 95.4% / mAP50: 98.8%
  >
  ><img src="https://github.com/qor6/SoftWare/assets/88486391/314b1f4e-f7b3-4a05-bff9-d02b28b1c8bb" alt="YOLO 결과"></img>
  > 결과 이미지
  > 
  > <img src="https://github.com/qor6/SoftWare/assets/88486391/641d8b8c-66f5-4d02-9ae4-0ddf6c4ac47a" alt="YOLO"></img>

 
(2) Openpose
  - 개발 과정
  > 1. 스마트폰 사용 보행자 사진 데이터 수집
  > 2. Coco Model로 사람 관절 인식
  > 3. 사람의 어깨부터 팔꿈치, 팔꿈치부터 손까지의 각도 인식
  > 4. 각도가 수직이거나 예각이면 스마트폰을 하고 있다고 인식


  - 개발 결과
  > <img src="https://github.com/qor6/SoftWare/assets/88486391/a5d7dca6-63ea-4501-af59-bfb005043bb2"  width="40%" height="30%" alt="Openpose"></img>




