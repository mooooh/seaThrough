# seaThrough — Unity RGB 영상으로 수면 높이 예측하기

현재 데이터에 맞춰 **RGB 1장 → 같은 시점·같은 픽셀 위치의 수면 높이 맵**을 학습하는 PyTorch 프로젝트입니다.
사용자 확인에 따라 `Depth/*.png`의 **G 채널은 수면 높이이며, 밝을수록 높습니다.**
따라서 이 모델의 출력은 카메라와 수면 사이의 거리(depth)가 아닌 수면 높이입니다.

물리 높이로의 변환식과 프레임 사이에 높이 범위가 일정한지는 확인되지 않았습니다.
학습 목표와 기본 출력은 `height_normalized = G / 255`입니다. 이미지별 min/max 정규화나 값 반전은 하지 않습니다.
검은색(0)도 유효한 높이로 학습합니다. 실제 미터 단위 높이나 파고를 계산하려면 Unity 저장 코드의 높이 매핑이 필요합니다.

## 확인한 데이터

| 항목 | 결과 |
|---|---|
| 페어 | 5,000쌍, ID 000001~005000 |
| 해상도 | 전부 736×617 |
| PNG 검사 | 누락 페어·PNG 무결성 오류 없음 |
| 타깃 | 8-bit RGB 이미지의 G 채널, 0~255 |
| G=0 / G=255 픽셀 비율 | 약 2.92% / 1.94% |
| 학습 | ID 1~3500, 3,500쌍 |
| 검증 | ID 3551~4250, 700쌍 |
| 테스트 | ID 4301~5000, 700쌍 |
| 제외 | 3501~3550, 4251~4300, 총 100쌍 |

파일명 번호가 시간 순서라고 가정해 구간별로 분리했습니다. 50프레임 간격은 임시 설정입니다.
인접 프레임을 무작위로 나누면 비슷한 장면이 학습·검증에 함께 들어갈 수 있습니다.
현재 분할은 독립된 시뮬레이션 실행을 보장하지 않습니다. **새 Unity 영상에 대한 성능은 다른 seed/촬영 세션 전체를 따로 보관해 평가해야 합니다.**
파고, 파장, 풍향, 조명, 카메라 위치·FOV가 달라지는 입력까지 사용할 계획이면 그 조건을 학습 데이터에도 포함하세요.

상세 통계는 `artifacts/dataset_report.json`, RGB·저장 타깃·G 채널 비교는 `artifacts/dataset_preview.jpg`에 있습니다.

## 모델과 학습 설정

- ImageNet 사전학습 ResNet-18 인코더 + skip connection이 있는 U-Net 형태의 디코더
- 출력: 원 채널 높이에 대응하는 1채널 sigmoid 값 `[0,1]`
- 손실: 높이 L1 + 0.5 × 여러 해상도에서 계산한 공간 기울기 L1
- RGB 밝기·대비·채도 변경, 약한 블러. 기하 변형은 기본적으로 사용하지 않음
- 검증 MAE로 `best.pt` 선택, early stopping. 테스트 데이터는 학습 중 사용하지 않음
- 학습 타깃의 픽셀별 평균 맵과 비교하여 고정 카메라의 평균 수면만 외우는 해와 비교

RTX 3070 시작 설정은 `configs/rtx3070.json`에 있습니다: **384×320(W×H), batch 8, AMP, 최대 60 epochs**.
CUDA 메모리가 부족하면 `--batch-size 4` 또는 `2`로 줄이세요. 전체 학습의 정확도와 RTX 3070 처리 시간은 아직 측정하지 않았습니다.
첫 학습 시 torchvision이 ResNet-18 사전학습 가중치를 다운로드합니다.

## RTX 3070 PC에서 시작하기

GitHub에서 코드를 가져올 수 있습니다. 원본 데이터는 별도로 복사해야 합니다.

```text
git clone https://github.com/mooooh/seaThrough.git
cd seaThrough
```

프로젝트의 `wave_height/`, `configs/`, `tests/`, requirements 파일과 데이터 폴더를 함께 복사하세요.
`artifacts/rtx3070_training_project.zip`에는 코드와 설명이 들어 있으며 원본 데이터와 가상환경은 포함하지 않습니다.
다음과 같이 놓으면 데이터 폴더 이름을 직접 입력하지 않아도 자동으로 찾습니다.

```text
Crest/
  wave_height/
  configs/
  requirements.txt
  crest_dataset 복사본/
    Original/Original000001.png ...
    Depth/Depth000001.png ...
```

Python 3.12를 권장합니다. 아래는 프로젝트 폴더에서 실행하는 Windows PowerShell 명령입니다.
GPU 드라이버에 맞는 CUDA 빌드는 [PyTorch 공식 설치 선택기](https://pytorch.org/get-started/locally/)에서 확인할 수 있습니다.
CUDA 12.8 빌드를 선택할 경우 설치 예시는 다음과 같습니다.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

GPU 확인 결과가 `True`, `NVIDIA GeForce RTX 3070`인지 확인한 다음 실행하세요.

```powershell
.venv\Scripts\python.exe -m wave_height.prepare
.venv\Scripts\python.exe -m wave_height.train --config configs/rtx3070.json
.venv\Scripts\python.exe -m wave_height.evaluate --checkpoint runs/rtx3070/best.pt --device cuda
```

Linux에서는 `python3 -m venv .venv`로 가상환경을 만들고 `.venv\Scripts\python.exe` 대신 `.venv/bin/python`을 사용하면 됩니다.
Linux에서도 CUDA 빌드 설치 명령은 공식 설치 선택기로 확인하세요.

`prepare`는 데이터를 복사하거나 수정하지 않습니다. 이미지 검사와 페어 목록·분할 파일 생성만 수행합니다.
데이터가 다른 위치에 있으면 `prepare --data-root "데이터 폴더"`를 사용하세요.
이미 생성된 manifest를 다른 PC로 옮겼다면 `train`과 `evaluate`에 `--data-root "Original과 Depth를 담은 폴더"`를 지정할 수 있습니다.

## 학습 결과 확인과 이어서 학습

```powershell
.venv\Scripts\python.exe -m wave_height.train --config configs/rtx3070.json --resume runs/rtx3070/last.pt
```

중단된 실행은 원래 설정과 manifest를 사용해 epoch 단위로 이어갑니다. 중단된 epoch는 처음부터 다시 실행합니다.
설정을 바꿔 새로 학습할 때는 `--output runs/experiment2`처럼 새 폴더를 지정하세요.
완료한 epoch 수를 넘어 학습을 연장하는 기능은 포함하지 않습니다.

| 결과 파일 | 의미 |
|---|---|
| `runs/rtx3070/best.pt` | 검증 MAE가 가장 작은 모델 |
| `runs/rtx3070/last.pt` | 마지막 epoch의 모델·optimizer·scheduler |
| `runs/rtx3070/history.jsonl` | epoch별 학습·검증 오차와 시간 |
| `runs/rtx3070/test_metrics.json` | 최종 모델과 평균 높이 맵의 테스트 오차 |
| `runs/rtx3070/test_metrics.jpg` | RGB / 정답 / 예측 / 학습 평균 / 절대 오차 비교 |

MAE와 RMSE는 `G/255` 단위이며 미터 단위가 아닙니다.
`mae_improvement_over_mean_percent`가 양수면 모델이 평균 높이 맵보다 MAE를 줄였다는 뜻입니다.
파도 형태 복원은 `gradient_mae`와 비교 이미지도 함께 보세요.
평가는 기본 학습 해상도에서 수행하며, 이미지별 보정이나 정답에 맞춘 값 정렬은 하지 않습니다.

## 새 RGB 이미지·영상에서 추론

이미지 1장 또는 이미지 폴더를 입력할 수 있습니다.

```powershell
.venv\Scripts\python.exe -m wave_height.predict --checkpoint runs/rtx3070/best.pt --input new_frame.png --output outputs/new_frame --device cuda
.venv\Scripts\python.exe -m wave_height.predict --checkpoint runs/rtx3070/best.pt --input new_frames --output outputs/new_frames --device cuda
```

동영상은 OpenCV를 추가 설치한 다음 실행합니다. `--every 5`면 5프레임마다 한 장씩 예측합니다.

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-video.txt
.venv\Scripts\python.exe -m wave_height.predict --checkpoint runs/rtx3070/best.pt --input new_video.mp4 --output outputs/new_video --device cuda --every 5
```

출력은 입력 이미지 크기로 복원됩니다. 복원은 보간이며 학습 해상도보다 세밀한 정보를 새로 만드는 것은 아닙니다.
영상도 프레임마다 독립적으로 예측하므로 시간적 일관성을 직접 학습하지는 않습니다.
영상에서는 프레임별 결과 파일을 저장하며 `video_info.json`에 원본 FPS와 프레임 간격을 기록합니다.
기존 결과가 섞이지 않도록 출력 폴더는 비어 있어야 합니다.

| 출력 | 용도 |
|---|---|
| `*_height_normalized.npy` | float32 `[0,1]` 높이 값. 후속 분석에 권장 |
| `*_height16.png` | 픽셀값 / 65535가 정규화된 높이 |
| `*_height_green.png` | 원본 타깃과 같은 G 채널 표현 |
| `*_preview.jpg` | RGB와 예측 높이의 비교 |
| `encoding.json` | 타깃 의미, 단위, 사용한 체크포인트 기록 |

16-bit PNG 출력은 저장 정밀도에 관한 것입니다. 기존 8-bit 학습 타깃의 물리 정밀도가 높아지는 것은 아닙니다.
모든 프레임의 시각화는 같은 `[0,1]` 범위를 사용합니다.

나중에 **모든 이미지에 공통인 선형 변환** `G=0 → h_min`, `G=255 → h_max`를 확인하면
추론 명령에 `--height-range h_min h_max`를 추가할 수 있습니다. 이때 `*_height_world_units.npy`도 저장됩니다.
단위는 사용자가 지정한 범위의 단위를 따르며, 현재는 이 옵션에 임의의 값을 넣지 마세요.
프레임마다 높이를 따로 정규화해 저장했던 경우라면 PNG만으로 공통 물리 높이를 복원할 수 없습니다.

## 구현 검증 상태

현재 Mac의 CPU에서 G 채널·0값 보존, 분할 간격, 학습 평균의 데이터 범위, 페어 좌우 반전,
절대값 회귀 손실, MAE 집계, 홀수 해상도의 forward/backward, 결과 저장을 테스트했습니다.

```text
python -m unittest discover -s tests -v
```

소량 데이터로 2 epochs 학습, 체크포인트 재로딩, 평가, 이미지 추론도 실행했습니다.
3프레임 영상에서 간격 2로 2개 결과를 저장하는 과정과, 첫 epoch 뒤 중단한 학습을 3번째 epoch까지 재개하는 과정도 확인했습니다.
검증 환경은 macOS CPU, PyTorch 2.14.1 / torchvision 0.29.1 / NumPy 2.3.5 / Pillow 12.3.0 / OpenCV 4.14.0입니다.
상세 확인 항목은 `artifacts/verification.json`에 기록했습니다.
`runs/smoke/`의 체크포인트는 사전학습 없이 4장으로 실행한 **동작 확인용**입니다.
이를 완성된 파도 예측 모델이나 정확도 근거로 사용하면 안 됩니다. 전체 데이터 학습과 CUDA 실행은 RTX 3070 PC에서 진행해야 합니다.

참고 구현 문서: [torchvision ResNet-18](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html).
Unity의 일반적인 camera depth와 현재 수면 높이 라벨은 서로 다른 값입니다.
[Unity camera depth 문서](https://docs.unity.com/en-us/engine/6000.0/manual/cameras/camera-output/sl-camera-depth-texture)는 원래 depth texture의 용도를 설명합니다.

