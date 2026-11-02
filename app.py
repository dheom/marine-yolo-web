import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os

# ==============================================================================
# [구동 안내서 반영] 1. 웹페이지 레이아웃 및 기본 설정 (3p 실행환경 사양 적용)
# ==============================================================================
st.set_page_config(
    page_title="AI 항로표지 객체 인식 서비스", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# 메인 타이틀 및 양식 기준 안내 문구 표출 (4p 제약사항 반영)
st.title("⚓ AI 항로표지 객체 인식 및 탐지 웹서비스")
st.caption(
    "본 시연 버전은 컴퓨터 비전 모델을 중심으로 구성되어 별도의 LLM API 키를 사용하지 않으며, "
    "YOLO 모델의 가중치 파일과 실행 환경을 통해 AI 추론을 수행합니다."
)

# ==============================================================================
# [구동 안내서 반영] 2. AI 모델 로드 설정 (3p 및 4p 모델 설정 적용)
# ==============================================================================
@st.cache_resource
def load_yolo_model():
    """안내서에 명시된 Ultralytics YOLO26n 기반 가중치 파일(yolo26n.pt) 로드"""
    # 최초 실행 시 파일이 없으면 인터넷에서 자동으로 다운로드됩니다.
    return YOLO("yolo26n.pt")

try:
    model = load_yolo_model()
    st.sidebar.success("✅ AI 모델 로드 완료 (yolo26n.pt)")
except Exception as e:
    st.sidebar.error(f"❌ AI 모델 로드 실패: {e}")

# ==============================================================================
# [구동 안내서 반영] 3. 사이드바 입력 및 제어 영역 (3p 설치·실행 절차 반영)
# ==============================================================================
st.sidebar.header("📥 기능 선택 및 설정")

# 영상 입력 방식 선택 (라디오 버튼)
input_option = st.sidebar.radio(
    "영상 입력 방식을 선택하세요:",
    ("시연용 해양 영상 파일 업로드", "사용자 웹캠 실시간 입력")
)

st.sidebar.markdown("---")
# 알림 및 탐지 경고 메시지가 표시될 사이드바 공간 확보
alert_placeholder = st.sidebar.empty() 

# 메인 화면에 실시간 영상을 출력할 윈도우 영역 생성
frame_window = st.image([], use_container_width=True)

# ==============================================================================
# [구동 안내서 반영] 4. 기능 사용 순서 및 추론 프로세스 (4p 및 5p 구현 범위 적용)
# ==============================================================================

# --- [옵션 A] 시연용 해양 영상 파일 처리 ---
if input_option == "시연용 해양 영상 파일 업로드":
    uploaded_file = st.file_uploader(
        "해양 동영상 파일을 업로드하세요. (지원 형식: mp4, avi, mov)", 
        type=["mp4", "avi", "mov"]
    )
    
    if uploaded_file is not None:
        # 안전한 분석을 위한 파일 실행 버튼 처리
        if st.button("▶️ 영상 분석 시작"):
            alert_placeholder.info("정상 상태: 분석 진행 중...")
            
            # 업로드 파일을 안전하게 임시 경로에 임베딩 후 로드
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                tfile.write(uploaded_file.read())
                temp_path = tfile.name
            
            cap = cv2.VideoCapture(temp_path)
            
            try:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break  # 영상이 끝나면 루프 종료
                    
                    # 브라우저 출력 품질 향상을 위한 컬러 모델 변환 (BGR -> RGB)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # YOLO 모델 실시간 추론 실행
                    results = model(frame_rgb, stream=True)
                    
                    annotated_frame = frame_rgb
                    object_detected = False
                    
                    for r in results:
                        # 인식 경계 상자(Bounding Box) 시각화 처리
                        annotated_frame = r.plot()
                        if len(r.boxes) > 0:
                            object_detected = True
                    
                    # [5p MVP 기능 범위 만족] 탐지 결과에 따른 웹사이트 실시간 경고 알림 표출
                    if object_detected:
                        alert_placeholder.warning("🚨 경고: 항로표지(또는 사물) 탐지됨!")
                    else:
                        alert_placeholder.info("정상 상태: 분석 진행 중...")
                    
                    # 최적화된 프레임 출력 업데이트
                    frame_window.image(annotated_frame)
                    
            finally:
                # 메모리 누수 및 리소스 꼬임 방지를 위한 비디오 자원 해제 최적화
                cap.release()
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
            alert_placeholder.success("🏁 완료: 영상 분석이 종료되었습니다.")
            st.success("영상 파일 분석 프로세스가 성공적으로 끝났습니다.")

# --- [옵션 B] 사용자 웹캠 실시간 입력 처리 ---
elif input_option == "사용자 웹캠 실시간 입력":
    st.info("💡 아래 '웹캠 활성화' 체크박스를 켜면 브라우저의 카메라 권한 허용 후 실시간 분석을 수행합니다.")
    run_webcam = st.checkbox("🎥 웹캠 활성화")
    
    if run_webcam:
        # 시스템 기본 카메라 자원 로드 (0번 카메라 기본값 설정)
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("카메라 장치를 인식할 수 없습니다. 웹캠 장치 연결 혹은 브라우저 권한을 확인하세요.")
        
        try:
            while run_webcam and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    alert_placeholder.error("카메라 신호를 유실했습니다.")
                    break
                    
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = model(frame_rgb, stream=True)
                
                annotated_frame = frame_rgb
                object_detected = False
                
                for r in results:
                    annotated_frame = r.plot()
                    if len(r.boxes) > 0:
                        object_detected = True
                        
                if object_detected:
                    alert_placeholder.warning("🚨 경고: 실시간 항로표지 탐지됨!")
                else:
                    alert_placeholder.info("정상 상태: 실시간 스캔 중...")
                    
                frame_window.image(annotated_frame)
        finally:
            # 체크박스 해제 또는 예외 발생 시 웹캠 하드웨어 권한 반환 최적화
            cap.release()
            frame_window.empty()
            alert_placeholder.info("웹캠 연결이 종료되었습니다.")
