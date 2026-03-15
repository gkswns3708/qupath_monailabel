HoverNet Inference Threshold 파라미터 가이드
3x3 (Fast) 전용
파라미터	기본값	낮추면	높이면
min_size	64 px	더 작은 핵도 검출 → 노이즈/파편 증가	작은 핵 무시 → 깔끔하지만 miss 증가
min_hole	64 px	핵 내부 작은 구멍도 유지 → 울퉁불퉁한 mask	큰 구멍만 유지 → 매끄러운 mask
3x3/5x5 공통
파라미터	기본값	낮추면	높이면
min_poly_area	30 px²	아주 작은 polygon도 포함 → 노이즈 ↑	작은 핵 걸러짐 → 큰 핵만 남음
max_poly_area	16384 px² (128×128)	큰 polygon 제거 → 대형 핵/클러스터 제외	더 큰 구조도 허용
buffer_distance	0.5 px	contour가 핵 경계 안쪽 → 실제보다 작게	contour 확장 → 핵이 더 크게 표시
5x5 (Original) 전용 — watershed 핵심 파라미터
파라미터	기본값	낮추면	높이면
marker_threshold	0.4	더 많은 영역을 핵으로 인식 → over-segmentation (하나의 핵이 여러 개로 분리)	확실한 핵만 인식 → under-segmentation (인접 핵 병합, 검출 수 ↓)
sobel_kernel_size	21	경계 감지가 예민 → 작은 변화에도 경계 인식 → 과분리	경계가 부드러워짐 → 인접 핵 경계 흐려짐 → 병합 경향
marker_radius	2	marker가 점처럼 작음 → seed가 정확하지만 약함	marker가 넓게 퍼짐 → 인접 핵이 하나로 합쳐질 수 있음
실전 조절 팁
핵이 너무 많이 쪼개질 때 (over-segmentation): marker_threshold ↑ (0.4→0.5~0.6)
인접 핵이 하나로 합쳐질 때 (under-segmentation): marker_threshold ↓ (0.4→0.3)
잡은 노이즈가 많을 때: min_poly_area ↑ 또는 min_size ↑
큰 세포 클러스터가 잡힐 때: max_poly_area ↓