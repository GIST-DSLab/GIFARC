import os

def get_files_by_extension(directory, extension):
    """
    지정된 디렉토리에서 주어진 확장자를 가진 파일들의 파일명을 반환합니다.
    
    Args:
    directory (str): 파일들을 검색할 디렉토리 경로
    extension (str): 찾고자 하는 파일의 확장자 (예: '.gif')
    
    Returns:
    list: 확장자가 일치하는 파일 이름들
    """
    file_names = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(extension.lower()):
                file_names.append(os.path.splitext(file)[0])  # 확장자를 제외한 파일 이름
    return file_names


def write_batches_to_file(file_names, output_dir, batch_size=50):
    """
    파일 이름들을 여러 배치로 나누어 지정된 디렉토리에 batch 파일로 저장합니다.
    
    Args:
    file_names (list): 저장할 파일 이름 목록
    output_dir (str): 배치 파일들을 저장할 디렉토리 경로
    batch_size (int): 하나의 배치 파일에 포함될 파일 이름 수 (기본값: 50)
    """
    # 배치 디렉토리 존재하지 않으면 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 배치 파일을 나누어서 저장
    for i in range(0, len(file_names), batch_size):
        batch_name = f"batch_{(i // batch_size) + 1}.txt"
        batch_path = os.path.join(output_dir, batch_name)
        
        # 배치 파일에 파일 이름을 작성
        with open(batch_path, 'w') as f:
            for file_name in file_names[i:i + batch_size]:
                f.write(f"{file_name},\n")
        print(f"{batch_name} 파일이 {batch_size}개의 항목으로 저장되었습니다.")


def process_files(directory, extension, output_dir, batch_size=50):
    """
    지정된 디렉토리에서 특정 확장자를 가진 파일들을 처리하고 배치 파일로 저장합니다.
    
    Args:
    directory (str): 파일들을 검색할 디렉토리 경로
    extension (str): 필터링할 파일의 확장자
    output_dir (str): 배치 파일들을 저장할 디렉토리 경로
    batch_size (int): 배치 파일에 저장할 파일 이름 수 (기본값: 50)
    """
    # 해당 확장자를 가진 파일 이름 가져오기
    file_names = get_files_by_extension(directory, extension)
    
    # 배치 파일로 나누어 저장
    write_batches_to_file(file_names, output_dir, batch_size)

